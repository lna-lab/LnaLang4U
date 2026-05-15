from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class DeepSeekV4TokenToKVPoolHost(HostKVCache):
    """Direct-I/O host pool for DeepSeek-V4 compressed KV cache.

    DeepSeek-V4 does not expose a plain MLA-style ``kv_buffer``.  Its logical
    full-token cache is split into SWA pages, c4/c128 compressed pages, c4
    indexer pages, and compressor state rows.  The standard HiCache MLA kernels
    cannot move this layout, so this host pool stores one complete DSV4 full
    page per HiCache page and copies the underlying 2D page buffers directly.

    This pool is intentionally direct-only.  Launch with:
    ``--hicache-io-backend direct --hicache-mem-layout page_first``.
    ServerArgs will normalize that layout to ``page_first_direct``.
    """

    device_pool: "DeepSeekV4TokenToKVPool"

    def __init__(
        self,
        device_pool: "DeepSeekV4TokenToKVPool",
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        if layout == "page_first":
            layout = "page_first_direct"
        if layout != "page_first_direct":
            raise ValueError(
                "DeepSeek-V4 HiCache requires page_first_direct host layout. "
                "Use --hicache-io-backend direct --hicache-mem-layout page_first."
            )
        if page_size != device_pool.page_size:
            raise ValueError(
                f"DeepSeek-V4 HiCache page_size mismatch: host={page_size}, "
                f"device={device_pool.page_size}."
            )
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _kv_page_bytes(self, pool) -> int:
        if not hasattr(pool, "bytes_per_page_padded"):
            pool.create_buffer(num_pages=1)
        return int(pool.bytes_per_page_padded)

    def _indexer_page_bytes(self, pool) -> int:
        num_scales_per_token = pool.index_head_dim // pool.quant_block_size
        return int(pool.page_size * (pool.index_head_dim + num_scales_per_token * 4))

    def _state_rows_per_page(self, state_pool, ratio: int) -> int:
        return max(1, int(state_pool.ring_size) // int(ratio))

    def _state_page_shape(self, state_pool, ratio: int) -> tuple[int, int]:
        rows = self._state_rows_per_page(state_pool, ratio)
        width = int(state_pool.kv_score_buffer.kv_score.shape[-1])
        return rows, width

    def get_size_per_token(self) -> int:
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.device_pool.swa_kv_pool.get_bytes_per_token()

        bytes_per_page = 0
        bytes_per_page += self.layer_num * self._kv_page_bytes(
            self.device_pool.swa_kv_pool
        )
        bytes_per_page += self.device_pool.c4_kv_pool.layer_num * self._kv_page_bytes(
            self.device_pool.c4_kv_pool
        )
        bytes_per_page += self.device_pool.c128_kv_pool.layer_num * self._kv_page_bytes(
            self.device_pool.c128_kv_pool
        )
        bytes_per_page += self.device_pool.c4_indexer_kv_pool.layer_num * (
            self._indexer_page_bytes(self.device_pool.c4_indexer_kv_pool)
        )

        for layer_id, item in enumerate(self.device_pool.layer_mapping):
            ratio = item.compress_ratio
            if ratio == 0:
                continue
            state_pool = self.device_pool.compress_state_pools[layer_id]
            rows, width = self._state_page_shape(state_pool, ratio)
            bytes_per_page += rows * width * state_pool.kv_score_buffer.kv_score.dtype.itemsize
            if ratio == 4:
                indexer_state_pool = self.device_pool.indexer_compress_state_pools[
                    layer_id
                ]
                rows, width = self._state_page_shape(indexer_state_pool, ratio)
                bytes_per_page += (
                    rows
                    * width
                    * indexer_state_pool.kv_score_buffer.kv_score.dtype.itemsize
                )

        self.page_bytes = int(bytes_per_page)
        return (self.page_bytes + self.page_size - 1) // self.page_size

    def get_ksize_per_token(self) -> int:
        return self.get_size_per_token()

    # ------------------------------------------------------------------
    # Host allocation
    # ------------------------------------------------------------------

    def _alloc(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(
            shape,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def init_kv_buffer(self):
        swa = self.device_pool.swa_kv_pool
        c4 = self.device_pool.c4_kv_pool
        c128 = self.device_pool.c128_kv_pool
        indexer = self.device_pool.c4_indexer_kv_pool

        self.swa_page_bytes = self._kv_page_bytes(swa)
        self.c4_page_bytes = self._kv_page_bytes(c4)
        self.c128_page_bytes = self._kv_page_bytes(c128)
        self.indexer_page_bytes = self._indexer_page_bytes(indexer)

        self.swa_buffer = [
            self._alloc((self.page_num, self.swa_page_bytes), swa.store_dtype)
            for _ in range(swa.layer_num)
        ]
        self.c4_buffer = [
            self._alloc((self.page_num, self.c4_page_bytes), c4.store_dtype)
            for _ in range(c4.layer_num)
        ]
        self.c128_buffer = [
            self._alloc((self.page_num, self.c128_page_bytes), c128.store_dtype)
            for _ in range(c128.layer_num)
        ]
        self.indexer_buffer = [
            self._alloc(
                (self.page_num, self.indexer_page_bytes),
                indexer.index_k_with_scale_buffer_dtype,
            )
            for _ in range(indexer.layer_num)
        ]

        self.attn_state_buffer: list[torch.Tensor | None] = []
        self.indexer_state_buffer: list[torch.Tensor | None] = []
        for layer_id, item in enumerate(self.device_pool.layer_mapping):
            ratio = item.compress_ratio
            if ratio == 0:
                self.attn_state_buffer.append(None)
                self.indexer_state_buffer.append(None)
                continue

            state_pool = self.device_pool.compress_state_pools[layer_id]
            rows, width = self._state_page_shape(state_pool, ratio)
            self.attn_state_buffer.append(
                self._alloc(
                    (self.page_num, rows, width),
                    state_pool.kv_score_buffer.kv_score.dtype,
                )
            )

            if ratio == 4:
                indexer_state_pool = self.device_pool.indexer_compress_state_pools[
                    layer_id
                ]
                rows, width = self._state_page_shape(indexer_state_pool, ratio)
                self.indexer_state_buffer.append(
                    self._alloc(
                        (self.page_num, rows, width),
                        indexer_state_pool.kv_score_buffer.kv_score.dtype,
                    )
                )
            else:
                self.indexer_state_buffer.append(None)

        logger.info(
            "Initialized DeepSeek-V4 host pool: pages=%s page_bytes=%.2f MiB "
            "swa_layers=%s c4_layers=%s c128_layers=%s indexer_layers=%s",
            self.page_num,
            self.page_bytes / (1024 * 1024),
            len(self.swa_buffer),
            len(self.c4_buffer),
            len(self.c128_buffer),
            len(self.indexer_buffer),
        )
        return self.swa_buffer

    # ------------------------------------------------------------------
    # Index translation
    # ------------------------------------------------------------------

    def _page_starts(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                "host_indices and device_indices must have the same length for DSV4."
            )
        if host_indices.numel() % self.page_size != 0:
            raise ValueError("DeepSeek-V4 HiCache transfers must be page aligned.")
        host_cpu = host_indices.cpu()
        device_cpu = device_indices.cpu()
        host_starts = host_cpu.reshape(-1, self.page_size)[:, 0].to(torch.long)
        device_starts = device_cpu.reshape(-1, self.page_size)[:, 0].to(torch.long)
        return host_starts, device_starts

    def _logical_pages(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        host_starts, device_starts = self._page_starts(host_indices, device_indices)
        host_pages = host_starts // self.page_size
        full_pages = device_starts // self.page_size

        starts_dev = device_starts.to(self.device_pool.swa_kv_pool.device)
        swa_starts = self.device_pool.translate_loc_from_full_to_swa(starts_dev).cpu()
        swa_pages = swa_starts.to(torch.long) // self.device_pool.swa_page_size
        return host_pages, full_pages, swa_pages, device_starts

    def _state_rows(self, swa_pages: torch.Tensor, state_pool, ratio: int) -> torch.Tensor:
        rows_per_page = self._state_rows_per_page(state_pool, ratio)
        offsets = torch.arange(rows_per_page, dtype=torch.long)
        return swa_pages[:, None] * rows_per_page + offsets[None, :]

    # ------------------------------------------------------------------
    # Tensor copies
    # ------------------------------------------------------------------

    def _backup_pages(
        self,
        host_buf: torch.Tensor,
        device_buf: torch.Tensor,
        host_pages: torch.Tensor,
        device_pages: torch.Tensor,
    ) -> None:
        src = device_buf.index_select(0, device_pages.to(device_buf.device))
        host_buf[host_pages] = src.to(self.device, non_blocking=True)

    def _load_pages(
        self,
        host_buf: torch.Tensor,
        device_buf: torch.Tensor,
        host_pages: torch.Tensor,
        device_pages: torch.Tensor,
    ) -> None:
        src = host_buf[host_pages].to(device_buf.device, non_blocking=True)
        device_buf.index_copy_(0, device_pages.to(device_buf.device), src)

    def _backup_state(
        self,
        host_buf: torch.Tensor,
        state_pool,
        ratio: int,
        host_pages: torch.Tensor,
        swa_pages: torch.Tensor,
    ) -> None:
        rows = self._state_rows(swa_pages, state_pool, ratio)
        state = state_pool.kv_score_buffer.kv_score
        src = state.index_select(0, rows.flatten().to(state.device))
        src = src.reshape(rows.shape[0], rows.shape[1], state.shape[-1])
        host_buf[host_pages] = src.to(self.device, non_blocking=True)

    def _load_state(
        self,
        host_buf: torch.Tensor,
        state_pool,
        ratio: int,
        host_pages: torch.Tensor,
        swa_pages: torch.Tensor,
    ) -> None:
        rows = self._state_rows(swa_pages, state_pool, ratio)
        state = state_pool.kv_score_buffer.kv_score
        src = host_buf[host_pages].to(state.device, non_blocking=True)
        state.index_copy_(0, rows.flatten().to(state.device), src.reshape(-1, state.shape[-1]))

    # ------------------------------------------------------------------
    # Device <-> host transfer
    # ------------------------------------------------------------------

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        if io_backend != "direct":
            raise ValueError("DeepSeek-V4 HiCache currently supports only direct IO.")

        host_pages, full_pages, swa_pages, _device_starts = self._logical_pages(
            host_indices, device_indices
        )
        c4_pages = full_pages
        c128_pages = full_pages

        for layer_id in range(device_pool.swa_kv_pool.layer_num):
            self._backup_pages(
                self.swa_buffer[layer_id],
                device_pool.swa_kv_pool.kv_buffer[layer_id],
                host_pages,
                swa_pages,
            )

            item = device_pool.layer_mapping[layer_id]
            ratio = item.compress_ratio
            if ratio == 4:
                local_layer = item.compress_layer_id
                self._backup_pages(
                    self.c4_buffer[local_layer],
                    device_pool.c4_kv_pool.kv_buffer[local_layer],
                    host_pages,
                    c4_pages,
                )
                self._backup_pages(
                    self.indexer_buffer[local_layer],
                    device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer[
                        local_layer
                    ],
                    host_pages,
                    c4_pages,
                )
                self._backup_state(
                    self.attn_state_buffer[layer_id],
                    device_pool.compress_state_pools[layer_id],
                    ratio,
                    host_pages,
                    swa_pages,
                )
                self._backup_state(
                    self.indexer_state_buffer[layer_id],
                    device_pool.indexer_compress_state_pools[layer_id],
                    ratio,
                    host_pages,
                    swa_pages,
                )
            elif ratio == 128:
                local_layer = item.compress_layer_id
                self._backup_pages(
                    self.c128_buffer[local_layer],
                    device_pool.c128_kv_pool.kv_buffer[local_layer],
                    host_pages,
                    c128_pages,
                )
                self._backup_state(
                    self.attn_state_buffer[layer_id],
                    device_pool.compress_state_pools[layer_id],
                    ratio,
                    host_pages,
                    swa_pages,
                )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        if io_backend != "direct":
            raise ValueError("DeepSeek-V4 HiCache currently supports only direct IO.")

        host_pages, full_pages, swa_pages, _device_starts = self._logical_pages(
            host_indices, device_indices
        )
        self._load_pages(
            self.swa_buffer[layer_id],
            device_pool.swa_kv_pool.kv_buffer[layer_id],
            host_pages,
            swa_pages,
        )

        item = device_pool.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio == 4:
            local_layer = item.compress_layer_id
            self._load_pages(
                self.c4_buffer[local_layer],
                device_pool.c4_kv_pool.kv_buffer[local_layer],
                host_pages,
                full_pages,
            )
            self._load_pages(
                self.indexer_buffer[local_layer],
                device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer[local_layer],
                host_pages,
                full_pages,
            )
            self._load_state(
                self.attn_state_buffer[layer_id],
                device_pool.compress_state_pools[layer_id],
                ratio,
                host_pages,
                swa_pages,
            )
            self._load_state(
                self.indexer_state_buffer[layer_id],
                device_pool.indexer_compress_state_pools[layer_id],
                ratio,
                host_pages,
                swa_pages,
            )
        elif ratio == 128:
            local_layer = item.compress_layer_id
            self._load_pages(
                self.c128_buffer[local_layer],
                device_pool.c128_kv_pool.kv_buffer[local_layer],
                host_pages,
                full_pages,
            )
            self._load_state(
                self.attn_state_buffer[layer_id],
                device_pool.compress_state_pools[layer_id],
                ratio,
                host_pages,
                swa_pages,
            )

    # ------------------------------------------------------------------
    # Storage page representation
    # ------------------------------------------------------------------

    def _component_payload(self, host_page: int) -> dict[str, Any]:
        return {
            "swa": [buf[host_page].clone() for buf in self.swa_buffer],
            "c4": [buf[host_page].clone() for buf in self.c4_buffer],
            "c128": [buf[host_page].clone() for buf in self.c128_buffer],
            "indexer": [buf[host_page].clone() for buf in self.indexer_buffer],
            "attn_state": [
                None if buf is None else buf[host_page].clone()
                for buf in self.attn_state_buffer
            ],
            "indexer_state": [
                None if buf is None else buf[host_page].clone()
                for buf in self.indexer_state_buffer
            ],
        }

    def get_data_page(self, index, flat: bool = True):
        host_page = int(index) // self.page_size
        return self._component_payload(host_page)

    def get_dummy_flat_data_page(self):
        return None

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        if data_page is None:
            return
        host_page = int(index) // self.page_size

        for dst, src in zip(self.swa_buffer, data_page["swa"]):
            dst[host_page].copy_(src)
        for dst, src in zip(self.c4_buffer, data_page["c4"]):
            dst[host_page].copy_(src)
        for dst, src in zip(self.c128_buffer, data_page["c128"]):
            dst[host_page].copy_(src)
        for dst, src in zip(self.indexer_buffer, data_page["indexer"]):
            dst[host_page].copy_(src)
        for dst, src in zip(self.attn_state_buffer, data_page["attn_state"]):
            if dst is not None and src is not None:
                dst[host_page].copy_(src)
        for dst, src in zip(self.indexer_state_buffer, data_page["indexer_state"]):
            if dst is not None and src is not None:
                dst[host_page].copy_(src)

    def get_page_buffer_meta(self, indices):
        raise NotImplementedError(
            "DeepSeek-V4 host pool stores heterogeneous page payloads and does not "
            "support zero-copy storage backends. Use disk_offload with generic IO."
        )
