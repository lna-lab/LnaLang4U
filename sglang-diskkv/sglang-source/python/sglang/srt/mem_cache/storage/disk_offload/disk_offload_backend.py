import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

try:
    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
        PoolTransferResult,
    )
except ImportError:
    class PoolName(str, Enum):
        KV = "kv"
        MAMBA = "mamba"
        SWA = "swa"
        INDEXER = "indexer"

        def __str__(self) -> str:
            return self.value

    class PoolHitPolicy(str, Enum):
        ALL_PAGES = "all_pages"
        TRAILING_PAGES = "trailing_pages"

    @dataclass
    class PoolTransfer:
        name: PoolName
        host_indices: Optional[torch.Tensor] = None
        device_indices: Optional[torch.Tensor] = None
        keys: Optional[List[str]] = None
        hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
        nodes_to_load: Optional[List[Any]] = None

    @dataclass
    class PoolTransferResult:
        kv_hit_pages: int
        extra_pool_hit_pages: dict[str, int]

        @classmethod
        def empty(cls) -> "PoolTransferResult":
            return cls(0, {})

        def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
            self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

        def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
            self.extra_pool_hit_pages.update(
                {name: sum(rs) for name, rs in results.items()}
            )

logger = logging.getLogger(__name__)


class DiskOffloadBackend(HiCacheStorage):
    """L3 storage backend that offloads KV cache pages to local SSD.

    Mirrors ds4-server's ``--kv-disk-dir`` + ``--kv-disk-space-mb`` design:
    pages are persisted as individual ``.pt`` files under ``disk_offload_dir``,
    and an LRU eviction policy keeps total disk usage within ``max_disk_space_mb``.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        disk_offload_dir: str = "/tmp/sglang-diskkv",
        max_disk_space_mb: int = 65536,
    ):
        extra_config = storage_config.extra_config or {}
        disk_offload_dir = (
            extra_config.get("disk_offload_dir")
            or os.environ.get("SGLANG_DISK_OFFLOAD_DIR")
            or disk_offload_dir
        )
        max_disk_space_mb = int(
            extra_config.get("max_disk_space_mb")
            or os.environ.get("SGLANG_DISK_OFFLOAD_MAX_SPACE_MB")
            or max_disk_space_mb
        )

        self.disk_offload_dir = disk_offload_dir
        self.max_disk_bytes = max_disk_space_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._pages_dir = os.path.join(disk_offload_dir, "pages")
        self._index_path = os.path.join(disk_offload_dir, "index.json")
        self._index: Dict[str, Dict[str, Any]] = {}
        self._total_bytes = 0
        self._dirty = False

        os.makedirs(self._pages_dir, exist_ok=True)
        self._load_index()
        logger.info(
            "DiskOffloadBackend: dir=%s max_space=%d MiB pages=%d used=%.2f MiB",
            disk_offload_dir,
            max_disk_space_mb,
            len(self._index),
            self._total_bytes / (1024 * 1024),
        )

    # ------------------------------------------------------------------ #
    # Index persistence
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        if not os.path.exists(self._index_path):
            self._index = {}
            self._total_bytes = 0
            return
        try:
            with open(self._index_path) as f:
                data = json.load(f)
            self._index = data.get("pages", {})
            self._total_bytes = data.get("total_bytes", 0)
        except (json.JSONDecodeError, KeyError):
            logger.warning("DiskOffloadBackend: corrupted index, rebuilding...")
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        self._total_bytes = 0
        if not os.path.isdir(self._pages_dir):
            return
        for name in os.listdir(self._pages_dir):
            if not name.endswith(".pt"):
                continue
            key = name[:-3]
            path = os.path.join(self._pages_dir, name)
            try:
                sz = os.path.getsize(path)
                mtime = os.path.getmtime(path)
                self._index[key] = {"size": sz, "last_access": mtime}
                self._total_bytes += sz
            except OSError:
                pass

    def _save_index(self) -> None:
        data = {
            "pages": self._index,
            "total_bytes": self._total_bytes,
        }
        tmp = self._index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self._index_path)

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _flush_index(self) -> None:
        if self._dirty:
            self._save_index()
            self._dirty = False

    # ------------------------------------------------------------------ #
    # Budget enforcement (LRU eviction)
    # ------------------------------------------------------------------ #

    def _enforce_budget(self, needed_bytes: int) -> None:
        """Evict least-recently-accessed pages until ``self._total_bytes + needed_bytes``
        fits within ``self.max_disk_bytes``."""
        if self.max_disk_bytes <= 0:
            return
        target = self.max_disk_bytes - needed_bytes
        if self._total_bytes <= target:
            return

        evictable = sorted(
            [(v["last_access"], k) for k, v in self._index.items()],
            key=lambda x: x[0],
        )
        for _ts, key in evictable:
            if self._total_bytes <= target:
                break
            self._evict_one(key)

    def _evict_one(self, key: str) -> None:
        info = self._index.pop(key, None)
        if info is None:
            return
        self._total_bytes -= info["size"]
        path = os.path.join(self._pages_dir, f"{key}.pt")
        try:
            os.remove(path)
        except OSError:
            pass
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # Key helpers
    # ------------------------------------------------------------------ #

    def _page_path(self, key: str) -> str:
        return os.path.join(self._pages_dir, f"{key}.pt")

    def _component_key(self, key: str, pool_name: str = PoolName.KV) -> str:
        pool_value = pool_name.value if hasattr(pool_name, "value") else str(pool_name)
        return key if pool_value == "kv" else f"{key}.{pool_value}"

    def _touch_access(self, key: str) -> None:
        info = self._index.get(key)
        if info is not None:
            info["last_access"] = time.time()
            self._mark_dirty()

    # ------------------------------------------------------------------ #
    # HiCacheStorage interface
    # ------------------------------------------------------------------ #

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        path = self._page_path(key)
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            self._touch_access(key)
            return data
        except FileNotFoundError:
            logger.warning("DiskOffloadBackend: page not found: %s", key)
            return None
        except Exception as e:
            logger.error("DiskOffloadBackend: failed to load page %s: %s", key, e)
            return None

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if value is None:
            return False
        path = self._page_path(key)
        with self._lock:
            if key in self._index:
                self._touch_access(key)
                self._flush_index()
                return True
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(value, path)
                sz = os.path.getsize(path)
                self._enforce_budget(sz)
                self._index[key] = {"size": sz, "last_access": time.time()}
                self._total_bytes += sz
                self._mark_dirty()
                self._flush_index()
                return True
            except Exception as e:
                logger.error("DiskOffloadBackend: failed to save page %s: %s", key, e)
                return False

    def exists(self, key: str) -> bool:
        return key in self._index and os.path.exists(self._page_path(key))

    def batch_exists(
        self, keys: List[str], extra_info: Optional[Any] = None
    ) -> int:
        """Return the number of consecutive existing keys from the start."""
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Optional[torch.Tensor]]:
        return [self.get(k) for k in keys]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if values is None:
            return False
        ok = True
        for key, value in zip(keys, values):
            ok = self.set(key, value) and ok
        self._flush_index()
        return ok

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = next(
            (i for i, key in enumerate(keys) if not self.exists(key)), len(keys)
        )
        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name

            def has_component(page_idx: int) -> bool:
                return self.exists(self._component_key(keys[page_idx], name))

            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i)), kv_pages
                )
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _batch_io_v2(self, transfers: List[PoolTransfer], write: bool):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if host_indices is None or host_indices.numel() != len(keys) * page_size:
                results[transfer.name] = [False] * len(keys)
                continue

            pool_results = []
            for i, key in enumerate(keys):
                storage_key = self._component_key(key, transfer.name)
                host_offset = host_indices[i * page_size].item()
                if write:
                    pool_results.append(
                        self.set(storage_key, host_pool.get_data_page(host_offset))
                    )
                else:
                    data = self.get(storage_key)
                    if data is None:
                        pool_results.append(False)
                    else:
                        host_pool.set_from_flat_data_page(host_offset, data)
                        pool_results.append(True)
            results[transfer.name] = pool_results
        self._flush_index()
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, write=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, write=True)

    def clear(self) -> None:
        with self._lock:
            for key in list(self._index.keys()):
                self._evict_one(key)
            self._flush_index()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        return {
            "disk_offload_dir": self.disk_offload_dir,
            "max_disk_bytes": self.max_disk_bytes,
            "total_bytes": self._total_bytes,
            "total_mb": round(self._total_bytes / (1024 * 1024), 2),
            "page_count": len(self._index),
        }
