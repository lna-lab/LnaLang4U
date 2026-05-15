# Architecture

## Problem

DeepSeek-V4-Flash is a 284B-parameter MoE model with MLA (Multi-head Latent Attention) and DeepSeek Sparse Attention. Its KV cache uses compressed representations (c4 at 1/4× and c128 at 1/128× compression), but at 1M context length even compressed KV exceeds GPU HBM capacity on 4× 96 GB GPUs.

## Solution: Three-tier KV cache hierarchy

### L1 — GPU HBM3

Primary KV cache. Holds active decode pages for fast random access. Managed by sglang's RadixAttention.

Capacity: ~12 GB per GPU (remaining after model weights).

### L2 — CPU DRAM (HiCache)

sglang's existing HiCache mechanism provides GPU→CPU page migration via DMA transfers. When GPU pages are evicted, they move to a host-side memory pool (`DeepSeekV4TokenToKVPoolHost`).

Capacity: configured via `--hicache-ratio` (1.5× device pool in our setup).

### L3 — Optane SSD (DiskOffloadBackend)

When DRAM is full, pages are evicted to SSD. The `DiskOffloadBackend` implements the `HiCacheStorage` interface and stores pages as individual `.pt` files with LRU eviction and a JSON-based index.

## Custom components

### `DeepSeekV4TokenToKVPoolHost`

Host-side KV cache pool for DeepSeek-V4's compressed MLA. Handles five sub-pool types:

- **SWA KV** — main sliding-window attention KV (anchor pool)
- **c4 KV** — 1/4 compressed KV
- **c128 KV** — 1/128 compressed KV
- **c4 indexer KV** — indexer for sparse attention
- **Compressor states** — c4/c128 compression state rows

Uses direct I/O (`--hicache-io-backend direct`) because the standard MLA JIT kernels cannot handle DS4V's heterogeneous page layout.

### `DiskOffloadBackend`

L3 storage backend implementing `HiCacheStorage`:

- Page-level `get`/`set`/`exists`
- Batch operations (`batch_get`/`batch_set`/`batch_exists`)
- LRU eviction with configurable disk budget
- JSON-based page index for persistence
- Compatible with both v1 and v2 HiCache APIs

### `HiRadixCache` patches

The Docker image's `HiRadixCache` required additions for DS4V compatibility:

- `sliding_window_size` — required by sglang's schedule policy for SWA
- `supports_swa()` / `full_evictable_size()` / `sanity_check()` — API surface expected by scheduler
- `dec_lock_ref(swa_uuid)` — three-argument overload for SWA lock refs

### `hybrid_pool_assembler.py` patch

Added `build_dsv4_stack()` and DS4V detection branch to `attach_hybrid_pool_to_unified_cache()`.

## Data flow

### Prefill

1. Prompt tokens are processed by the SM120 flash_mla kernel
2. KV cache pages are written to GPU L1
3. When L1 is full, evicted pages move to L2 (DRAM) via HiCache

### Decode

1. Decode reads KV from L1 (fast path) or L2 (slower DMA path)
2. New KV is appended to L1
3. Page eviction follows L1 → L2 → L3 chain

### L3 eviction

1. HiCache controller detects L2 is full
2. LRU pages are backed up via `DiskOffloadBackend.set()`
3. Pages are written to Optane SSD as `.pt` files
4. On cache miss, pages are loaded back via `DiskOffloadBackend.get()`

## Failure modes

- **OOM** — GPU memory exhausted. Reduce `mem-fraction-static` or `max-running-requests`.
- **Page mismatch** — Incorrect c4/c128 page mapping. Check `_logical_pages()` assumptions.
- **Disk latency bottleneck** — Non-Optane SSDs may increase TTFT significantly.
- **Incompatible HiCache API** — The Docker image's sglang version may lack features expected by newer code.

## Future optimizations

- Async I/O for DiskOffload (io_uring, thread pool)
- Page compression before SSD write
- Pinned host memory for faster L2→L1 DMA
- Smarter eviction policy (frequency-aware, model-aware)
