# sglang-diskkv

SSD-backed KV cache offload backend for sglang HiCache, developed for DeepSeek-V4-Flash long-context inference on NVIDIA Blackwell.

## What this module provides

- **`DiskOffloadBackend`** — L3 SSD storage backend with LRU eviction, batch I/O, and JSON-based page index
- **`DeepSeekV4TokenToKVPoolHost`** — host-side KV pool for DeepSeek-V4's compressed MLA (SWA, c4, c128, indexer, compressor states)
- **HiCache compatibility patches** — `HiRadixCache` extensions for DS4V support (`sliding_window_size`, `supports_swa()`, `full_evictable_size()`, `sanity_check()`, `dec_lock_ref(swa_uuid)`)
- **`hybrid_pool_assembler.py` patch** — `build_dsv4_stack()` DS4V integration
- **Dockerfile** — builds patched sglang image with all components

## Relationship to root project

This directory contains the runtime modifications used by [LnaLang4U](https://github.com/lna-lab/LnaLang4U). The vendored `sglang-source/` is a snapshot of the sglang repository with only the patched files diverging.

## Implementation status

| Area | Status | Notes |
|------|--------|-------|
| DiskOffloadBackend | Implemented | page-level and batch operations |
| LRU eviction | Implemented | JSON index based |
| HiCache v1/v2 compatibility | Implemented | compatibility layer for Pool* types |
| DeepSeek-V4 host pool | Implemented | SWA/c4/c128/indexer/compressor pools |
| 1M context | Verified | passes smoke test |
| Async IO | Planned | future optimization |
| Benchmark instrumentation | In progress | raw log data needed |

## Usage

- Build: `docker build -f Dockerfile.sglang-dsv4 -t sglang-dsv4-diskkv:latest .`
- Run: see [root README](../README.md) for launch commands
- Architecture: see [docs/architecture.md](../docs/architecture.md)
- Benchmark: see [docs/benchmark.md](../docs/benchmark.md)

## Original design goal

> Historical note: The original concept was "1 GPU, 1M context via SSD KV cache offload on DS4-server." The current implementation uses 4 GPUs with sglang for production throughput, while the SSD offload design remains faithful to the original vision.
