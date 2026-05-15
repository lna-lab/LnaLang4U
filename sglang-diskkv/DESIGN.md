# DiskOffload: SSD KV Cache Backend for sglang

> **Status:** Implemented prototype used by LnaLang4U benchmark runs. Some optimization phases remain experimental.

## Original goal

Implement an SSD-backed L3 storage backend for sglang HiCache, inspired by d4-server's `--kv-disk-dir` + `--kv-disk-space-mb`. Enable 1M-context inference without OOM by offloading KV cache pages to local SSD.

## Current implementation

The prototype runs on **4 GPUs** with sglang for production throughput, while the SSD KV cache offload design remains faithful to the original single-GPU concept. See the [root README](../README.md) for current capabilities and [docs/architecture.md](../docs/architecture.md) for design details.

## Architecture

```
sglang ModelRunner
  └─ HiCacheController (L2: GPU ↔ CPU)
       └─ HiCacheStorage (L3: CPU ↔ Storage)
            └─ DiskOffloadBackend  (local SSD)
```

The `disk_offload` backend is added to the existing 3-tier HiCache structure at the L3 layer.
Data path: `GPU → CPU (L2) → Disk (L3)`

## Interface

Implements the `HiCacheStorage` ABC:

| Method | Purpose |
|--------|---------|
| `get(key) -> Tensor` | Read page from disk |
| `set(key, value) -> bool` | Write page to disk |
| `exists(key) -> bool` | Check page existence |
| `batch_get(keys) -> List[Tensor]` | Batch read |
| `batch_set(keys, values) -> bool` | Batch write |
| `batch_exists(keys) -> int` | Consecutive existence check |
| `clear()` | Remove all data |

## Data Layout

```
{kv-disk-dir}/
  ├── index.json              # Metadata index (LRU order, size)
  └── pages/
      ├── {key1}.pt           # Individual KV page file (torch.save)
      ├── {key2}.pt
      └── ...
```

- key format: `"{pool_name}/{page_id}"` (e.g., `"kv/00000000"`, `"swa/00000042"`)
- Each page is an individual file (`torch.save` / `torch.load`)
- `index.json` tracks last access time and size for all pages

## Budget Management

Concepts inherited from ds4-server:

| ds4-server | DiskOffloadBackend |
|------------|-------------------|
| `--kv-disk-dir` | `disk_offload_dir` (argument) |
| `--kv-disk-space-mb` | `max_disk_space_mb` (argument) |
| SHA-1 content addressing | Not needed (sglang manages keys) |
| LRU eviction via kv_cache_evict | LRU eviction via JSON index |

## Implementation plan

### Phase 1: Minimal DiskOffloadBackend
- `DiskOffloadBackend` class implementing `HiCacheStorage`
- Single-key `get` / `set` / `exists`
- Per-file serialization (torch.save/load)
- Registered in backend factory as `"disk_offload"`

### Phase 2: Batch operations
- `batch_get` / `batch_set` / `batch_exists`
- `batch_exists_v2` / `batch_get_v2` / `batch_set_v2` (v2 interface)
- CLI arguments: `--kv-disk-dir`, `--kv-disk-space-mb`

### Phase 3: Eviction
- Disk usage tracking
- LRU eviction
- File deletion for space reclamation
- Index persistence

### Phase 4: Performance optimization
- Async IO (thread pool)
- Parallel batch reads
- Page-size-aware chunk management
