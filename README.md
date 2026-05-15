# LnaLang4U — DeepSeek-V4-Flash on Blackwell with SSD KV Cache Offload

🚀 **First production-speed DeepSeek-V4-Flash inference with 1M context via SSD KV cache offload on NVIDIA Blackwell RTX PRO 6000.**

## Achievement

| Metric | Value |
|--------|-------|
| **Throughput** (single) | **63 tok/s** (284B model, FP8, TP=4) |
| **Throughput** (8 concurrent) | **400 tok/s aggregate** |
| **TTFT** | 125–212 ms |
| **Context Length** | **1,048,576 tokens** (1M) |
| **KV Cache** | L1 GPU → L2 DRAM → L3 **Optane SSD** |
| **Hardware** | 4 × RTX PRO 6000 Blackwell (96 GB each, SM120) |

## Architecture

```
Client ──▶ sglang (port 9000)
              │
        ┌─────┴──────┐
        │ SM120 Kernel │ (0xSero flash_mla sparse decode)
        └─────┬──────┘
              │
    ┌─────────┴──────────┐
    │  HiCache (L2 DRAM)  │
    │                     │
 ┌──▼──────────────┐ ┌───▼──────────────┐
 │  GPU (L1, 87GB) │ │ DiskOffload (L3) │
 │  KV cache       │ │ Optane SSD pages │
 └─────────────────┘ └──────────────────┘
```

## What We Built

Four custom components that extend sglang for DeepSeek-V4 on Blackwell:

### 1. `DeepSeekV4TokenToKVPoolHost`
Host-side (CPU DRAM) KV cache pool for DeepSeek-V4's compressed MLA architecture. Handles SWA, c4, c128, indexer, and compressor state sub-pools.

### 2. `DiskOffloadBackend`
L3 SSD storage backend implementing the `HiCacheStorage` interface:
- Page-level `get`/`set`/`exists` with `torch.save`/`load`
- Batch I/O (`batch_get`/`batch_set`/`batch_exists`)
- LRU eviction with configurable disk budget (`SGLANG_DISK_OFFLOAD_MAX_SPACE_MB`)
- JSON-based page index for persistence across restarts
- Compatible with both v1 and v2 HiCache APIs

### 3. `HiRadixCache` Patch
Extended the Docker image's `HiRadixCache` with:
- `sliding_window_size` for DS4V SWA support
- `supports_swa()`, `full_evictable_size()`, `sanity_check()`
- `dec_lock_ref(swa_uuid)` three-argument overload

### 4. `hybrid_pool_assembler.py` Patch
Added `build_dsv4_stack()` and DS4V detection branch in `attach_hybrid_pool_to_unified_cache()`.

## Key Insight

Everything runs on **Optane SSD** — model weights (274 GB FP8 checkpoint), KV cache pages, and OS. The low latency of Optane makes L3 disk offload practical for real-time inference.

The `ds4-server` project inspired the SSD KV cache design (SHA-based content addressing, LRU eviction, budget management), while sglang provides the SM120-optimized inference engine.

## Performance

### CUDA Graphs ON (production)

```
200 tokens: 56.9–57.4 tok/s
100 tokens: 62.8–63.0 tok/s (warm)
  TTFT:    125–212 ms
```

### Parallel Scaling (200 tokens each, CUDA Graphs ON)

| Concurrent | Aggregate TPS | Scaling vs 1 |
|-----------|--------------|-------------|
| 1 | 55.6 tok/s | 1.0× |
| 2 | 107.1 tok/s | 1.9× |
| 4 | 214.9 tok/s | 3.9× |
| **8** | **400.7 tok/s** | **7.2×** |

sglang's continuous batching packs multiple decode requests into larger batches across 4 Blackwell GPUs, achieving near-linear scaling up to 8 concurrent requests.

### CUDA Graphs OFF

```
200 tokens: 9.6 tok/s
```

Enabling CUDA graphs gives a **6× throughput improvement**.

## Launch Commands

### Baseline (no HiCache)

```bash
docker run --name sglang-dsv4 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,2,3,4 \
  --shm-size=64g --ipc=host --network host \
  -v /path/to/DeepSeek-V4-Flash-FP8:/workspace/model:ro \
  -v /path/to/sm120-kernel:/dsv4:ro \
  -e PYTHONPATH=/dsv4 \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --model-path /workspace/model --host 0.0.0.0 --port 9000 \
    --served-model-name deepseek-v4-flash --trust-remote-code \
    --tensor-parallel-size 4 --context-length 393216 \
    --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
    --fp8-gemm-backend triton --moe-runner-backend triton \
    --attention-backend compressed --page-size 256 \
    --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja
```

### 1M Context with HiCache + DiskOffload

```bash
docker run --name sglang-dsv4 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,2,3,4 \
  --shm-size=128g --ipc=host --network host \
  -v /path/to/DeepSeek-V4-Flash-FP8:/workspace/model:ro \
  -v /path/to/sm120-kernel:/dsv4:ro \
  -v /path/to/diskkv:/diskkv \
  -e PYTHONPATH=/dsv4 \
  -e SGLANG_DISK_OFFLOAD_DIR=/diskkv \
  -e SGLANG_DISK_OFFLOAD_MAX_SPACE_MB=1048576 \
  sglang-dsv4-diskkv:latest \
  python3 -m sglang.launch_server \
    --model-path /workspace/model --host 0.0.0.0 --port 9000 \
    --served-model-name deepseek-v4-flash --trust-remote-code \
    --tensor-parallel-size 4 --context-length 1048576 \
    --mem-fraction-static 0.80 --kv-cache-dtype fp8_e4m3 \
    --fp8-gemm-backend triton --moe-runner-backend triton \
    --page-size 256 \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-io-backend direct --hicache-mem-layout page_first \
    --hicache-storage-backend disk_offload
```

## Build Custom Image

```bash
cd /path/to/project
docker build -f sglang-diskkv/Dockerfile.sglang-dsv4 -t sglang-dsv4-diskkv:latest .
```

## Project Structure

```
Sm120-LNALAB-V4F/
├── launch.sh                          # Unified launcher
├── models/                            # Model symlinks
├── diskkv/                            # DiskOffloadBackend storage
├── patch_scheduler.py                 # Debug helper
├── Dockerfile.debug                   # Debug build
└── sglang-diskkv/
    ├── DESIGN.md
    ├── Dockerfile.sglang-dsv4         # Production Dockerfile
    ├── hiradix_cache_patched.py       # Patched HiRadixCache
    ├── UNRESOLVED.md
    └── sglang-source/
        └── python/sglang/srt/mem_cache/
            ├── deepseek_v4_memory_pool_host.py  # DS4V host pool
            ├── storage/disk_offload/            # DiskOffloadBackend
            │   └── disk_offload_backend.py
            └── hybrid_cache/
                └── hybrid_pool_assembler.py     # DS4V branch
```

## Model Weights

The FP8 checkpoint is available on HuggingFace:

- **Primary (recommended):** [`sgl-project/DeepSeek-V4-Flash-FP8`](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) (274 GB, true FP8)
- **Alternative:** [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (149 GB, packed FP4 — requires `SGLANG_DSV4_FP4_EXPERTS=1`)

Download:

```bash
# 274 GB FP8 version (recommended)
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-FP8 \
  --local-dir-use-symlinks False
```

The model path is mounted into the container at `/workspace/model:ro`. Set `MODEL_DIR` environment variable or create a symlink in `models/DeepSeek-V4-Flash-FP8`.

## Prerequisites

- [Docker](https://docs.docker.com/) with NVIDIA Container Toolkit
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) for model download
- NVIDIA Blackwell GPU (RTX PRO 6000 or similar, SM120)
- SM120 kernel (auto-built by `launch.sh`)

## Known Issues

- The Docker image's `HiRadixCache` requires multiple compatibility patches for DS4V (sliding_window_size, supports_swa, sanity_check, dec_lock_ref)
- `DiskOffloadBackend` L3 eviction works automatically when L2 is full; with `--hicache-ratio 1.5` and short contexts, L2 is usually sufficient
- FP4-packed experts (149 GB model) require `SGLANG_DSV4_FP4_EXPERTS=1`; our setup uses the true-FP8 274 GB model

## Credits

- **sglang** project for the inference engine and SM120 support
- **0xSero** for the SM120 flash_mla kernel
- **[antirez/ds4](https://github.com/antirez/ds4)** — the original DwarfStar 4 server. The SSD KV cache offload design (`--kv-disk-dir`, SHA-based content addressing, LRU eviction, cold/continued/evict save strategies) is a direct homage to this brilliant project. Thank you, Salvatore! 🙌
- Built with ❤️ at Lna-Lab
