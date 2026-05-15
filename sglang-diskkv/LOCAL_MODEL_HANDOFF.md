# Local Model Handoff — DeepSeek-V4 HiCache + DiskOffload

Date: 2026-05-15
Workspace: `/media/tonoken/Optane_DATA/Sm120-LNALAB-V4F`

## Goal

Bring up `sglang` DeepSeek-V4-Flash with:

- SM120 custom kernel
- HiCache enabled
- local SSD L3 backend: `disk_offload`
- normal short inference first, then long-context validation

Codex has already made an initial patch set. The local model should continue from
the current workspace, review the patch, run the validation checklist, and make
small corrective edits only where the checklist exposes a concrete failure.

## Current Patch Set

Files changed/added by Codex:

- `sglang-diskkv/sglang-source/python/sglang/srt/mem_cache/deepseek_v4_memory_pool_host.py`
  - Replaced the old SWA-only host pool with a direct-I/O DSV4 host pool.
  - Stores one complete DSV4 full page per HiCache page.
  - Copies SWA, c4, c128, c4 indexer, and compressor state rows.
  - Requires `--hicache-io-backend direct --hicache-mem-layout page_first`.

- `sglang-diskkv/hiradix_cache_patched.py`
  - DeepSeek-V4 now constructs `DeepSeekV4TokenToKVPoolHost`.
  - It rejects non-direct HiCache I/O early with a clear error.

- `sglang-diskkv/sglang-source/python/sglang/srt/mem_cache/storage/disk_offload/disk_offload_backend.py`
  - Reads `disk_offload_dir` / `max_disk_space_mb` from extra config or env vars.
  - Flushes `index.json` after writes.
  - Adds v2 batch methods for future hybrid-pool use.

- `sglang-diskkv/Dockerfile.sglang-dsv4`
  - Reproducible Dockerfile for the patched image.
  - Copies DSV4 host pool, patched HiRadix, DiskOffload backend, and backend factory.
  - Patches `server_args.py` inside the image to accept `disk_offload`.

- `/tmp/Dockerfile.sglang-dsv4`
  - Updated to mirror the repository Dockerfile for legacy commands.

- `launch.sh`
  - `sglang-diskkv` mode is now implemented.
  - Builds `sglang-dsv4-diskkv:latest` if missing.
  - Starts sglang with HiCache direct I/O and `disk_offload`.

Already completed light checks:

```bash
python3 -m py_compile \
  sglang-source/python/sglang/srt/mem_cache/deepseek_v4_memory_pool_host.py \
  sglang-source/python/sglang/srt/mem_cache/storage/disk_offload/disk_offload_backend.py \
  hiradix_cache_patched.py

bash -n /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F/launch.sh

docker build \
  -f sglang-diskkv/Dockerfile.sglang-dsv4 \
  -t sglang-dsv4-diskkv:test \
  /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F

docker run --rm --entrypoint bash sglang-dsv4-diskkv:test -lc \
  "python3 -m py_compile \
   /workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool_host.py \
   /workspace/sglang/python/sglang/srt/mem_cache/storage/disk_offload/disk_offload_backend.py \
   /workspace/sglang/python/sglang/srt/mem_cache/hiradix_cache.py"
```

## P0 Validation Checklist

Run these in order. Do not jump to long context until each prior step passes.

### 1. Rebuild the final image

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
docker build \
  -f sglang-diskkv/Dockerfile.sglang-dsv4 \
  -t sglang-dsv4-diskkv:latest \
  .
```

### 2. Start a short-context smoke server

Use a small context first to validate startup and one inference.

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
BUILD_DISKKV_IMAGE=0 \
CONTEXT_LENGTH=32768 \
HICACHE_RATIO=1.25 \
DISKKV_MB=65536 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

Expected log signals:

- `Initialized DeepSeek-V4 host pool`
- `DiskOffloadBackend: dir=/diskkv`
- model weight loading starts and reaches `Load weight end`
- no `HiRadixCache only supports MHA and MLA yet`
- no `DeepSeek-V4 HiCache requires direct IO`

### 3. Probe the server

From another shell:

```bash
curl -s http://127.0.0.1:9000/v1/models
```

Then:

```bash
curl -s http://127.0.0.1:9000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello, introduce yourself in one sentence.",
    "sampling_params": {
      "max_new_tokens": 32,
      "temperature": 0
    }
  }'
```

Pass condition: HTTP 200 with coherent generated text.

### 4. Confirm DiskOffload activity

After at least one completed request:

```bash
find /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F/diskkv -maxdepth 2 -type f | head
du -sh /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F/diskkv
```

Pass condition: `index.json` and page files are created after HiCache backup.

## P1 Correctness Checks

If P0 passes, test a repeated prompt to exercise L3 hit/load:

1. Send a prompt with at least several page-aligned chunks, for example 8k-16k tokens.
2. Send the same prompt again.
3. Watch logs for storage prefetch / backup messages.
4. Confirm no CUDA illegal access, no shape mismatch, and no 500 response.

If the second request fails around cache load, inspect:

- `DeepSeekV4TokenToKVPoolHost._logical_pages`
- c4/c128 compressed page mapping in `load_to_device_per_layer`
- `_state_rows_per_page` assumptions for c4 and c128 compressor states

The current implementation assumes:

- HiCache transfers are full-page aligned.
- full page id equals c4/c128/indexer page id after DSV4 compression page-size conversion.
- c4 compressor state rows per full page are `ring_size // 4`.
- c128 compressor state rows per full page are `max(1, ring_size // 128)`.

If logs disprove any assumption, patch only that mapping.

## P2 Long Context Ramp

Only after P0 and P1 pass:

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
CONTEXT_LENGTH=393216 \
HICACHE_RATIO=1.5 \
DISKKV_MB=524288 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

Then try 1M:

```bash
CONTEXT_LENGTH=1048576 \
MEM_FRACTION_STATIC=0.80 \
HICACHE_RATIO=1.25 \
DISKKV_MB=1048576 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

If host RAM is tight, reduce `HICACHE_RATIO` or set `HICACHE_SIZE` explicitly.
If GPU OOM occurs, reduce `MEM_FRACTION_STATIC`, `MAX_RUNNING_REQUESTS`, or context.

## Known Risks

- Direct I/O uses PyTorch copies instead of a custom HiCache kernel. It is for
  correctness first, not maximum throughput.
- DSV4 storage pages are Python dictionaries of tensors, so `disk_offload` is the
  intended backend. Zero-copy backends and `file` are not compatible with this
  heterogeneous payload.
- The first successful short inference does not prove L3 load correctness. A
  repeated long prompt is required to exercise load-back.
- Do not refactor unrelated sglang modules while validating this path.

## Ask Codex For Review

When local validation fails, send Codex:

- exact command used
- last 120 lines of container logs
- HTTP status and response body
- whether files appeared under `diskkv/pages`
- any local patch made by the local model

Codex should review the failure and recommend the next smallest patch.
