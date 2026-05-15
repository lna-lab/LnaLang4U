# Running Inference

## Prerequisites

Before starting, run these checks:

```bash
nvidia-smi                               # verify GPUs are available
docker --version                         # verify Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi  # verify GPU access
df -h /path/to/diskkv                    # verify disk space for KV cache
```

## 1. Build the image

```bash
export PROJECT_DIR=/path/to/LnaLang4U
cd "$PROJECT_DIR"
docker build -f sglang-diskkv/Dockerfile.sglang-dsv4 \
  -t sglang-dsv4-diskkv:latest .
```

## 2. Launch tiers

### Tier 1: Smoke test (32K context)

```bash
export MODEL_DIR=/path/to/DeepSeek-V4-Flash-FP8
export DISKKV_DIR=/path/to/diskkv
export DSV4_KERNEL_DIR=/path/to/deepseek-v4-flash-sm120/build-docker

docker run --name sglang-dsv4 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,2,3,4 \
  --shm-size=64g --ipc=host --network host \
  -v "$MODEL_DIR":/workspace/model:ro \
  -v "$DSV4_KERNEL_DIR":/dsv4:ro \
  -v "$DISKKV_DIR":/diskkv \
  -e PYTHONPATH=/dsv4 \
  -e SGLANG_DISK_OFFLOAD_DIR=/diskkv \
  -e SGLANG_DISK_OFFLOAD_MAX_SPACE_MB=65536 \
  sglang-dsv4-diskkv:latest \
  python3 -m sglang.launch_server \
    --model-path /workspace/model --host 0.0.0.0 --port 9000 \
    --served-model-name deepseek-v4-flash --trust-remote-code \
    --tensor-parallel-size 4 --context-length 32768 \
    --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
    --fp8-gemm-backend triton --page-size 256 \
    --enable-hierarchical-cache --hicache-ratio 1.25 \
    --hicache-io-backend direct --hicache-mem-layout page_first \
    --hicache-storage-backend disk_offload
```

### Tier 2: Medium context (393K)

Increase `--context-length 393216` and `--max-running-requests 8`.

### Tier 3: Full 1M context

```bash
  -e SGLANG_DISK_OFFLOAD_MAX_SPACE_MB=1048576 \
  ...
  --context-length 1048576 --mem-fraction-static 0.80 \
  --hicache-ratio 1.5
```

## 3. Test

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","max_tokens":20,"messages":[{"role":"user","content":"hello"}]}'
```

Expected: HTTP 200 with a coherent response.

## Troubleshooting

See [`docs/troubleshooting.md`](../docs/troubleshooting.md) for common issues.
