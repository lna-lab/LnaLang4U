# Reproducibility

## Tested hardware

| Component | Detail |
|-----------|--------|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell (96 GB HBM3 each, SM120) |
| CPU | AMD Ryzen Threadripper |
| RAM | 128 GB |
| System SSD | NVMe (for OS) |
| Cache SSD | Intel Optane SSD 905P (for DiskOffload) |

## Software versions

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.4 LTS |
| NVIDIA driver | 595.58.03 |
| CUDA (driver) | 13.2 |
| CUDA (runtime) | 12.9.1 |
| Python | 3.12 |
| sglang base image | `lmsysorg/sglang:deepseek-v4-blackwell` |
| Custom image | `sglang-dsv4-diskkv:latest` (build from this repo) |
| SM120 kernel | 0xSero/deepseek-v4-flash-sm120 build-docker output |
| Model | `sgl-project/DeepSeek-V4-Flash-FP8` (274 GB, FP8) |

## Model download

```bash
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir /path/to/DeepSeek-V4-Flash-FP8 \
  --local-dir-use-symlinks False
```

## Docker build

```bash
cd /path/to/LnaLang4U
docker build -f sglang-diskkv/Dockerfile.sglang-dsv4 \
  -t sglang-dsv4-diskkv:latest .
```

## SM120 kernel

The kernel is built separately using [0xSero/deepseek-v4-flash-sm120](https://github.com/0xSero/deepseek-v4-flash-sm120):

```bash
cd /path/to/deepseek-v4-flash-sm120
bash scripts/build_in_sglang_docker.sh
```

Output: `build-docker/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so`

## Exact launch command (1M context)

```bash
docker run --name sglang-dsv4 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,2,3,4 \
  --shm-size=128g --ipc=host --network host \
  -v /path/to/DeepSeek-V4-Flash-FP8:/workspace/model:ro \
  -v /path/to/kernel/build-docker:/dsv4:ro \
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
    --fp8-gemm-backend triton --page-size 256 \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-io-backend direct --hicache-mem-layout page_first \
    --hicache-storage-backend disk_offload
```

Run on GPUs 0, 2, 3, 4 to leave GPU 1 for display.

## Benchmark command

```bash
# Single request
curl -s -w "\nHTTP: %{http_code}" http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","max_tokens":100,"messages":[{"role":"user","content":"What is AI?"}],"temperature":0}'

# See benchmarks/scripts/ for automated measurement
```

## Expected output

- Model loads to ~82-87 GiB per GPU
- `/v1/models` returns `max_model_len: 1048576`
- Inference returns HTTP 200 with coherent text
- First request may take longer (CUDA graph capture)

## Verifying DiskOffload is active

```bash
# Check that diskkv directory has index and page structure
ls -la /path/to/diskkv/

# Check for DiskOffloadBackend in startup logs
docker logs sglang-dsv4 2>&1 | grep -i "disk_offload\|DiskOffload"
```

## Comparing with published numbers

- Use the same hardware and launch command
- Ensure CUDA Graphs are ON (remove `--disable-cuda-graph`)
- Use short prompts (~5-14 tokens) for single-request benchmarks
- Wait for warmup (first run may be slower)
