#!/usr/bin/env bash
# Lna-Lab LnaLang4U unified launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"

usage() {
    echo "Usage: $0 {ds4-server|sglang|sglang-diskkv} [extra args...]"
    echo ""
    echo "  ds4-server        — GGUF server with cuBLAS workaround"
    echo "  sglang            — FP8 4GPU sglang (SM120 kernel)"
    echo "  sglang-diskkv     — FP8 + DiskOffload SSD KV cache (experimental)"
    exit 1
}

MODE="${1:-}"
shift || true

case "$MODE" in
    ds4-server)
        exec "${SCRIPT_DIR}/ds4-sm120/launch_server.sh" "$@"
        ;;
    sglang)
        # standard LnaLab sglang launch (4GPU, FP8, SM120 kernel)
        if [[ ! -f "${MODEL_DIR}/DeepSeek-V4-Flash/config.json" ]] && \
           [[ ! -f "${MODEL_DIR}/DeepSeek-V4-Flash-FP8/config.json" ]]; then
            echo "ERROR: FP8 model not found. Expected at:"
            echo "  ${MODEL_DIR}/DeepSeek-V4-Flash/"
            echo "  or ${MODEL_DIR}/DeepSeek-V4-Flash-FP8/"
            exit 1
        fi
        # prefer sgl-project FP8 model; fall back to deepseek-ai FP4 model
        FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash-FP8"
        [[ -f "${FP8_DIR}/config.json" ]] || FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash"

        # SM120 kernel path
        DSV4_BUILD_DIR="/media/tonoken/SN8100/repos/deepseek-v4-flash-sm120/build-docker"
        if [[ ! -f "${DSV4_BUILD_DIR}/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so" ]]; then
            echo "Building SM120 kernel (one-time)..."
            bash "/media/tonoken/SN8100/repos/deepseek-v4-flash-sm120/scripts/build_in_sglang_docker.sh"
        fi

        GPUS="${GPUS:-0,1,2,3}"
        PORT="${PORT:-9000}"
        CONTAINER_NAME="${CONTAINER_NAME:-sglang-dsv4-sm120}"

        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
        exec docker run --name "${CONTAINER_NAME}" \
          --gpus all \
          -e CUDA_VISIBLE_DEVICES="${GPUS}" \
          --shm-size=64g --ipc=host \
          --ulimit memlock=-1 --ulimit stack=67108864 \
          --network host \
          -v "${FP8_DIR}:/workspace/model:ro" \
          -v "${DSV4_BUILD_DIR}:/dsv4:ro" \
          -e PYTHONPATH=/dsv4 \
          -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
          -e TORCH_CUDA_ARCH_LIST=12.0 \
          -e SAFETENSORS_FAST_GPU=1 \
          -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
          -e NCCL_P2P_DISABLE=0 -e NCCL_IB_DISABLE=1 \
          -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
          -e NCCL_DEBUG=WARN -e NCCL_CUMEM_HOST_ENABLE=0 \
          -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
          -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
          -e SGLANG_DSV4_FP4_EXPERTS=0 \
          -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
          -e SGLANG_OPT_USE_TILELANG_INDEXER=1 \
          -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
          -e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1 \
          -e SGLANG_OPT_USE_TILELANG_MHC_PRE=1 \
          -e SGLANG_OPT_USE_TILELANG_MHC_POST=1 \
          -e SGLANG_ENABLE_SPEC_V2=True \
          -e SGLANG_SET_CPU_AFFINITY=1 \
          lmsysorg/sglang:deepseek-v4-blackwell \
          python3 -m sglang.launch_server \
            --model-path /workspace/model \
            --host 0.0.0.0 --port "${PORT}" \
            --served-model-name deepseek-v4-flash \
            --trust-remote-code \
            --tensor-parallel-size 4 \
            --context-length 393216 \
            --mem-fraction-static 0.85 \
            --max-running-requests 8 \
            --kv-cache-dtype fp8_e4m3 \
            --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja \
            --fp8-gemm-backend triton \
            --moe-runner-backend triton \
            --attention-backend compressed \
            --chunked-prefill-size 8192 \
            --watchdog-timeout 3600 \
            --page-size 256 \
            --disable-cuda-graph "$@"
        ;;
    sglang-diskkv)
        if [[ ! -f "${MODEL_DIR}/DeepSeek-V4-Flash/config.json" ]] && \
           [[ ! -f "${MODEL_DIR}/DeepSeek-V4-Flash-FP8/config.json" ]]; then
            echo "ERROR: FP8 model not found. Expected at:"
            echo "  ${MODEL_DIR}/DeepSeek-V4-Flash/"
            echo "  or ${MODEL_DIR}/DeepSeek-V4-Flash-FP8/"
            exit 1
        fi

        FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash-FP8"
        [[ -f "${FP8_DIR}/config.json" ]] || FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash"

        DSV4_BUILD_DIR="/media/tonoken/SN8100/repos/deepseek-v4-flash-sm120/build-docker"
        if [[ ! -f "${DSV4_BUILD_DIR}/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so" ]]; then
            echo "Building SM120 kernel (one-time)..."
            bash "/media/tonoken/SN8100/repos/deepseek-v4-flash-sm120/scripts/build_in_sglang_docker.sh"
        fi

        IMAGE="${SGLANG_DISKKV_IMAGE:-sglang-dsv4-diskkv:latest}"
        if [[ "${BUILD_DISKKV_IMAGE:-0}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
            docker build \
              -f "${SCRIPT_DIR}/sglang-diskkv/Dockerfile.sglang-dsv4" \
              -t "${IMAGE}" \
              "${SCRIPT_DIR}"
        fi

        GPUS="${GPUS:-0,1,2,3}"
        PORT="${PORT:-9000}"
        CONTAINER_NAME="${CONTAINER_NAME:-sglang-dsv4-diskkv}"
        DISKKV_DIR="${DISKKV_DIR:-${SCRIPT_DIR}/diskkv}"
        DISKKV_MB="${DISKKV_MB:-524288}"
        CONTEXT_LENGTH="${CONTEXT_LENGTH:-393216}"
        HICACHE_RATIO="${HICACHE_RATIO:-2}"
        HICACHE_SIZE="${HICACHE_SIZE:-0}"
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
        mkdir -p "${DISKKV_DIR}"

        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
        exec docker run --name "${CONTAINER_NAME}" \
          --gpus all \
          -e CUDA_VISIBLE_DEVICES="${GPUS}" \
          --shm-size=64g --ipc=host \
          --ulimit memlock=-1 --ulimit stack=67108864 \
          --network host \
          -v "${FP8_DIR}:/workspace/model:ro" \
          -v "${DSV4_BUILD_DIR}:/dsv4:ro" \
          -v "${DISKKV_DIR}:/diskkv" \
          -e PYTHONPATH=/dsv4 \
          -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
          -e TORCH_CUDA_ARCH_LIST=12.0 \
          -e SAFETENSORS_FAST_GPU=1 \
          -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
          -e NCCL_P2P_DISABLE=0 -e NCCL_IB_DISABLE=1 \
          -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
          -e NCCL_DEBUG=WARN -e NCCL_CUMEM_HOST_ENABLE=0 \
          -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
          -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
          -e SGLANG_DSV4_FP4_EXPERTS=0 \
          -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
          -e SGLANG_OPT_USE_TILELANG_INDEXER=1 \
          -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
          -e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1 \
          -e SGLANG_OPT_USE_TILELANG_MHC_PRE=1 \
          -e SGLANG_OPT_USE_TILELANG_MHC_POST=1 \
          -e SGLANG_ENABLE_SPEC_V2=True \
          -e SGLANG_SET_CPU_AFFINITY=1 \
          -e SGLANG_DISK_OFFLOAD_DIR=/diskkv \
          -e SGLANG_DISK_OFFLOAD_MAX_SPACE_MB="${DISKKV_MB}" \
          "${IMAGE}" \
          python3 -m sglang.launch_server \
            --model-path /workspace/model \
            --host 0.0.0.0 --port "${PORT}" \
            --served-model-name deepseek-v4-flash \
            --trust-remote-code \
            --tensor-parallel-size 4 \
            --context-length "${CONTEXT_LENGTH}" \
            --mem-fraction-static "${MEM_FRACTION_STATIC}" \
            --max-running-requests "${MAX_RUNNING_REQUESTS}" \
            --kv-cache-dtype fp8_e4m3 \
            --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja \
            --fp8-gemm-backend triton \
            --moe-runner-backend triton \
            --attention-backend compressed \
            --chunked-prefill-size 8192 \
            --watchdog-timeout 3600 \
            --page-size 256 \
            --disable-cuda-graph \
            --enable-hierarchical-cache \
            --hicache-ratio "${HICACHE_RATIO}" \
            --hicache-size "${HICACHE_SIZE}" \
            --hicache-io-backend direct \
            --hicache-mem-layout page_first \
            --hicache-storage-backend disk_offload \
            --hicache-storage-backend-extra-config "{\"disk_offload_dir\":\"/diskkv\",\"max_disk_space_mb\":${DISKKV_MB}}" \
            "$@"
        ;;
    *)
        usage
        ;;
esac
