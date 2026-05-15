#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# cuBLAS 12.9 does not support f16 GemmEx on sm_120 (Blackwell).
# Fall back to custom CUDA kernels via these env vars.
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_SERIAL_F16_MATMUL=1

# Select a free GPU
: "${DS4_CUDA_DEVICE:=4}"
export DS4_CUDA_DEVICE

exec "$ROOT/ds4-server" \
  -m "$ROOT/ds4flash.gguf" \
  --ctx 32768 \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
