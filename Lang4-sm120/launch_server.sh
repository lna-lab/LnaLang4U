#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# cuBLAS 12.9 は sm_120 (Blackwell) で f16 GemmEx 非対応 (status 14)。
# カスタム CUDA カーネルで代替するため以下が必要:
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_SERIAL_F16_MATMUL=1

# 使用する GPU (空いてるものを選ぶ)
: "${DS4_CUDA_DEVICE:=4}"
export DS4_CUDA_DEVICE

exec "$ROOT/ds4-server" \
  -m "$ROOT/ds4flash.gguf" \
  --ctx 32768 \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
