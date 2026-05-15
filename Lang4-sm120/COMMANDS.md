## DS4 Server 起動

```bash
cd /media/tonoken/P4800X/Lna-Lab/ds4-sm120
./launch_server.sh
```

- `http://127.0.0.1:8000` で待受
- ctx 32K, GPU 4 使用 (空き GPU は `DS4_CUDA_DEVICE=N` で指定)
- env var `DS4_CUDA_NO_Q8_F16_CACHE=1` + `DS4_CUDA_SERIAL_F16_MATMUL=1` を内蔵
- **理由**: cuBLAS 12.9 は sm_120 (Blackwell RTX PRO 6000) で f16 GemmEx 非対応 (status 14)。カスタム CUDA カーネルで代替

## Claude Code CLI 起動

```bash
claude
```

`.bashrc` で `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` + `--bare` を自動設定済み。
元の Anthropic API が必要な時は:

```bash
claude-orig
```
