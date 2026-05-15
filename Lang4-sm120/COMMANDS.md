## DS4 Server Launch

```bash
cd /media/tonoken/P4800X/Lna-Lab/ds4-sm120
./launch_server.sh
```

- Listens on `http://127.0.0.1:8000`
- MTP + ctx 1M + disk KV 64GB
- Requires cuBLAS f16 workaround env vars (see `launch_server.sh`)

## Claude Code CLI

```bash
claude
```

`.bashrc` auto-configures `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` + `--bare`.
For original Anthropic API:

```bash
claude-orig
```
