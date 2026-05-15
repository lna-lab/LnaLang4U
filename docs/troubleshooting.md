# Troubleshooting

## Server exits immediately with exit code 0

**Likely cause:** HiCache initialization assertion failed (`self.size > device_pool.size`).

**Fix:** Increase `--hicache-ratio` or set `--hicache-size` explicitly to a value larger than the device pool.

## "HiRadixCache only supports MHA and MLA yet"

**Cause:** The Docker image's `HiRadixCache` does not recognize `DeepSeekV4TokenToKVPool`.

**Fix:** Use the custom Docker image (`sglang-dsv4-diskkv:latest`) which includes the DS4V patch.

## AttributeError: 'HiRadixCache' object has no attribute 'sliding_window_size'

**Cause:** Missing DS4V compatibility attribute in the cache object.

**Fix:** Ensure the patched `hiradix_cache.py` is used. Rebuild the Docker image.

## TypeError: RadixCache.dec_lock_ref() takes 2 positional arguments but 3 were given

**Cause:** The scheduler passes a `swa_uuid` argument that the base class doesn't accept.

**Fix:** Use the patched `hiradix_cache.py` which overrides `dec_lock_ref` to accept the extra argument.

## RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048)

**Cause:** Model checkpoint format mismatch. The 149 GB `deepseek-ai/DeepSeek-V4-Flash` checkpoint uses packed FP4 format with different weight shapes.

**Fix:** Use the 274 GB `sgl-project/DeepSeek-V4-Flash-FP8` checkpoint (true FP8), or set `SGLANG_DSV4_FP4_EXPERTS=1` for the FP4 model.

## Container stays up but /generate returns 500

**Cause:** The scheduler process crashed. Check container logs:

```bash
docker logs sglang-dsv4 2>&1 | grep -i "scheduler\|error\|exception"
```

Common causes:
- CUDA graph capture failure (add `--disable-cuda-graph`)
- Out of memory (reduce `--mem-fraction-static`)
- Model weight loading error (check model path and format)

## Performance lower than expected

- **CUDA Graphs OFF:** Remove `--disable-cuda-graph`. This gives approximately 6× speedup.
- **Small output tokens:** Throughput is measured in output-token TPS. Short outputs have higher overhead.
- **Cold start:** First request includes CUDA graph capture. Discard it from benchmarks.
- **SSD latency:** Non-Optane SSDs may increase L3 load latency.

## DiskOffload not writing to SSD

**Cause:** L2 (DRAM) cache is sufficient for current workload.

HiCache only evicts to L3 when L2 is full. With short prompts and `--hicache-ratio 1.5`, L2 may never fill up. Send longer prompts or reduce `--hicache-ratio`.

## Model not loading on GPU 1

GPU 1 is reserved for display on this workstation. Use `CUDA_VISIBLE_DEVICES=0,2,3,4` to skip it.

## CUDA OOM during model load

**Fix:** Reduce `--mem-fraction-static` (try 0.80 or lower). Reduce `--max-running-requests`. Reduce `--context-length`.

## Host memory allocation failure

```bash
# Increase shared memory for Docker
--shm-size=128g
```

## Docker build fails on server_args.py

The Dockerfile patches `server_args.py` to add `disk_offload` to the choices list. If the patch fails:

```bash
# Check that the choices list format matches expectations
docker run --rm --entrypoint bash sglang-dsv4-diskkv:latest \
  -c 'grep -A2 "hicache-storage-backend" /workspace/sglang/python/sglang/srt/server_args.py'
```

## Still stuck?

Open an issue at [github.com/lna-lab/LnaLang4U/issues](https://github.com/lna-lab/LnaLang4U/issues) with:
- Exact command used
- Full container logs
- GPU model and driver version
- Model checkpoint used
- `nvidia-smi` output
