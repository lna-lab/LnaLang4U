# Contributing

## How to report issues

Open a [GitHub issue](https://github.com/lna-lab/LnaLang4U/issues) with:

- Exact command used
- Hardware configuration (GPU, CPU, RAM, SSD)
- Docker image tag and build date
- Launch command and flags
- Full container logs (`docker logs <container>`)
- `nvidia-smi` output
- Model checkpoint used

## How to submit PRs

1. Fork the repository.
2. Create a feature branch.
3. Make changes with clear commit messages.
4. Update relevant documentation.
5. If adding benchmark results, follow the format in `benchmarks/results/`.
6. Submit a PR against `main`.

## Coding style

- Python: follow PEP 8.
- Commit messages: concise, descriptive, reference issues.
- Documentation: English, technical, reproducible.

## Benchmark contribution format

If submitting benchmark results from different hardware:

1. Create a new directory `benchmarks/results/<date>-<hardware-description>/`.
2. Copy the CSV templates from `benchmarks/results/2026-05-15-blackwell-4x-rtx-pro-6000/`.
3. Fill in `metadata.yaml` with your hardware details.
4. Run the benchmarks and populate CSVs.
5. Generate graphs using `benchmarks/scripts/plot_benchmarks.py`.
6. Submit as a PR.

### Hardware result submission template

```yaml
# Include in metadata.yaml
hardware:
  gpu_model: "<GPU model>"
  gpu_count: <N>
  vram_gb_each: <GB>
  tensor_parallel_size: <TP>
  cpu: "<CPU>"
  ram_gb: <GB>
  storage:
    type: "<SSD type>"
    model: "<SSD model>"
```
