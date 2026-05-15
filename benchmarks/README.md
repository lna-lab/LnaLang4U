# Benchmarks

This directory contains benchmark data, results, and graph generation scripts for LnaLang4U.

## Structure

```
benchmarks/
├── README.md
├── prompts/                    # Prompt templates used in benchmarks
├── results/                    # Benchmark results by date and hardware
│   └── 2026-05-15-blackwell-4x-rtx-pro-6000/
│       ├── metadata.yaml       # Hardware and software configuration
│       ├── single_request_throughput.csv
│       ├── parallel_scaling.csv
│       ├── cuda_graphs_ablation.csv
│       ├── ttft.csv
│       └── raw_logs/           # Server and benchmark logs
└── scripts/
    ├── plot_benchmarks.py      # Generate SVGs from CSV data
    └── validate_results.py     # Validate data consistency
```

## CSV schemas

Each CSV follows a schema documented in the column headers. Key conventions:

- `source`: indicates data origin — `measured` (from raw logs), `README-summary` (from project documentation)
- TPS values: output-token throughput unless noted
- `summary-only` values should be replaced with raw log measurements

## Graph generation

```bash
python3 benchmarks/scripts/plot_benchmarks.py
```

Output SVGs are written to `docs/assets/`.

## Adding new results

1. Create a new directory `results/<date>-<hardware-description>/`.
2. Copy CSV templates and `metadata.yaml`.
3. Run benchmarks and populate data.
4. Add raw logs to `raw_logs/`.
5. Regenerate graphs.
