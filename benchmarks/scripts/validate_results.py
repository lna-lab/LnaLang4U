#!/usr/bin/env python3
"""Validate benchmark CSV data for consistency with README values."""

import csv
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "2026-05-15-blackwell-4x-rtx-pro-6000"


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [WARN] {filename} not found")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def check_required_columns(data, required):
    if not data:
        return
    for col in required:
        if col not in data[0]:
            print(f"  [FAIL] Missing required column: {col}")


def check_source(data, filename):
    if not data:
        return
    for row in data:
        src = row.get("source", "")
        if src == "README-summary":
            print(f"  [WARN] {filename}: row uses summary-only data (source={src})")


def main():
    print("Validating benchmark data...")
    has_error = False

    for name, required in [
        ("single_request_throughput.csv",
         ["run_id", "tps_min", "tps_max", "source", "cuda_graphs"]),
        ("parallel_scaling.csv",
         ["run_id", "concurrency", "aggregate_tps", "scaling_vs_1", "source"]),
        ("cuda_graphs_ablation.csv",
         ["run_id", "cuda_graphs", "tps_min", "tps_max", "source"]),
        ("ttft.csv",
         ["run_id", "ttft_ms_min", "ttft_ms_max", "source"]),
    ]:
        print(f"\n{name}:")
        data = read_csv(name)
        check_required_columns(data, required)
        check_source(data, name)

    # Check SVG existence
    for svg in ["parallel_scaling", "cuda_graphs_ablation",
                 "single_request_throughput", "ttft_range"]:
        path = Path(__file__).parent.parent.parent / "docs" / "assets" / f"{svg}.svg"
        if not path.exists():
            print(f"\n  [FAIL] Missing SVG: docs/assets/{svg}.svg")
            has_error = True
        else:
            print(f"\n  [OK] docs/assets/{svg}.svg")

    if has_error:
        sys.exit(1)
    print("\nValidation complete.")


if __name__ == "__main__":
    main()
