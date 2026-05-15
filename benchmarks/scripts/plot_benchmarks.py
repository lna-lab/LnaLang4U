#!/usr/bin/env python3
"""Generate benchmark figures from CSV data for LnaLang4U README."""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}
plt.rcParams.update(STYLES)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "2026-05-15-blackwell-4x-rtx-pro-6000"
ASSETS_DIR = Path(__file__).parent.parent.parent / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

COLOR_ON = "#2ecc71"
COLOR_OFF = "#e74c3c"
COLOR_ACCENT = "#3498db"
COLOR_BAR = ["#3498db", "#2980b9", "#1abc9c", "#16a085"]


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [SKIP] {filename} not found")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_parallel_scaling(data):
    if not data:
        return
    rows = sorted(data, key=lambda r: int(r["concurrency"]))
    x = [int(r["concurrency"]) for r in rows]
    y = [float(r["aggregate_tps"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, y, color=COLOR_BAR, width=0.6, edgecolor="white")
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate throughput (tok/s)")
    ax.set_title("Parallel scaling — 200 tokens each, CUDA Graphs ON")
    ax.set_xticks(x)
    ax.set_ylim(0, max(y) * 1.15)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "parallel_scaling.svg")
    plt.close(fig)
    print(f"  → docs/assets/parallel_scaling.svg")


def plot_cuda_graphs_ablation(data):
    if not data:
        return
    rows = {r["cuda_graphs"]: r for r in data}
    labels = ["CUDA Graphs OFF", "CUDA Graphs ON"]
    colors = [COLOR_OFF, COLOR_ON]
    values = []
    for label, key in zip(labels, ["false", "true"]):
        if key in rows:
            v = (float(rows[key]["tps_min"]) + float(rows[key]["tps_max"])) / 2
        else:
            v = 0
        values.append(v)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f} tok/s", ha="center", va="bottom", fontweight="bold")

    speedup = values[1] / values[0] if values[0] > 0 else 0
    ax.annotate(f"{speedup:.0f}× speedup", xy=(0.5, values[1] / 2),
                fontsize=13, fontweight="bold", color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_ON, ec="none"))
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("CUDA Graphs Ablation — 200 tokens")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "cuda_graphs_ablation.svg")
    plt.close(fig)
    print(f"  → docs/assets/cuda_graphs_ablation.svg")


def plot_single_throughput(data):
    if not data:
        return
    rows = sorted(data, key=lambda r: int(r["output_tokens"]))
    labels = [f'{r["output_tokens"]} tok' for r in rows]
    mins = [float(r["tps_min"]) for r in rows]
    maxs = [float(r["tps_max"]) for r in rows]
    means = [(m + mx) / 2 for m, mx in zip(mins, maxs)]
    errors = [(m - mn, mx - m) for m, mn, mx in zip(means, mins, maxs)]
    err_low = [m - mn for m, mn in zip(means, mins)]
    err_high = [mx - m for m, mx in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[err_low, err_high], capsize=5, color=COLOR_ACCENT,
           width=0.5, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Single-request throughput, CUDA Graphs ON")
    ax.set_ylim(0, max(maxs) * 1.2)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "single_request_throughput.svg")
    plt.close(fig)
    print(f"  → docs/assets/single_request_throughput.svg")


def plot_ttft(data):
    if not data:
        return
    rows = [r for r in data if r["concurrency"] == "1"]
    if not rows:
        return
    r = rows[0]
    mn, mx = float(r["ttft_ms_min"]), float(r["ttft_ms_max"])

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(["TTFT"], mx - mn, left=mn, color=COLOR_ON, height=0.4, edgecolor="white")
    ax.axvline(mn, color="gray", ls="--", alpha=0.5)
    ax.axvline(mx, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Milliseconds")
    ax.set_title("Time to First Token (warm, CUDA Graphs ON)")
    ax.text(mn + (mx - mn) / 2, 0, f"{mn:.0f}–{mx:.0f} ms",
            ha="center", va="center", fontweight="bold", fontsize=12)
    ax.set_xlim(mn * 0.5, mx * 1.5)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "ttft_range.svg")
    plt.close(fig)
    print(f"  → docs/assets/ttft_range.svg")


def main():
    print("Generating benchmark figures...")

    data_parallel = read_csv("parallel_scaling.csv")
    plot_parallel_scaling(data_parallel)

    data_cuda = read_csv("cuda_graphs_ablation.csv")
    plot_cuda_graphs_ablation(data_cuda)

    data_single = read_csv("single_request_throughput.csv")
    plot_single_throughput(data_single)

    data_ttft = read_csv("ttft.csv")
    plot_ttft(data_ttft)

    print("\nAll figures generated in docs/assets/")


if __name__ == "__main__":
    main()
