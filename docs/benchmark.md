# Benchmark Methodology

## Hardware

See [`reproducibility.md`](reproducibility.md) for the full hardware configuration.

## Scenarios

### Single-request throughput

Short prompt (~5-14 tokens), variable output length. Measures end-to-end latency including prefill and decode.

### Parallel scaling

Identical short prompts sent concurrently using `ThreadPoolExecutor`. All requests complete before timing stops. Aggregate throughput = total output tokens / wall clock time.

### CUDA Graphs ablation

Same 200-token workload run with `--disable-cuda-graph` (OFF) and without it (ON).

### TTFT

Short prompt, single output token. Time to first token measured from request submission to first token in response.

## Measurement rules

- Warmup: first run discarded, subsequent runs averaged
- Runs per scenario: 3
- Prompt: short factual question
- Sampling: temperature=0 (deterministic)
- Throughput: output-token TPS (completion_tokens / elapsed)
- Aggregate: total output tokens across all concurrent requests / wall clock

## Results

Generated figures:

![Parallel scaling](assets/parallel_scaling.svg)
![CUDA Graphs ablation](assets/cuda_graphs_ablation.svg)
![Single-request throughput](assets/single_request_throughput.svg)
![TTFT range](assets/ttft_range.svg)

## Raw data

Available in [`benchmarks/results/2026-05-15-blackwell-4x-rtx-pro-6000/`](../benchmarks/results/2026-05-15-blackwell-4x-rtx-pro-6000/).

## Caveats

- All results are hardware-specific (4× RTX PRO 6000 Blackwell, TP=4).
- DiskOffload L3 was active but L2 cache was sufficient for short prompts; L3 was not exercised.
- CUDA Graphs ON is the production path; OFF is provided as ablation baseline only.
