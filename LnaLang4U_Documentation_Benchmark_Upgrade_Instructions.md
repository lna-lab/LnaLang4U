# LnaLang4U Documentation & Benchmark Publication Upgrade 指示書

## 0. Mission

あなたは Lna-Lab 初の大型公開リポジトリ `lna-lab/LnaLang4U` を、研究ログではなく、世界中のLLM推論エンジニア・GPU最適化研究者・OSS利用者が読み、信頼し、Starしたくなる公開リポジトリへ改修してください。

本リポジトリの中核価値は以下です。

- DeepSeek-V4-Flash を NVIDIA Blackwell 環境で実用速度推論すること
- SSD KV cache offload により、1M context 推論を現実的な構成で成立させたこと
- sglang / HiCache / DiskOffload / SM120 kernel / CUDA Graphs の統合実装を公開していること
- Lna-Lab の初期大型OSS成果として、再現性・信頼性・発見性を高めること

最重要方針：

**強い主張は、必ずデータ・ログ・再現手順で支える。  
不明なものは推測で埋めず、`TODO`, `TBD`, `not yet measured`, `summary-only` と明記する。  
“すごい”ではなく、“再現できるからすごい”リポジトリにする。**

---

## 1. 全体トーンと文章方針

### 1.1 Primary language

READMEおよび公開向けドキュメントは英語を主言語にしてください。

理由：

- GitHubでの発見性を最大化する
- sglang / DeepSeek / Blackwell / CUDA / inference optimization 関係者に直接届くようにする
- 海外エンジニアがREADMEだけで価値を理解できるようにする

日本語は必要に応じて以下の形に限定してください。

- `docs/ja/README.ja.md`
- README末尾の “Japanese note”
- 開発メモ由来の日本語は、公開文書では英語化する

### 1.2 Tone

文体は以下に統一してください。

- Clear
- Technical
- Reproducible
- Calmly impressive
- No exaggerated hype
- No unverified “world first” style claim

特に、現在のREADMEにある `First production-speed ...` のような強い表現は、裏付けがある場合のみ維持してください。裏付けが薄い場合は次のように弱めること。

推奨表現：

```text
LnaLang4U demonstrates production-speed DeepSeek-V4-Flash inference with 1M context using SSD KV cache offload on NVIDIA Blackwell.
```

避ける表現：

```text
World first
The fastest
The only implementation
Guaranteed production-ready
```

ただし、実測ログ・比較対象・日付・条件が明示できるなら、強い表現を残してもよいです。

---

## 2. 最優先で直すべき公開印象

### 2.1 GitHub About欄

GitHub repository settings で以下を設定してください。

Description:

```text
1M-context DeepSeek-V4-Flash inference on NVIDIA Blackwell using sglang and SSD KV cache offload.
```

Website:

```text
https://Lna-Lab.com
```

Topics:

```text
deepseek
deepseek-v4
deepseek-v4-flash
sglang
blackwell
nvidia
rtx-pro-6000
kv-cache
ssd-offload
optane
cuda-graphs
fp8
llm-inference
long-context
hierarchical-cache
```

必要なら `dsv4`, `flash-mla`, `sm120`, `hicache` も追加してください。

### 2.2 Social Preview

GitHubのSocial Preview画像を作成してください。

推奨構成：

- 背景：Blackwell / KV cache hierarchy / 1M context を想起させる落ち着いた技術系ビジュアル
- 大きな文字：`LnaLang4U`
- サブコピー：`1M-context DeepSeek-V4-Flash inference with SSD KV cache offload`
- 小さな実績値：`63 tok/s single · 400 tok/s @ 8 concurrent · 4× RTX PRO 6000`
- 画像サイズ：1280×640px
- ファイル：`docs/assets/social-preview.png`

注意：

- 誤解を招くグラフや未検証値は入れない
- 視認性を最優先
- README内の主要グラフとデザインを合わせる

---

## 3. README.md 改修方針

ルート `README.md` は、最初の30秒で以下が分かる構成にしてください。

1. これは何か
2. 何が新しいのか
3. どれくらい速いのか
4. どう動かすのか
5. どこまで再現できるのか
6. 何が未解決か

### 3.1 推奨README構成

以下の構成へ全面改修してください。

```markdown
# LnaLang4U

> Production-speed DeepSeek-V4-Flash inference with 1M context on NVIDIA Blackwell, powered by sglang and SSD KV cache offload.

[badges...]

## Highlights

- 1,048,576-token context length
- 63 tok/s single-request throughput
- 400 tok/s aggregate throughput at 8 concurrent requests
- 125–212 ms TTFT in measured runs
- 4× RTX PRO 6000 Blackwell, 96GB each
- GPU → DRAM → Optane SSD hierarchical KV cache
- Custom DeepSeek-V4 host KV pool and DiskOffload backend for sglang

## Why this matters

Explain the practical problem:
- Long-context inference is limited by GPU memory.
- DeepSeek-V4-Flash has compressed MLA / SWA complexity.
- Pure GPU KV cache is expensive at 1M context.
- SSD-backed KV cache makes long-context inference feasible on local hardware.
- Optane’s latency profile makes L3 KV offload practical.

## Architecture

Show hierarchy:
Client → sglang → SM120 kernel → HiCache → GPU L1 / DRAM L2 / SSD L3

Use both:
- Mermaid diagram in README
- Static SVG/PNG in `docs/assets/architecture.svg`

## Benchmark results

Include graphs and tables generated from raw benchmark data.

Required:
- Throughput summary table
- Parallel scaling graph
- CUDA Graphs ON/OFF graph
- TTFT range graph
- Benchmark methodology link

## Quick start

Provide the shortest realistic path:
1. Hardware assumptions
2. Download model weights
3. Build Docker image
4. Launch baseline
5. Launch 1M-context DiskOffload mode
6. Run test request

## Reproducibility

Link to:
- `docs/benchmark.md`
- `benchmarks/results/...`
- exact Docker image
- model revision
- launch command
- benchmark command
- hardware metadata

## Project structure

Present a clean tree matching the actual repository.

## Documentation

- Architecture
- Benchmark methodology
- Running inference
- Troubleshooting
- Design notes

## Known limitations

Be honest:
- Hardware-specific
- Blackwell/SM120 assumptions
- DiskOffload performance depends on SSD latency
- CUDA Graphs are critical for production throughput
- 1M-context path may require careful memory tuning

## Roadmap

Split into:
- Near-term documentation
- Benchmark improvements
- Runtime improvements
- Upstreaming opportunities

## Credits

Credit:
- sglang
- 0xSero
- antirez/ds4
- DeepSeek / model providers
- Lna-Lab

## Citation

Add `CITATION.cff` if appropriate.

## License

Only state license after maintainer approval.
```

### 3.2 Badges

Add only truthful badges. Do not add fake CI/license badges.

Recommended examples:

```markdown
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-Blackwell%20SM120-green)
![Model](https://img.shields.io/badge/Model-DeepSeek--V4--Flash-purple)
![Context](https://img.shields.io/badge/Context-1M%20tokens-orange)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
```

License badgeは、ライセンスが確定してから追加してください。

### 3.3 Achievement table

現在の `Metric Value` はMarkdownテーブルとして正しく整形してください。

例：

```markdown
| Metric | Value | Notes |
|---|---:|---|
| Single-request throughput | 63 tok/s | 100-token warm run, CUDA Graphs ON |
| 8-concurrent aggregate throughput | 400.7 tok/s | 200 tokens each, CUDA Graphs ON |
| TTFT | 125–212 ms | measured range |
| Context length | 1,048,576 tokens | 1M context |
| KV hierarchy | GPU L1 → DRAM L2 → Optane SSD L3 | HiCache + DiskOffload |
| Hardware | 4× RTX PRO 6000 Blackwell, 96GB each | TP=4, SM120 |
```

注意：

- `63 tok/s` が100-token warm run由来なら、そう明記する
- `56.9–57.4 tok/s` が200-token runなら、混同しない
- `400.7 tok/s` は8並列aggregateであり、single throughputではないことを明確にする

---

## 4. Benchmark / Graph Data 完備タスク

このリポジトリを“売れっ子”にする最大要素は、ベンチマークの見せ方です。README内の数値だけでなく、**raw data, metadata, graph script, generated figures** をすべて公開してください。

### 4.1 ディレクトリ構成

以下を追加してください。

```text
benchmarks/
├── README.md
├── prompts/
│   ├── short_prompt.txt
│   ├── long_context_prompt_template.txt
│   └── parallel_scaling_prompt.txt
├── results/
│   └── 2026-05-15-blackwell-4x-rtx-pro-6000/
│       ├── metadata.yaml
│       ├── single_request_throughput.csv
│       ├── parallel_scaling.csv
│       ├── cuda_graphs_ablation.csv
│       ├── ttft.csv
│       ├── context_length_sweep.csv
│       ├── disk_offload_metrics.csv
│       └── raw_logs/
│           ├── server.log
│           ├── benchmark_single_request.log
│           ├── benchmark_parallel_scaling.log
│           └── benchmark_cuda_graphs_ablation.log
└── scripts/
    ├── plot_benchmarks.py
    └── validate_results.py

docs/
├── assets/
│   ├── architecture.svg
│   ├── parallel_scaling.svg
│   ├── cuda_graphs_ablation.svg
│   ├── single_request_throughput.svg
│   ├── ttft_range.svg
│   └── context_length_sweep.svg
├── benchmark.md
├── architecture.md
├── reproducibility.md
└── troubleshooting.md
```

### 4.2 metadata.yaml 必須項目

`metadata.yaml` には以下を含めてください。

```yaml
benchmark_id: 2026-05-15-blackwell-4x-rtx-pro-6000
date: 2026-05-15
repository: lna-lab/LnaLang4U
git_commit: "<commit hash>"
docker_image: "sglang-dsv4-diskkv:latest"
sglang_base_image: "lmsysorg/sglang:deepseek-v4-blackwell"
model:
  name: "DeepSeek-V4-Flash"
  checkpoint: "sgl-project/DeepSeek-V4-Flash-FP8"
  dtype: "FP8"
  checkpoint_size_gb: 274
hardware:
  gpu_model: "NVIDIA RTX PRO 6000 Blackwell"
  gpu_count: 4
  vram_gb_each: 96
  tensor_parallel_size: 4
  cpu: "<CPU model>"
  ram_gb: "<system RAM>"
  storage:
    type: "Optane SSD"
    model: "<SSD model>"
    filesystem: "<filesystem>"
software:
  os: "<OS>"
  nvidia_driver: "<driver version>"
  cuda: "<CUDA version>"
  python: "<Python version>"
  pytorch: "<PyTorch version>"
runtime:
  context_length: 1048576
  kv_cache_dtype: "fp8_e4m3"
  page_size: 256
  hicache_ratio: 1.5
  disk_offload_max_space_mb: 1048576
  cuda_graphs: true
notes:
  - "Do not fabricate missing fields. Use TBD if unknown."
```

### 4.3 single_request_throughput.csv

現在READMEにある summary 値を使う場合は、必ず `source=README-summary` としてください。可能なら実測ログから再計測してください。

推奨スキーマ：

```csv
run_id,date,source,model,hardware,gpu_count,tp,context_length,output_tokens,cuda_graphs,tps_min,tps_max,tps_mean,ttft_ms_min,ttft_ms_max,notes
single_100_warm,2026-05-15,README-summary,DeepSeek-V4-Flash,RTX PRO 6000 Blackwell,4,4,1048576,100,true,62.8,63.0,,125,212,warm run; replace with raw log when available
single_200,2026-05-15,README-summary,DeepSeek-V4-Flash,RTX PRO 6000 Blackwell,4,4,1048576,200,true,56.9,57.4,,125,212,replace with raw log when available
```

`tps_mean` は生ログから平均を計算できる場合のみ埋めてください。範囲しかない場合は空欄でよいです。

### 4.4 parallel_scaling.csv

推奨スキーマ：

```csv
run_id,date,source,concurrency,output_tokens_each,cuda_graphs,aggregate_tps,scaling_vs_1,per_request_tps_mean,notes
parallel_1,2026-05-15,README-summary,1,200,true,55.6,1.0,55.6,
parallel_2,2026-05-15,README-summary,2,200,true,107.1,1.9,53.55,
parallel_4,2026-05-15,README-summary,4,200,true,214.9,3.9,53.725,
parallel_8,2026-05-15,README-summary,8,200,true,400.7,7.2,50.0875,
```

注意：

- `per_request_tps_mean` は `aggregate_tps / concurrency` として算出した派生値であることを `benchmarks/README.md` に明記する
- 生ログがある場合は、各requestごとのTPS分布も別CSVに残す

### 4.5 cuda_graphs_ablation.csv

推奨スキーマ：

```csv
run_id,date,source,output_tokens,cuda_graphs,tps_min,tps_max,tps_mean,speedup_vs_off,notes
cuda_graphs_on_200,2026-05-15,README-summary,200,true,56.9,57.4,,,production path
cuda_graphs_off_200,2026-05-15,README-summary,200,false,9.6,9.6,,,baseline without CUDA Graphs
```

`speedup_vs_off` は、生ログまたは明確な代表値を決めた上で計算してください。単純にREADMEの `6×` を使う場合は `source=README-summary` と明記してください。

### 4.6 ttft.csv

TTFTは売れ筋グラフになります。必ず分離してください。

推奨スキーマ：

```csv
run_id,date,source,scenario,context_length,output_tokens,concurrency,ttft_ms_min,ttft_ms_p50,ttft_ms_p95,ttft_ms_max,notes
ttft_summary,2026-05-15,README-summary,single_request,1048576,,,125,,,212,range only; replace with raw distribution
```

可能なら最低30回測定し、以下を埋めてください。

- min
- p50
- p90
- p95
- p99
- max
- mean
- std

### 4.7 context_length_sweep.csv

1M contextの価値を伝えるため、コンテキスト長ごとの挙動を記録してください。

推奨スキーマ：

```csv
run_id,date,context_length,hicache_enabled,disk_offload_enabled,cuda_graphs,status,tps_mean,ttft_ms_p50,peak_gpu_memory_gb,disk_kv_size_gb,notes
ctx_32k,2026-05-15,32768,true,true,true,pass,,,,,
ctx_393k,2026-05-15,393216,true,true,true,pass,,,,,
ctx_1m,2026-05-15,1048576,true,true,true,pass,,,,,
```

測定できない値は空欄でよいですが、`status` と `notes` は必ず埋めてください。

### 4.8 disk_offload_metrics.csv

DiskOffloadの信頼性を上げるため、可能なら以下を測定してください。

```csv
run_id,date,context_length,disk_backend,disk_used_gb,page_count,read_ops,write_ops,cache_hit_rate,lru_evictions,avg_read_ms,p95_read_ms,avg_write_ms,p95_write_ms,notes
```

測定できない場合は、まず `disk_used_gb`, `page_count`, `lru_evictions` だけでもよいです。

### 4.9 生成するグラフ

`benchmarks/scripts/plot_benchmarks.py` を追加し、CSVから以下を生成してください。

1. `docs/assets/parallel_scaling.svg`
   - x軸：concurrency
   - y軸：aggregate TPS
   - 1/2/4/8並列のスケーリングを見せる
   - `400.7 tok/s @ 8 concurrent` を注記

2. `docs/assets/cuda_graphs_ablation.svg`
   - CUDA Graphs ON/OFF のTPS比較
   - “CUDA Graphs are critical for production throughput” を視覚化

3. `docs/assets/single_request_throughput.svg`
   - 100 tokens / 200 tokens の throughput range
   - min/maxレンジをエラーバー表示

4. `docs/assets/ttft_range.svg`
   - TTFT 125–212ms のrange
   - 生ログがあればp50/p95も表示

5. `docs/assets/context_length_sweep.svg`
   - 32K / 393K / 1M のpass/failおよびTPS/TTFT
   - 1M到達を明確に見せる

6. `docs/assets/architecture.svg`
   - GPU L1 → DRAM L2 → Optane SSD L3 の階層図
   - Mermaidでもよいが、README表示用にSVGも用意する

### 4.10 グラフ作成ルール

- グラフはCSVから再生成可能にする
- PNGだけを置かない
- SVGを優先する
- READMEに埋め込む画像は `docs/assets/*.svg`
- 生データ・生成スクリプト・生成済み画像をセットでコミットする
- グラフ内の数値とREADMEの数値が一致しているか `validate_results.py` で検証する
- データがsummary由来の場合は、グラフまたはキャプションに `summary data` と明記する
- 生ログがないのにp50/p95などを捏造しない

---

## 5. docs/benchmark.md の内容

`docs/benchmark.md` を新規作成し、以下を書く。

```markdown
# Benchmark Methodology

## Hardware

Describe GPUs, CPU, RAM, SSD, OS.

## Software

Describe Docker image, sglang image, CUDA, PyTorch, driver, model checkpoint.

## Model

DeepSeek-V4-Flash FP8 checkpoint, model source, local mount path placeholder.

## Runtime configuration

List:
- tensor parallel size
- context length
- kv cache dtype
- page size
- hicache ratio
- disk offload directory
- CUDA Graphs setting

## Benchmark scenarios

1. Single request throughput
2. Parallel scaling
3. CUDA Graphs ablation
4. TTFT measurement
5. Context length sweep
6. DiskOffload metrics

## Measurement rules

- Warmup count
- Number of repeated runs
- Prompt template
- Sampling parameters
- Whether TTFT includes queueing
- Whether throughput is output-token TPS only or total-token TPS
- Whether aggregate TPS includes all concurrent requests

## Results

Embed generated graphs.

## Raw data

Link to `benchmarks/results/...`

## Known caveats

Be explicit:
- Hardware-specific result
- Disk latency matters
- CUDA Graphs ON is production path
- Summary-only numbers should be replaced by raw logs
```

---

## 6. docs/architecture.md の内容

`docs/architecture.md` を作成し、READMEより詳細に書く。

必須項目：

- Problem: GPU KV cache memory pressure
- DeepSeek-V4-Flash memory characteristics
- Why host-side KV pool is needed
- Why SSD offload is useful
- L1/L2/L3 cache hierarchy
- `DeepSeekV4TokenToKVPoolHost`
- `DiskOffloadBackend`
- `HiRadixCache` patch
- `hybrid_pool_assembler.py` patch
- Data flow:
  - prefill
  - decode
  - eviction
  - reload
- Failure modes:
  - OOM
  - page mismatch
  - disk latency bottleneck
  - incompatible HiCache API version
- Future optimization:
  - async IO
  - io_uring
  - mmap
  - pinned host memory
  - page compression
  - smarter eviction policy

---

## 7. sglang-diskkv/README.md 改修

現在の `sglang-diskkv/README.md` は短く、プロジェクト初期メモのように見えます。これをサブモジュールREADMEとして整備してください。

推奨構成：

```markdown
# sglang-diskkv

SSD-backed KV cache offload backend for sglang HiCache, developed for DeepSeek-V4-Flash long-context inference on NVIDIA Blackwell.

## What this module provides

- `DiskOffloadBackend`
- DeepSeek-V4 host-side KV pool support
- HiCache compatibility patches
- Docker image for DS4V + DiskOffload

## Relationship to root project

This directory contains the runtime modifications used by LnaLang4U.

## Current status

Implemented:
- page-level get/set/exists
- batch get/set/exists
- LRU eviction
- JSON index persistence
- v1/v2 HiCache compatibility

Experimental:
- 1M-context path
- Optane-backed L3 KV cache
- CUDA Graphs production path

Planned:
- async IO
- better benchmark instrumentation
- upstreaming strategy

## Usage

Link to:
- `../README.md`
- `RUN_INFERENCE.md`
- `../docs/architecture.md`
- `../docs/benchmark.md`
```

注意：

- 「目標: 1GPU, 1M context」は、現在の成果と混同されるので `Historical design goal` または `Original goal` として扱う
- 現在のREADMEが4 GPU構成を示しているなら、1GPU目標と4GPU実績を明確に分ける

---

## 8. DESIGN.md 改修

`sglang-diskkv/DESIGN.md` は設計メモとして価値がありますが、現在はPhase計画のまま見える部分があります。

改修方針：

- 冒頭にステータスを追加する

```markdown
> Status: Implemented prototype used by LnaLang4U benchmark runs. Some optimization phases remain experimental.
```

- `Goal` を `Original goal` と `Current implementation` に分ける
- Phase 1〜4を以下に変える

```markdown
## Implementation status

| Area | Status | Notes |
|---|---|---|
| DiskOffloadBackend | Implemented | page-level and batch operations |
| LRU eviction | Implemented | JSON index based |
| HiCache v1/v2 compatibility | Implemented | compatibility layer |
| DeepSeek-V4 host pool | Implemented | SWA/c4/c128/indexer/compressor pools |
| Async IO | Planned | future optimization |
| Benchmark instrumentation | In progress | raw metrics needed |
```

- 日本語中心の説明は英語化する
- 詳細設計として残す価値があるため削除しない
- ルートREADMEから `docs/architecture.md` 経由で参照する

---

## 9. RUN_INFERENCE.md 改修

`sglang-diskkv/RUN_INFERENCE.md` は実用上重要です。以下を修正してください。

### 9.1 ローカル固有パスの置換

現在のような個人環境パスは公開向けには避ける。

置換前：

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
```

置換後：

```bash
export PROJECT_DIR=/path/to/LnaLang4U
export MODEL_DIR=/path/to/DeepSeek-V4-Flash-FP8
export DISKKV_DIR=/path/to/diskkv

cd "$PROJECT_DIR"
```

### 9.2 手順を英語化

ファイル全体を英語に統一してください。

### 9.3 Safety checks

起動前チェックを追加してください。

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
df -h "$DISKKV_DIR"
```

### 9.4 Launch tiers

以下の段階を明確化してください。

1. Smoke test: 32K context
2. Medium test: 393K context
3. Full test: 1M context
4. Benchmark mode

### 9.5 Troubleshooting links

失敗時は `docs/troubleshooting.md` へ誘導してください。

---

## 10. UNRESOLVED.md / LOCAL_MODEL_HANDOFF.md の扱い

これらは公開リポジトリでは慎重に扱ってください。

方針：

- 個人環境名、未整理ログ、秘密情報、ローカルパスがないか確認
- 公開してよい技術的な問題は `docs/troubleshooting.md` または `docs/known-issues.md` に整理
- 内部引き継ぎ文書としての性質が強い場合は、READMEから直接目立たせない
- ただしOSSではKnown Issuesが信頼につながるため、隠さず、整理して公開する

推奨：

```text
sglang-diskkv/UNRESOLVED.md
→ docs/known-issues.md に統合またはリンク

sglang-diskkv/LOCAL_MODEL_HANDOFF.md
→ 公開向けに sanitize し、必要なら docs/troubleshooting.md に統合
```

---

## 11. Reproducibility強化

以下のファイルを追加してください。

### 11.1 docs/reproducibility.md

内容：

- Required hardware
- Tested hardware
- Model download
- Docker build
- Exact launch command
- Benchmark command
- Expected output
- How to verify DiskOffload is active
- How to compare with README numbers

### 11.2 benchmarks/README.md

内容：

- データ構造
- CSVスキーマ
- どの数値がraw log由来か
- どの数値がREADME summary由来か
- グラフ再生成方法

### 11.3 validate_results.py

最低限、以下を検証してください。

- CSVが存在する
- 必須列が存在する
- READMEに埋め込む主要数値とCSVが一致する
- 生成済みSVGが存在する
- `source=README-summary` の行がある場合、警告を出す

---

## 12. README内に入れるグラフ表示例

READMEには以下を追加してください。

```markdown
## Performance

The following figures are generated from machine-readable benchmark data in [`benchmarks/results`](benchmarks/results).

### Parallel scaling

![Parallel scaling](docs/assets/parallel_scaling.svg)

### CUDA Graphs ablation

![CUDA Graphs ON vs OFF](docs/assets/cuda_graphs_ablation.svg)

### Single-request throughput

![Single request throughput](docs/assets/single_request_throughput.svg)

### TTFT

![TTFT range](docs/assets/ttft_range.svg)

For methodology and raw data, see [`docs/benchmark.md`](docs/benchmark.md).
```

---

## 13. Claims audit

READMEとdocs全体で、以下を確認してください。

### 13.1 Strong claims requiring evidence

以下の表現を使う場合、必ず根拠を添える。

- first
- fastest
- production-speed
- real-time
- near-linear scaling
- 1M context
- 6× improvement
- 400 tok/s
- 63 tok/s
- low latency
- practical

### 13.2 推奨する注釈

```markdown
Measured on 4× NVIDIA RTX PRO 6000 Blackwell GPUs with TP=4. Results may vary depending on storage latency, driver versions, CUDA Graphs settings, and model checkpoint format.
```

### 13.3 Avoid ambiguity

`TPS` は必ず定義してください。

- output-token throughput
- aggregate throughput
- per-request throughput
- warm run or cold run
- CUDA Graphs ON/OFF
- context length
- output token count

---

## 14. Release準備

`v0.1.0` のReleaseを準備してください。

Release title:

```text
v0.1.0 — DeepSeek-V4-Flash 1M-context Blackwell + SSD KV offload prototype
```

Release notes draft:

```markdown
Initial public release of LnaLang4U.

Highlights:
- DeepSeek-V4-Flash inference on NVIDIA Blackwell
- 1,048,576-token context configuration
- SSD-backed KV cache offload via custom DiskOffloadBackend
- DeepSeek-V4 host-side KV pool support
- HiCache compatibility patches
- Benchmark summary: 63 tok/s single-request, 400.7 tok/s aggregate at 8 concurrent requests
- Reproducibility docs and benchmark data included

Known limitations:
- Hardware-specific prototype
- Requires Blackwell/SM120-compatible environment
- DiskOffload performance depends heavily on SSD latency
- Benchmark raw data is being expanded
```

Releaseは、READMEとbenchmark data整備後に作成してください。

---

## 15. Community files

売れっ子OSSにするため、以下を追加してください。

### 15.1 CONTRIBUTING.md

内容：

- how to report issues
- how to submit PRs
- coding style
- benchmark contribution format
- hardware result submission template

### 15.2 SECURITY.md

内容：

- security issue reporting
- do not post private model tokens or local secrets in issues
- dependency/security caveats

### 15.3 CITATION.cff

研究用途で引用される可能性があるため追加してください。

ただし、著者名・バージョン・DOI等は確認できる範囲で記載し、不明な値は入れない。

### 15.4 LICENSE

ライセンスはメンテナー承認なしに決めないでください。

READMEには、ライセンス未確定なら以下のように書く。

```markdown
## License

License information will be added after maintainer confirmation.
```

---

## 16. Suggested final file changes

最低限、以下を変更・追加してください。

```text
README.md
sglang-diskkv/README.md
sglang-diskkv/DESIGN.md
sglang-diskkv/RUN_INFERENCE.md
docs/architecture.md
docs/benchmark.md
docs/reproducibility.md
docs/troubleshooting.md
docs/known-issues.md
docs/assets/*.svg
benchmarks/README.md
benchmarks/results/2026-05-15-blackwell-4x-rtx-pro-6000/*.csv
benchmarks/results/2026-05-15-blackwell-4x-rtx-pro-6000/metadata.yaml
benchmarks/scripts/plot_benchmarks.py
benchmarks/scripts/validate_results.py
CONTRIBUTING.md
SECURITY.md
CITATION.cff
```

ライセンスは保留。

---

## 17. Acceptance criteria

作業完了条件：

- README冒頭30秒で価値が伝わる
- Achievement tableが正しいMarkdownになっている
- すべての主要数値に測定条件がある
- グラフがREADMEに表示される
- グラフがCSVから再生成できる
- raw data / metadata / graph script が揃っている
- ローカル個人パスが公開文書から除去されている
- 日本語の作業メモが英語の公開文書に整理されている
- 1GPU目標、4GPU実績、1M context実績が混同されていない
- `production-speed`, `first`, `near-linear`, `6×` などの強い主張が検証可能になっている
- About / Topics / Social Preview / Release draft まで整っている
- 未確定情報は捏造せず `TBD` または `summary-only` と明示している

---

## 18. Final instruction

Do not merely make the repository look prettier.

Make it more credible.

Make it easier to reproduce.

Make it easier to cite.

Make it easier to share.

Make it easier for a serious inference engineer to say:

> “This is real. I want to try it.”
