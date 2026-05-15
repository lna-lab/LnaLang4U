# Unresolved Issues — Development Notes

> **Note:** Internal development log tracking technical challenges. Contains Japanese developer notes. Public users may find [docs/troubleshooting.md](../docs/troubleshooting.md) more helpful.

## Codex フォローアップ — 2026-05-15 12:16 JST

### 結論

今回の `fused_moe_triton/layer.py:_load_w13` エラーは、まず `layer.py` の shard_dim を直す問題として扱わないでください。現時点の第一原因は **モデル実体と `SGLANG_DSV4_FP4_EXPERTS` の指定が食い違っていること**です。

現在の `launch.sh sglang-diskkv` は次の順でモデルを選びます。

```bash
FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash"
[[ -f "${FP8_DIR}/config.json" ]] || FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash-FP8"
```

つまり、存在すれば 149GB の `deepseek-ai/DeepSeek-V4-Flash` を優先します。一方で Docker launch環境には以下が固定されています。

```bash
-e SGLANG_DSV4_FP4_EXPERTS=0
```

これは「FP4-to-FP8 変換済み checkpoint」を読む設定です。149GB の `deepseek-ai/DeepSeek-V4-Flash` は expert weight が packed FP4/int8 形状なので、この設定と矛盾します。

### 実測した形状

149GB 側:

```text
models/DeepSeek-V4-Flash
layers.0.ffn.experts.0.w1.weight: (2048, 2048), torch.int8
layers.0.ffn.experts.0.w3.weight: (2048, 2048), torch.int8
layers.0.ffn.experts.0.w2.weight: (4096, 1024), torch.int8
```

274GB 側:

```text
models/DeepSeek-V4-Flash-FP8
layers.0.ffn.experts.0.w1.weight: (2048, 4096), torch.float8_e4m3fn
layers.0.ffn.experts.0.w3.weight: (2048, 4096), torch.float8_e4m3fn
layers.0.ffn.experts.0.w2.weight: (4096, 2048), torch.float8_e4m3fn
```

今回のエラー:

```text
RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048)
```

これは 149GB の packed FP4/int8 weight `(2048, 2048)` を、FP8 変換済み用に確保された `(2048, 4096)` 側へコピーしようとしている症状と一致します。

### 重要な訂正

`--moe-runner-backend triton` では `get_moe_runner_backend().is_triton_kernels()` は False です。したがって、この実行では `_load_w13()` の `self.use_triton_kernels` by transpose 分岐は本筋ではありません。

そのため、次の作業は避けてください。

- `loaded_weight` 側だけ shard_dim を変える
- `_load_w13()` の transpose/narrow を勘で変更する
- `fused_moe_triton/layer.py` を Dockerfile に追加コピーして先に焼く

まずモデル選択と FP4/FP8 expert 設定を一致させるのが先です。

### 推奨解決策 A: 274GB の FP8 変換済みモデルを使う

DiskKV の P0 smoke test ではこちらを推奨します。`SGLANG_DSV4_FP4_EXPERTS=0` と整合します。

`launch.sh` の `sglang-diskkv)` ブロックを、少なくとも P0 中は次の優先順にしてください。

```bash
FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash-FP8"
[[ -f "${FP8_DIR}/config.json" ]] || FP8_DIR="${MODEL_DIR}/DeepSeek-V4-Flash"
```

同じ修正を `sglang)` ブロックにも入れると、通常launchと DiskKV launchでモデル差が消えて比較しやすくなります。

verify:

```bash
cd /path/to/LnaLang4U
BUILD_DISKKV_IMAGE=1 \
CONTEXT_LENGTH=32768 \
HICACHE_RATIO=1.25 \
DISKKV_MB=65536 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

### 代替解決策 B: 149GB の packed FP4 expert モデルを使う

149GB の `DeepSeek-V4-Flash` を使うなら、少なくとも次の固定指定をやめてください。

```bash
-e SGLANG_DSV4_FP4_EXPERTS=0
```

packed FP4 expert として読むなら:

```bash
-e SGLANG_DSV4_FP4_EXPERTS=1
```

ただし、この経路は FP4 expert 用の追加前提に入るため、DiskKV の問題切り分けとしては遠回りです。まずは 274GB の `DeepSeek-V4-Flash-FP8` で HiCache/DiskOffload を進める方が安全です。

### 形状verifyコマンド

ローカルモデルは、次のコマンドでモデル実体をverifyしてからlaunchしてください。

```bash
docker run --rm \
  -v /path/to/LnaLang4U \
  --entrypoint python3 \
  lmsysorg/sglang:deepseek-v4-blackwell \
  -c 'from safetensors import safe_open; from pathlib import Path; import json
base=Path("/m")
wm=json.loads((base/"model.safetensors.index.json").read_text())["weight_map"]
for key in ["layers.0.ffn.experts.0.w1.weight","layers.0.ffn.experts.0.w3.weight","layers.0.ffn.experts.0.w2.weight"]:
    fn=wm[key]
    with safe_open(base/fn, framework="pt", device="cpu") as f:
        t=f.get_tensor(key)
        print(key, tuple(t.shape), t.dtype)'
```

### 次の判定

274GB の FP8 変換済みモデルに切り替えても同じ `_load_w13()` エラーが出るなら、その時点で初めて `fused_moe_triton/layer.py` に shape ログを入れてください。

見るべき値:

```python
logger.error(
    "MoE load debug: shard_id=%s shard_dim=%s expert_data=%s loaded=%s "
    "use_presharded=%s use_triton_kernels=%s quant_method=%s",
    shard_id,
    shard_dim,
    tuple(expert_data.shape),
    tuple(loaded_weight.shape),
    self.use_presharded_weights,
    self.use_triton_kernels,
    getattr(param, "quant_method", None),
)
```

ただし、現時点では `layer.py` パッチより **モデル選択修正** が優先です。

## current state: fused_moe_triton weight loader の transposed shard_dim 不整合

### エラー内容（deepseek-ai モデルでverify）

```
RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048)
  at non-singleton dimension 1
  File: fused_moe_triton/layer.py:448, _load_w13
```

### 原因の詳細

deepseek-ai FP8 チェックポイントは各 expert の w1/gate_proj, w3/up_proj を
**個別テンソル**として保存している (`layers.N.ffn.experts.M.w1.weight`)。

sglang の `deepseek_v4.py` は `stacked_params_mapping` を使ってこれらを
`gate_up_proj` (w13 combined) パラメータにマージする。

しかし、`fused_moe_triton/layer.py` の `_load_w13` は `shard_dim`
(TP sharding 用の次元) を `SHARD_ID_TO_SHARDED_DIM` + transposed 補正で
決定するが、これが `loaded_weight` と `expert_data` で異なるレイアウトを
持つ場合に不整合を起こす。

```
expert_data = param.data[expert_id]  # shape [2, 512, 4096] (w13 combined, transposed)
shard_dim = 1  (transposed: shard_dim=0 → 1)
shard_size = 512 // 2 = 256

loaded_weight.shape = [2048, 4096]  # チェックポイントの個別 w1
loaded_weight.narrow(shard_dim=1, 0, 256)  # → shape [2048, 256] ❌ 次元が逆
expert_data[0:256, :]  # shape [256, 4096] ← これに上記を copy_() しようとして失敗
```

**期待される動作**: `loaded_weight` を intermediate_size (dim=0) 方向に narrow すべき。
**実際の動作**: `shard_dim=1` が loaded_weight にも適用され、hidden_size 方向に narrow してしまう。

### 影響

- deepseek-ai モデル (149GB, 個別 w1/w3): このエラーが直接出る
- sgl-project モデル (274GB, 形式不明): 同様または別の次元エラー。scheduler が
  silent exit するためエラーメッセージがキャプチャ不能

### 試したが解決しなかったこと

- `SGLANG_APPLY_CONFIG_BACKUP=0`: 別の次元エラー (32 vs 128) が出る
- sgl-project モデルへの切り替え: silent exit でデバッグ不能
- scheduler の例外キャプチャ全手法: 全て失敗

### 求める解決策 (Codex へ)

1. **本修正**: `fused_moe_triton/layer.py` の `_load_w13` で、`loaded_weight` の
   narrow に使う `shard_dim` を非 transposed で計算し直す方法
   - `loaded_weight` は常に `[intermediate_size, hidden_size]` 形式
   - `expert_data` は `[2, interm_size_pp, hidden_size]` (w13 combined, transposed)
   - 両者で次元の意味が異なる
2. **ワークアラウンド**: 
   - `--disable-cuda-graph` → 影響なしとverify済み
   - `--moe-runner-backend` の指定変更で weight_loader のパスを変更できるか？
   - `SGLANG_DSV4_MODE=2601` と `2604` で weight loading パスが変わるか？

### verify済み動作環境（本件とは無関係に動作）

```
sglang + FP8 4GPU (HiCache なし): 正常動作 ✅
→ validateに使ったコマンド:
  docker run --name sglang-dsv4-sm120 --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,2,3,4 --shm-size=64g --ipc=host --network host \
    -v /path/to/model:/workspace/model:ro \
    -v /path/to/kernel:/dsv4:ro -e PYTHONPATH=/dsv4 \
    lmsysorg/sglang:deepseek-v4-blackwell \
    python3 -m sglang.launch_server --model-path /workspace/model --host 0.0.0.0 --port 9000 ... \
    (--fp8-gemm-backend triton --moe-runner-backend triton --attention-backend compressed ...)
```

このベースラインでは `--enable-hierarchical-cache` なしで 393K context まで動作verify済み。
HiCache + DiskOffload 追加時にのみ weight loading が失敗する（上記の transposed shard_dim 問題）。

## 直前のattempt結果: P0 smoke test (2回目)

### Codex パッチ適用状況

| 修正 | 状態 |
|------|------|
| `disk_offload_backend.py` に Pool* 互換 shim | ✅ 実装・テスト済み |
| `_component_key()` の安全な文字列化 | ✅ 実装済み |
| Docker ビルド | ✅ 成功 |
| Docker 内 import テスト | ✅ `DiskOffloadBackend` import OK, get/set OK |

### P0 Step 2 smoke test 結果 ❌ (2回連続)

**パターン（毎回同一）:**

```
[10s] Up 9 seconds      GPU0: 23MiB   ← モデル読み込み開始
[20s] Up 20 seconds     GPU0: 72455MiB ← 重み読み込み完了 (83GB/GPU)
[30s] Up 30 seconds     GPU0: 23MiB   ← GPU メモリ解放（scheduler 終了）
[40s] Up 43 seconds     GPU0: 23MiB   ← watchdog 待機中
[60s] Up About a minute GPU0: 23MiB   ← そのままハング → その後 exit(0)
```

**観測された stdout/stderr:**
- モデル重み読み込み成功
- `Memory pool end. avail mem=13.96 GB`
- `max_total_num_tokens=...` (DSV4 pool sizing 成功)
- Gloo ランク接続完了
- その後 `POST /generate 500 Internal Server Error` → scheduler 死亡の証拠
- `Scheduler hit an exception` のログが **一切出力されない**

### デバッグattempt一覧（全て失敗）

| 方法 | 結果 |
|------|------|
| stdout/stderr 両方をファイルキャプチャ | scheduler の例外トレース無し |
| `subprocess.Popen` monkeypatch | sglang は `mp.start_process()` (multiprocessing) を使用 |
| `run_scheduler_process` に try/except + file write | パッチが子プロセスに伝播せず |
| multiprocessing.BaseProcess.run の monkeypatch | プロセスlaunch前に monkeypatch が完了せず |
| `--log-level DEBUG` | 追加情報なし |
| `docker exec` でコンテナ内verify | zombie `[python3] <defunct>` のみverify |
| stderr ファイルリダイレクト (entrypoint bash -c) | 子プロセスの stderr は multiprocessing pipe 経由、ログに到達せず |

### 原因の推定

scheduler 子プロセスは `mp.start_process()` (torch.multiprocessing または sglang fork) 
でlaunchされる。子プロセスの stderr は `multiprocessing.Process` の内部 pipe で
親プロセスに送られ、`logging` モジュールで出力される。

しかし、scheduler の init が途中で **silent abort** (例外ではなく `sys.exit(0)`
または `os._exit(0)`) している可能性が高い。これにより:
- 例外トレースバックは生成されない
- 子プロセスは exit(0) で正常終了
- 親は sigquit を受信 → "child failed" とログ
- GPU メモリは解放される

`sys.exit(0)` が呼ばれる場所の推定:
1. `HostKVCache.__init__()` の `assert self.size > device_pool.size` → 失敗して exit
2. `init_kv_buffer()` でのメモリ確保失敗 → abort
3. `attach_hybrid_pool_to_unified_cache()` での DS4V 分岐 → `DeepSeekV4TokenToKVPoolHost` 内の未初期化属性アクセス

### 次の一手 (Codex への依頼)

1. **子プロセス内の silent abort の特定方法を提供**
   - `sys.exit(0)` を仕込まれている可能性のある場所のリスト
   - または、`atexit` / `sys.excepthook` を使って終了原因をファイルに記録するブートストラップパッチ
2. **`HostKVCache.__init__()` の assert line 188-189 の回避方法**
   - DS4V の device_pool.size と host pool の size 計算が適切かverify
   - `self.size > device_pool.size` の条件を DS4V で満たすfor最小 `--hicache-size`
3. **scheduler init の正常終了パスverify**
   - `scheduler.py` の `init_cache_with_memory_pool` → `HiRadixCache.__init__` → 戻り
   - その後どのメソッドが呼ばれるかverify

### validateコマンド (次回用)

```bash
# P0 smoke test
cd /path/to/LnaLang4U
docker build -f sglang-diskkv/Dockerfile.sglang-dsv4 -t sglang-dsv4-diskkv:latest .
docker rm -f sglang-dsv4-diskkv 2>/dev/null
CONTEXT_LENGTH=32768 HICACHE_RATIO=1.25 DISKKV_MB=65536 GPUS=0,2,3,4 PORT=9000 \
./launch.sh sglang-diskkv
```
