# DiskOffload: SSD KV Cache Backend for sglang

> **Status:** Implemented prototype used by LnaLang4U benchmark runs. Some optimization phases remain experimental.

## Original goal

Implement an SSD-backed L3 storage backend for sglang HiCache, inspired by d4-server's `--kv-disk-dir` + `--kv-disk-space-mb`. Enable 1M-context inference without OOM by offloading KV cache pages to local SSD.

## Current implementation

The prototype runs on **4 GPUs** with sglang for production throughput, while the SSD KV cache offload design remains faithful to the original single-GPU concept. See the [root README](../README.md) for current capabilities and [docs/architecture.md](../docs/architecture.md) for design details.

## Architecture

```
sglang ModelRunner
  └─ HiCacheController (L2: GPU ↔ CPU)
       └─ HiCacheStorage (L3: CPU ↔ Storage)
            └─ DiskOffloadBackend ★NEW★ (ローカル SSD)
```

HiCache 既存の3層構造のうち、L3 層に `disk_offload` バックエンドを追加する。
データパス: `GPU → CPU (L2) → Disk (L3, 新規)`

> **Status:** Implemented prototype. See the [root README](../README.md) for current capabilities.

## Interface

`HiCacheStorage` ABC を実装:

| Method | 役割 |
|--------|------|
| `get(key) -> Tensor` | ページをディスクから読む |
| `set(key, value) -> bool` | ページをディスクに書く |
| `exists(key) -> bool` | ページの存在確認 |
| `batch_get(keys) -> List[Tensor]` | バッチ読み込み |
| `batch_set(keys, values) -> bool` | バッチ書き込み |
| `batch_exists(keys) -> int` | 連続存在確認 |
| `clear()` | 全データ削除 |

## Data Layout

```
{kv-disk-dir}/
  ├── index.json              # メタデータインデックス (LRU順, サイズ)
  └── pages/
      ├── {key1}.pt           # 個別 KV page ファイル (torch.save)
      ├── {key2}.pt
      └── ...
```

- key = `"{pool_name}/{page_id}"` 形式 (例: `"kv/00000000"`, `"swa/00000042"`)
- 各ページは個別ファイル (`torch.save` / `torch.load` で高速IO)
- `index.json` で全ページの最終アクセス時刻とサイズを管理

## Budget Management

ds4-server から継承する概念:

| ds4-server | DiskOffloadBackend |
|------------|-------------------|
| `--kv-disk-dir` | `disk_offload_dir` (引数) |
| `--kv-disk-space-mb` | `max_disk_space_mb` (引数) |
| SHA-1 content addressing | 不要 (sglang 側が key 管理) |
| LRU eviction (kv_cache_evict) | LRU eviction (index.json ベース) |

## 実装計画

### Phase 1: 最小限の DiskOffloadBackend
- `HiCacheStorage` を実装する `DiskOffloadBackend` クラス
- 単一キーの `get` / `set` / `exists`
- ファイル単位のシリアライズ (torch.save/load)
- バックエンドファクトリに登録 (`"disk_offload"`)

### Phase 2: バッチ操作 + 予備動作確認
- `batch_get` / `batch_set` / `batch_exists`
- `batch_exists_v2` / `batch_get_v2` / `batch_set_v2` (v2 インターフェース)
- 起動引数の追加 (`--kv-disk-dir`, `--kv-disk-space-mb`)

### Phase 3: 予算管理 + エビクション
- ディスク使用量の追跡
- LRU エビクション
- ファイル削除によるディスク領域解放
- インデックスの永続化

### Phase 4: パフォーマンス最適化
- 非同期IO (スレッドプール)
- バッチ読み込みの並列化
- ページサイズに応じたチャンク管理
