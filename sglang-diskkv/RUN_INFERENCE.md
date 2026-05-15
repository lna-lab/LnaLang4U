# 推論起動手順

## 1. イメージをビルド

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
docker build \
  -f sglang-diskkv/Dockerfile.sglang-dsv4 \
  -t sglang-dsv4-diskkv:latest \
  .
```

## 2. まず短い context で起動

```bash
cd /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F
CONTEXT_LENGTH=32768 \
HICACHE_RATIO=1.25 \
DISKKV_MB=65536 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

重要な起動オプションは `launch.sh` 側で入ります。

```bash
--enable-hierarchical-cache
--hicache-io-backend direct
--hicache-mem-layout page_first
--hicache-storage-backend disk_offload
```

ログに以下が出れば、今回のパッチが使われています。

```text
Initialized DeepSeek-V4 host pool
DiskOffloadBackend: dir=/diskkv
```

## 3. 疎通確認

別ターミナルで:

```bash
curl -s http://127.0.0.1:9000/v1/models
```

推論:

```bash
curl -s http://127.0.0.1:9000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello, introduce yourself in one sentence.",
    "sampling_params": {
      "max_new_tokens": 32,
      "temperature": 0
    }
  }'
```

## 4. DiskOffload 確認

```bash
find /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F/diskkv -maxdepth 2 -type f | head
du -sh /media/tonoken/Optane_DATA/Sm120-LNALAB-V4F/diskkv
```

`index.json` と `pages/*.pt` が増えれば、L3 への退避が動いています。

## 5. 長い context へ段階的に上げる

短い context の推論が通ってから:

```bash
CONTEXT_LENGTH=393216 \
HICACHE_RATIO=1.5 \
DISKKV_MB=524288 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

1M context は最後に:

```bash
CONTEXT_LENGTH=1048576 \
MEM_FRACTION_STATIC=0.80 \
HICACHE_RATIO=1.25 \
DISKKV_MB=1048576 \
GPUS=0,2,3,4 \
PORT=9000 \
./launch.sh sglang-diskkv
```

失敗時は `LOCAL_MODEL_HANDOFF.md` の P0/P1 チェックリストに沿ってログを取ってください。
