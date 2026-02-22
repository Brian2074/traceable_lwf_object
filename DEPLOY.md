# 🚀 Multi-Machine Federated Learning Deployment Guide

## Architecture

| Machine | Role              | RAM   | GPU       |
|---------|-------------------|-------|-----------|
| 本機 A  | Server + Client 0 | 32 GB | RTX 2060  |
| 電腦 B  | Client 1          | 16 GB | RTX 2060  |
| 電腦 C  | Client 2          | 16 GB | RTX 2060  |

> [!IMPORTANT]
> 請先在本機用 `ip addr` 或 `hostname -I` 查出本機的 LAN IP（例如 `192.168.1.100`），下面的指令都用這個 IP。

---

## Step 0：在三台電腦上都準備好專案

```bash
# 在每一台電腦上
git clone <your-repo-url> traceable_lwf_object
cd traceable_lwf_object
```

---

## Step 1：在三台電腦上產生資料集

```bash
# 安裝 python dependencies (或用 Docker 內建)
pip install -r requirements.txt

# 下載 + 轉換資料集
python tools/hf2yolo.py

# 切成 3 個 Non-IID 分片
python tools/split_non_iid.py --data_dir datasets/spscd_coco_yolo --clients 3 --alpha 0.5
```

> [!TIP]
> 如果網路慢，可以直接從本機用 `scp -r datasets/ user@電腦B:~/traceable_lwf_object/` 把資料傳過去。

---

## Step 2：在三台電腦上 Build Docker Image

```bash
cd traceable_lwf_object
docker build -t fedrep-yolo:latest -f docker/Dockerfile .
```

---

## Step 3：啟動 Server（本機 A）

```bash
cd traceable_lwf_object/docker

NUM_CLIENTS=3 \
ROUNDS=10 \
TASKS=1 \
TASKS_EPOCH=3 \
EXP_NAME=distributed_3client \
  docker compose -f server-compose.yml up
```

Server 會開始監聽 `0.0.0.0:8080`，等待 3 個 Client 連入。

---

## Step 4：啟動 Client 0（本機 A，開另一個終端機）

```bash
cd traceable_lwf_object/docker

SERVER_ADDRESS=<本機IP>:8080 \
CLIENT_ID=0 \
BATCH_SIZE=16 \
BODY_EPOCHS=5 \
HEAD_EPOCHS=5 \
EXP_NAME=distributed_3client \
  docker compose -f client-compose.yml up
```

---

## Step 5：啟動 Client 1（電腦 B）

```bash
cd traceable_lwf_object/docker

SERVER_ADDRESS=<本機IP>:8080 \
CLIENT_ID=1 \
BATCH_SIZE=16 \
BODY_EPOCHS=5 \
HEAD_EPOCHS=5 \
EXP_NAME=distributed_3client \
  docker compose -f client-compose.yml up
```

---

## Step 6：啟動 Client 2（電腦 C）

```bash
cd traceable_lwf_object/docker

SERVER_ADDRESS=<本機IP>:8080 \
CLIENT_ID=2 \
BATCH_SIZE=16 \
BODY_EPOCHS=5 \
HEAD_EPOCHS=5 \
EXP_NAME=distributed_3client \
  docker compose -f client-compose.yml up
```

---

## 🎉 開始訓練

當 3 個 Client 全部連上 Server 後，Flower 會自動開始第一輪 FedRep 訓練！

你會在 Server 終端機看到：
```
Round 1 → Task 1 (10 classes), 3 clients
```

每個 Client 的訓練 Log 會直接輸出在各自的終端機裡。

---

## Troubleshooting

| 問題 | 解法 |
|------|------|
| Client 連不上 Server | 確認防火牆有開 port 8080：`sudo ufw allow 8080` |
| CUDA out of memory | 把 `BATCH_SIZE` 降到 `8` |
| 16GB RAM 被 OOM Kill | 已內建 `_cleanup_trainer()` 防護，不應該發生 |
| Docker GPU 找不到 | 確認 `nvidia-container-toolkit` 已安裝 |
