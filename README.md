# Federated Continual Object Detection (YOLOv8 + FedRep)

Federated Continual Learning for Object Detection using YOLOv8, Federated Representation Learning (FedRep), and Feature-based Knowledge Distillation.

## Project Structure

```
├── src/                 # Core FL training code
│   ├── main.py          # Standalone local simulation
│   ├── client.py        # FedRep client wrapper
│   ├── manager.py       # Training manager (Body/Head/KD)
│   ├── option.py        # CLI arguments
│   ├── start_server.py  # Flower server launcher
│   ├── start_client.py  # Flower client launcher
│   ├── flower_strategy.py
│   ├── models/          # YOLOv8 wrapper, Feature KD
│   └── utils/
├── tools/               # Dataset & utility scripts
│   ├── hf2yolo.py       # HuggingFace → YOLO converter
│   ├── split_non_iid.py # Dirichlet Non-IID data splitter
│   ├── coco2yolo.py     # COCO → YOLO converter
│   └── check_env.py     # Environment checker
├── weights/             # Pretrained model weights
├── datasets/            # YOLO-format datasets
├── docker/              # Dockerfiles & compose files
├── tests/               # Unit tests
├── docs/                # Documentation
└── DEPLOY.md            # Multi-machine deployment guide
```

## Quick Start

### 1. Prepare Dataset

```bash
python tools/hf2yolo.py --repo ARG-NCTU/SPSCD_coco --output_dir datasets/spscd_coco_yolo
python tools/split_non_iid.py --data_dir datasets/spscd_coco_yolo --clients 3 --alpha 0.5
```

### 2. Local Standalone Training

```bash
python src/main.py --model weights/yolov8n.pt --num_clients 2 --tasks_epoch 3
```

### 3. Distributed FL Training (Docker)

See [DEPLOY.md](DEPLOY.md) for full multi-machine setup instructions.

```bash
# All-in-one local test
docker compose -f docker/docker-compose.yml up --build
```