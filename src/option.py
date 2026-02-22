"""Command-line argument parser for YOLOv8n + FedRep + Feature KD."""

import argparse
import torch


def args_parser() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # ── Federated Learning ──────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="cifar100", help="name of dataset"
    )
    parser.add_argument(
        "--numclass",
        type=int,
        default=10,
        help="number of dataset classes each task",
    )
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument(
        "--num_clients", type=int, default=2, help="initial number of clients"
    )
    parser.add_argument("--img_size", type=int, default=640, help="input image size")
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./save_folder",
        help="folder name for saving checkpoints",
    )
    parser.add_argument(
        "--epochs_local",
        type=int,
        default=20,
        help="local epochs of each global round",
    )
    parser.add_argument(
        "--epochs_server",
        type=int,
        default=10,
        help="server epochs of each global round",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./log/result",
        help="the path of result log file",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="unnamed",
        help="experiment name for identification in logs",
    )

    parser.add_argument("--device", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument(
        "--epochs_global",
        type=int,
        default=40,
        help="total number of global rounds",
    )
    parser.add_argument(
        "--tasks_global", type=int, default=10, help="total number of tasks"
    )
    parser.add_argument(
        "--tasks_epoch", type=int, default=4, help="each task epoch count"
    )

    parser.add_argument(
        "--temperature",
        type=int,
        default=15,
        help="temperature of distillation",
    )
    parser.add_argument(
        "--a_c", type=float, default=0.9, help="alpha of client loss weighting"
    )
    parser.add_argument(
        "--a_s", type=float, default=0.9, help="alpha of server loss weighting"
    )

    # ── Batch / Training ────────────────────────────────────────────────
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument(
        "--lr_local", type=float, default=0.01, help="local learning rate"
    )
    parser.add_argument(
        "--lr_server", type=float, default=0.2, help="server learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="use CUDA"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="use Automatic Mixed Precision (FP16) training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="number of steps to accumulate gradients",
    )

    # ── YOLOv8 / Model ─────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help="model name (yolov8n, yolov8s, etc.)",
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="weights/yolov8n.pt",
        help="path to pretrained YOLO weights",
    )

    # ── FedRep ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--fedrep_body_epochs",
        type=int,
        default=5,
        help="epochs for training global body (FedRep)",
    )
    parser.add_argument(
        "--fedrep_head_epochs",
        type=int,
        default=5,
        help="epochs for training local head (FedRep)",
    )

    # ── Feature-based Knowledge Distillation ────────────────────────────
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=0.5,
        help="weight for feature KD loss",
    )
    parser.add_argument(
        "--kd_layer",
        type=int,
        default=21,
        help="layer index for feature extraction in KD",
    )

    # ── Flower ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--client_id", type=int, default=0, help="client ID for partitioning"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address",
    )

    args = parser.parse_args()
    return args