"""Main entry point for standalone (non-Flower) YOLOv8n + FedRep training."""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from client import Client
from option import args_parser
from utils.Fed_utils import setup_seed

logger = logging.getLogger(__name__)


def record_result(
    acc: list,
    epoch: int,
    args: object,
) -> None:
    """Append accuracy results to the log file.

    Args:
        acc: List of accuracy values per client.
        epoch: Current global epoch.
        args: Parsed arguments.
    """
    with open(args.log_path, "a") as f:
        f.write(f"Epoch_{epoch}:\n")
        arr = np.array(acc)
        mean_acc = np.mean(arr)
        f.write(f"  Mean Accuracy: {mean_acc * 100:.2f}\n")
        for i, a in enumerate(acc):
            f.write(f"  Client_{i}: {a * 100:.2f}\n")
        f.write("\n")
        f.flush()


def main() -> None:
    """Run the standalone FedRep + Feature KD training loop."""
    args = args_parser()
    setup_seed(args.seed)

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Model: {args.model}")
    print(f"  lr={args.lr_local}, bs={args.batch_size}")
    print(f"  FedRep: body_ep={args.fedrep_body_epochs}, head_ep={args.fedrep_head_epochs}")
    print(f"  KD: alpha={args.kd_alpha}, layer={args.kd_layer}")
    print(f"  AMP: {args.use_amp}")
    print(f"{'=' * 60}\n")

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # NOTE: Dataset and DataLoader creation is a placeholder.
    # YOLOv8 requires detection-format datasets (COCO/VOC).
    # Replace the following with actual dataset setup.
    train_dataset = None
    test_dataset = None

    # Create clients
    clients = [
        Client(args, train_dataset, test_dataset, i) for i in range(args.num_clients)
    ]

    # Get client-specific yaml paths for this standalone run
    # We assume download_dataset.py generated datasets/spscd_coco_yolo/client_{id}.yaml
    yaml_paths = [
        f"datasets/spscd_coco_yolo/client_{i}.yaml" for i in range(args.num_clients)
    ]

    num_classes_so_far = 0

    for task_id in range(1, args.tasks_global + 1):
        num_classes_so_far += args.numclass
        print(f"\n--- Task {task_id}: {num_classes_so_far} total classes ---")

        # Prepare all clients for new task
        for client in clients:
            client.prepare_new_task(task_id, num_classes_so_far)

        # Get client-specific yaml paths for this standalone run
        # We assume download_dataset.py generated datasets/spscd/client_{id}.yaml
        yaml_paths = []
        for c in clients:
            yaml_path = f"datasets/spscd_coco_yolo/client_{c.client_id}.yaml"
            if not os.path.exists(yaml_path):
                logger.error(f"Dataset config not found for client {c.client_id}: {yaml_path}")
                yaml_path = "datasets/spscd_coco_yolo/data.yaml"
            yaml_paths.append(yaml_path)

        for ep in range(args.tasks_epoch):
            print(f"\n  Global Round {ep + 1}/{args.tasks_epoch} for Task {task_id}")

            # 1. Train body (global) on each client
            for c_idx, client in enumerate(clients):
                client.train_body(task_id, yaml_paths[c_idx])

            # 2. Aggregate body parameters (FedAvg)
            body_dicts = [c.get_body_state_dict() for c in clients]
            avg_body = _fedavg_state_dicts(body_dicts)

            # 3. Distribute aggregated body
            for client in clients:
                client.load_body_state_dict(avg_body)

            # 4. Train head (local) on each client
            for c_idx, client in enumerate(clients):
                client.train_head(task_id, yaml_paths[c_idx])

            # 5. Optional: KD training (after task 1)
            if args.tasks_global > 1 and args.kd_alpha > 0:
                for c_idx, client in enumerate(clients):
                    # Teacher model needs to be created inside manager before kd
                    client.train_with_kd(task_id, f"datasets/spscd_coco_yolo/client_{c_idx}.yaml")

            # 6. Validate
            acc = []
            for c_idx, client in enumerate(clients):
                acc.append(client.validate(yaml_paths[c_idx], task_id))
            record_result(acc, ep + 1, args)


def _fedavg_state_dicts(
    state_dicts: list,
) -> dict:
    """Average multiple state dicts (FedAvg).

    Args:
        state_dicts: List of state dicts from clients.

    Returns:
        Averaged state dict.
    """
    if not state_dicts:
        return {}

    avg = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = stacked.mean(dim=0)
    return avg


if __name__ == "__main__":
    main()