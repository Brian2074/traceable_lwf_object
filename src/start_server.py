"""Flower server launcher for FedRep YOLOv8n."""

import os
import sys

import flwr as flw

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flower_strategy import FedRepStrategy
from option import args_parser
from utils import setup_seed


def main() -> None:
    """Start the Flower server with FedRep strategy."""
    args = args_parser()
    setup_seed(args.seed)

    strategy = FedRepStrategy(args=args)

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Model: {args.model}")
    print(f"  FedRep: body_ep={args.fedrep_body_epochs}, head_ep={args.fedrep_head_epochs}")
    print(f"  KD: alpha={args.kd_alpha}, layer={args.kd_layer}")
    print(f"  Rounds: {args.epochs_global}, Clients: {args.num_clients}")
    print(f"{'=' * 60}\n")

    server_address = "0.0.0.0:8080"
    print(f"Starting Flower Server at {server_address}...")

    flw.server.start_server(
        server_address=server_address,
        config=flw.server.ServerConfig(num_rounds=args.epochs_global),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
