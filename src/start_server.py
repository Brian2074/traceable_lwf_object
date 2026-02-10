
import flwr as flw
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flower_strategy import TagFCLStrategy
from option import args_parser
from utils import setup_seed

def main():
    # Parse arguments
    args = args_parser()
    setup_seed(args.seed)

    # Initialize Strategy
    strategy = TagFCLStrategy(args=args)

    # Start Flower Server
    # Total rounds = epochs_global from original logic
    # Each round corresponds to one epoch of coupled Head-Tail training
    
    print(f"Starting Flower Server for {args.epochs_global} rounds...")
    print(f"Min Clients: {args.num_clients}")
    
    server_address = "0.0.0.0:8080"
    
    flw.server.start_server(
        server_address=server_address,
        config=flw.server.ServerConfig(num_rounds=args.epochs_global),
        strategy=strategy,
    )

if __name__ == '__main__':
    main()
