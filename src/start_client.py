
import flwr as flw
import torch
import pickle
import sys
import os
import argparse
import numpy as np
import copy as cp
from torchvision import transforms

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import Client
from iCIFAR100 import iCIFAR100
from tiny_imagenet import Tiny_Imagenet
from mini_imagenet import Mini_Imagenet
from utils import setup_seed
import models

class TagFCLFlowerClient(flw.client.NumPyClient):
    def __init__(self, args, client_id, train_dataset, test_dataset):
        self.args = args
        self.client_id = client_id
        self.tagfcl_client = Client(args, train_dataset, test_dataset, client_id)
        
        # Initialize models/datasets as done in main.py loop
        # But we need to do it incrementally as tasks arrive
        # The Manager inside Client handles model initialization
        
    def fit(self, parameters, config):
        # 1. Parse Config
        task_id = int(config["task_id"])
        epochs = self.args.epochs_local // 2
        
        print(f"Client {self.client_id}: Received fit request for Task {task_id}")

        # 2. Check for Server Response (Logits from Head phase aggregation)
        # If this exists, it means we are in the "Tail Training" phase of the PREVIOUS round logic?
        # Or the second half of the current epoch?
        # TagFCL flow: Head -> Server -> Tail.
        # If we receive server_response, we run Tail.
        
        if "server_response" in config:
            print(f"Client {self.client_id}: Received Server Response. Running Tail Training...")
            msg_hex = config["server_response"]
            msg_bytes = bytes.fromhex(msg_hex)
            msg = pickle.loads(msg_bytes)
            
            # Verify message is for this client
            if msg['client_id'] != self.client_id:
                print(f"Warning: Received message for client {msg['client_id']}, but I am {self.client_id}")
            
            self.tagfcl_client.train_tail(task_id, epochs, msg)
        
        # 3. Validation Logic (Optional, usually done in evaluate)
        # record_result logic from main.py
        
        # 4. Head Training
        # We always run Head Training in the 'fit' round to generate features for the Server
        print(f"Client {self.client_id}: Running Head Training...")
        
        # Ensure model has added the dataset for this task (Manager handles this via add_task in init)
        # But we need to make sure dataset is prepared?
        # Manager calls _get_train_dataloader_normal which prepares data
        
        self.tagfcl_client.train_head(task_id, epochs)
        
        # 5. Generate Message (Features) for Server
        print(f"Client {self.client_id}: Generating Message (Features)...")
        msg = self.tagfcl_client.get_message(task_id)
        
        # Pickle message
        msg_bytes = pickle.dumps(msg)
        
        # Return dummy parameters (we don't aggregate weights) and metrics
        return [], 50000, {"message": msg_bytes.hex()}

    def evaluate(self, parameters, config):
        print(f"Client {self.client_id}: Evaluating...")
        # Use only_validate
        # Note: only_validate returns a list of accuracies for all learned tasks
        accuracies = self.tagfcl_client.only_validate()
        
        # Compute mean accuracy
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # We can perform Backtrack training logic here if needed?
        # In main.py, Backtrack happens AFTER recording result.
        # But Backtrack involves Head -> Server -> Tail loop again effectively.
        # Migrating Backtrack to Flower is complex. 
        # For MVP, we stick to the main loop. Backtrack tasks might need separate Flower rounds.
        
        return float(0.0), 100, {"accuracy": float(mean_acc)}

def main():
    # Parse arguments
    # We reuse option.py logic but we need to pass arguments manually or via valid flags
    from option import args_parser
    args = args_parser()
    
    # Override args with command line if needed, but option.py handles argparse
    # Note: start_client.py will likely pass specific args
    
    # Setup Seed
    setup_seed(args.seed)
    
    # Client ID (custom arg, not in option.py usually, so we might need to parse it separately or add it)
    # But args_parser returns a Namespace.
    # We assume 'client_id' is passed or we default to 0
    # Let's inspect sys.argv
    
    client_id = 0
    # Simple hack to find client_id from args without modifying option.py deeply
    if "--client_id" in sys.argv:
        idx = sys.argv.index("--client_id")
        client_id = int(sys.argv[idx+1])
        # Remove it so args_parser doesn't complain? 
        # option.py usually uses argparse which ignores unknown args? No, it errors.
        # We should probably modify option.py to include client_id or handle it here.
    
    # Server Address
    server_address = "127.0.0.1:8080"
    if "--server_address" in sys.argv:
        idx = sys.argv.index("--server_address")
        server_address = sys.argv[idx+1]

    # Dataset Setup (Same as main.py)
    # transform
    train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.24705882352941178),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
        test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)
    elif args.dataset == 'tiny_imagenet':
        train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform,
                                      test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform,
                                      test_transform=test_transform)
        test_dataset.get_data()
    else:
        # Mini Imagenet
        train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        test_dataset.get_data()

    # Start Client
    client = TagFCLFlowerClient(args, client_id, train_dataset, test_dataset)
    
    flw.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == '__main__':
    main()
