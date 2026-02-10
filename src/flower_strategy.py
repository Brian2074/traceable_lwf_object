
import flwr as flw
from flwr.common import FitRes, Parameters, Scalar, FitIns
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
import copy as cp

from utils import server_utils
import models

class TagFCLStrategy(flw.server.strategy.FedAvg):
    def __init__(self, args, initial_parameters: Parameters = None):
        super().__init__(initial_parameters=initial_parameters)
        self.args = args
        self.server_model = models.ServerModel([], {}, args)
        if args.cuda:
            self.server_model.cuda(args.device)
        
        # Dictionary to store messages (logits) to be sent back to specific clients in the next round
        # Key: client_cid (string), Value: message dict
        self.messages_to_clients = {}
        
        # Keep track of task definitions if dynamic
        for i in range(1, args.tasks_global + 1):
            self.server_model.add_dataset(i, args.numclass)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        """Aggregate fit results using TagFCL logic."""
        
        if not results:
            return None, {}

        # 1. Collect messages (Features) from all clients
        messages_to_server = []
        client_map = {} # Map internal client_id (from TagFCL) to Flower cid

        print(f"Server Round {server_round}: Received {len(results)} results")

        for client_proxy, fit_res in results:
            # Unpickle the message sent by client via metrics
            if "message" in fit_res.metrics:
                # We expect the message to be a hex string or bytes
                # Flower metrics only support int/float/str/bytes (depending on version)
                # It is safest to assume it comes as bytes or we coded it as hex string
                msg_bytes = fit_res.metrics["message"]
                if isinstance(msg_bytes, str):
                    # If it was sent as hex string
                    msg_bytes = bytes.fromhex(msg_bytes)
                
                msg = pickle.loads(msg_bytes)
                messages_to_server.append(msg)
                
                # Map TagFCL client_id to Flower cid
                # We need this to know which Flower client corresponds to which TagFCL message
                tag_client_id = msg['client_id']
                client_map[tag_client_id] = client_proxy.cid
            else:
                print(f"Warning: No message found in metrics from client {client_proxy.cid}")

        if not messages_to_server:
            return None, {}

        # 2. Train Server Model (TagFCL Aggregation)
        # This function updates self.server_model in-place and returns responses
        # Note: train_and_get_message expects a list of dicts
        print(f"Training Server Model with gathered features...")
        response_messages = server_utils.train_and_get_message(
            self.server_model, 
            messages_to_server, 
            self.args
        )

        # 3. Store response messages to be sent in configure_fit of NEXT round
        # Wait, standard Flower loop calls configure_fit -> fit -> aggregate_fit
        # The result of aggregate_fit is usually parameters for the *next* round's configure_fit.
        # But we want to send specific data.
        
        self.messages_to_clients.clear()
        for msg in response_messages:
            tag_client_id = msg['client_id']
            if tag_client_id in client_map:
                cid = client_map[tag_client_id]
                self.messages_to_clients[cid] = msg
            else:
                print(f"Error: Could not map TagFCL client {tag_client_id} to Flower CID")

        # We return the server model parameters (optional, mostly for checkpointing)
        # We don't necessarily need to broadcast them if we use the logits-only strategy
        # But keeping it standard doesn't hurt.
        # parameters_aggregated = flw.common.ndarrays_to_parameters(
        #     [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
        # )
        
        # For efficiency, we can return None or empty parameters if we don't assume client needs generic weights
        # But let's return None to save bandwidth, assuming we rely on config
        return None, {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: flw.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        """Configure the next round of training."""
        
        # Standard configuration (sample clients)
        config = {}
        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.args.num_clients, 
            min_num_clients=self.args.num_clients
        )

        # Custom configuration per client
        client_instructions = []
        
        # Calculate current Task ID based on round
        # Task ID starts from 1
        # Round 1..N -> Task 1, Round N+1..2N -> Task 2
        # Note: server_round comes from Flower server (starts at 1)
        current_task_id = ((server_round - 1) // self.args.tasks_epoch) + 1
        
        # Ensure we don't exceed total tasks
        if current_task_id > self.args.tasks_global:
            current_task_id = self.args.tasks_global

        print(f"Server Round {server_round} -> Task {current_task_id}")
        
        for client in clients:
            cid = client.cid
            client_config = config.copy()
            client_config["task_id"] = current_task_id
            
            # Check if we have a pending message (logits) for this client
            if cid in self.messages_to_clients:
                msg = self.messages_to_clients[cid]
                # Pickle and encode to hex string for transport
                msg_bytes = pickle.dumps(msg)
                client_config["server_response"] = msg_bytes.hex()
                # Clear the message after sending? 
                # Ideally yes, but if client fails, we might lose it. For now, keep it simple.
            
            # Add Task ID or other global args if needed
            # We assume synchoronous task progression handled by args or external coordinator
            # For now, just passing the server response is the key
            
            fit_ins_custom = FitIns(parameters, client_config)
            client_instructions.append((client, fit_ins_custom))

        return client_instructions
