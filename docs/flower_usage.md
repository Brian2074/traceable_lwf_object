# Flower Framework Usage Guide

This repository integrates the **Flower (flwr)** framework to implement Federated Continual Learning (FCL) based on the **TagFed** method. Unlike standard Federated Averaging (FedAvg) which aggregates model weights, this implementation aggregates **Features and Logits** to enable Traceable Federated Learning.

## Key Files

*   **`src/flower_strategy.py`**
    *   **Role**: Defines the Server-side aggregation strategy (`TagFCLStrategy`).
    *   **Mechanism**: Inherits from `flwr.server.strategy.FedAvg`. It collects features (messages) from clients, trains the server-side model, and sends back logits (server response) for distillation. It **does not** average client weights.
*   **`src/start_server.py`**
    *   **Role**: Entry point to start the Flower Server.
    *   **Logic**: Initializes the strategy and starts the gRPC server.
*   **`src/start_client.py`**
    *   **Role**: Entry point to start a Flower Client.
    *   **Logic**: 
        1.  **Head Training**: Trains feature extractor.
        2.  **Generate Message**: Extracts features and sends to Server.
        3.  **Receive Response**: Gets logits from Server.
        4.  **Tail Training**: Distills knowledge using Server logits.

---

## Prerequisites & Configuration

Before running the system, ensure you have the following configurations ready:

### 1. Environment
Make sure `flower` is installed:
```bash
pip install flwr
```

### 2. Dataset Preparation
The system supports `cifar100`, `tiny_imagenet`, and `mini_imagenet`.
*   **CIFAR-100**: Automatically downloaded.
*   **Tiny-ImageNet / Mini-ImageNet**: Must be manually downloaded and placed in the root directory (or as specified in `src/option.py` defaults).

### 3. Network Configuration
*   **Localhost**: If running everything on one machine, use `127.0.0.1:8080`.
*   **Remote Server**: If Server and Clients are on different machines, ensure the Server's IP is accessible and port `8080` is open.

---

## Running with Docker (Recommended)

To simplify environment setup, you can use Docker Compose.

### 1. Build and Run
```bash
docker-compose -f docker/docker-compose.yml up --build
```
This command will:
*   Build the Docker image using `docker/Dockerfile`.
*   Start the Server container.
*   Start 2 Client containers automatically.

### 2. Scale Clients
To run more clients (e.g., 5 clients):
```bash
docker-compose -f docker/docker-compose.yml up --scale client1=1 --scale client2=4
```
*Note: You may need to duplicate client services in `docker-compose.yml` to assign unique `client_id`s properly, as the current compose file hardcodes `client_id` in the command.*

---

## Running Manually

You need to run the Server and Clients in separate terminals.

### 1. Start the Server

Open a terminal and run:

```bash
cd src
python start_server.py --num_clients <MIN_CLIENTS> --epochs_global <TOTAL_ROUNDS>
```

**Parameters**:
*   `--num_clients`: The minimum number of clients required to start a round (default: 2).
*   `--epochs_global`: Total number of communication rounds (default: 40).
*   `--dataset`: Dataset to use (cifar100, tiny_imagenet, etc).

**Example**:
```bash
python start_server.py --num_clients 2 --dataset cifar100
```

### 2. Start Clients

Each client must have a unique `client_id`.

**Client 1**:
```bash
cd src
python start_client.py --client_id 0 --dataset cifar100 --device 0
```

**Client 2**:
```bash
cd src
python start_client.py --client_id 1 --dataset cifar100 --device 0
```

**Parameters**:
*   `--client_id`: **CRITICAL**. Unique ID (0, 1, 2...). Used to distinguish data splits and logs.
*   `--dataset`: Must match the server's dataset.
*   `--device`: GPU ID (e.g., 0, 1) or -1 for CPU.
*   `--server_address`: (Optional) IP:Port of the server (Default: `127.0.0.1:8080`).

---

## Adding New Clients

To scale up the system (e.g., add a 3rd client), follow these steps:

### 1. Server Configuration
Update the `--num_clients` argument when starting the server to reflect the new total.
```bash
python start_server.py --num_clients 3
```

### 2. Client Execution
Launch a new terminal for the new client with a unique `client_id` (e.g., `2`).
```bash
python start_client.py --client_id 2 ...
```

### 3. Data Partitioning (Implicit)
This repository uses `client_id` to partition data implicitly or log results.
*   **Verification**: Ensure your `num_users` or data split logic in `src/option.py` uses consistent seeds across all clients.

**Best Practice**: Always pass the same `--num_clients` and `--seed` to all instances (Server and Clients) to ensure consistent data splitting if the split happens locally.

```bash
# Server
python start_server.py --num_clients 10 --seed 2024

# Client 0
python start_client.py --client_id 0 --num_clients 10 --seed 2024
```
