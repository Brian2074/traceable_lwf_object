# traceable_lwf_object
This repo utilize the official code of TagFed paper to implement federated continual learning. See the original [repo](https://github.com/P0werWeirdo/TagFCL) for more details.

## Features

- Migrated to flower to perform federated learning on multiple computers as clients.

## Quick Start

### Running with Docker

**1. Local Testing (All-in-One)**
```bash
docker-compose -f docker/docker-compose.yml up --build
```

**2. Distributed Deployment (Production)**
*   **Server**: `docker-compose -f docker/server-compose.yml up -d`
*   **Client**:
    ```bash
    SERVER_ADDRESS=<SERVER_IP>:8080 CLIENT_ID=0 docker-compose -f docker/client-compose.yml up -d
    ```

### Detailed Documentation

For full details on configuration, adding new clients, and multi-machine setup, please refer to the **[Flower Usage Guide](docs/flower_usage.md)**.