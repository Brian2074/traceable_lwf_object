"""Federated learning client for YOLOv8n + FedRep + Feature KD."""

import logging
from typing import Any, Dict, List, Optional

from manager import Manager
from utils import set_logger, remove_logger

logger = logging.getLogger(__name__)


class Client:
    """Client wrapper that orchestrates FedRep training phases.

    Each client owns a local ``Manager`` instance with a YOLOv8
    model whose body is shared (aggregated globally) and whose
    head is private (trained locally).

    Attributes:
        client_id: Unique identifier.
        manager: The training manager.
        max_task: Highest task ID seen so far.
    """

    def __init__(
        self,
        args: Any,
        train_dataset: Any,
        test_dataset: Any,
        client_id: int,
    ) -> None:
        """Initialise the client.

        Args:
            args: Parsed argument namespace.
            train_dataset: Training dataset.
            test_dataset: Test dataset.
            client_id: Client identifier.
        """
        self.client_id = client_id
        self.args = args
        self.max_task: int = 0
        self.log_path = f"log/client{client_id}"

        self.manager = Manager(args, train_dataset, test_dataset, self.log_path, client_id=client_id)

    # ------------------------------------------------------------------ #
    #  FedRep training phases
    # ------------------------------------------------------------------ #
    def train_body(self, task_id: int, yaml_path: str) -> None:
        """Train the global body (shared across clients).

        Args:
            task_id: Current task ID.
            yaml_path: Path to YOLO dataset yaml config.
        """
        set_logger(self.log_path)
        self.manager.train_body(
            task_id, self.args.fedrep_body_epochs, yaml_path
        )
        remove_logger()

    def train_head(self, task_id: int, yaml_path: str) -> None:
        """Train the local head (private to this client).

        Args:
            task_id: Current task ID.
            yaml_path: Path to YOLO dataset yaml config.
        """
        set_logger(self.log_path)
        self.manager.train_head(
            task_id, self.args.fedrep_head_epochs, yaml_path
        )
        remove_logger()

    def train_with_kd(self, task_id: int, yaml_path: str) -> None:
        """Train with feature-based knowledge distillation.

        Args:
            task_id: Current task ID.
            yaml_path: Path to YOLO dataset yaml config.
        """
        set_logger(self.log_path)
        self.manager.train_with_kd(
            task_id, self.args.fedrep_body_epochs, yaml_path
        )
        remove_logger()

    # ------------------------------------------------------------------ #
    #  Task lifecycle
    # ------------------------------------------------------------------ #
    def prepare_new_task(self, task_id: int, new_num_classes: int) -> None:
        """Prepare the client for a new task.

        Args:
            task_id: The new task ID.
            new_num_classes: Total classes after this task.
        """
        self.max_task = max(self.max_task, task_id)
        self.manager.prepare_new_task(task_id, new_num_classes)

    # ------------------------------------------------------------------ #
    #  FL body exchange
    # ------------------------------------------------------------------ #
    def get_body_state_dict(self) -> Dict[str, Any]:
        """Return body parameters for server aggregation.

        Returns:
            Body state dict.
        """
        return self.manager.get_body_state_dict()

    def load_body_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load aggregated body parameters from server.

        Args:
            state_dict: Aggregated body state dict.
        """
        self.manager.load_body_state_dict(state_dict)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def validate(self, val_loader: Any, task_id: int) -> float:
        """Run validation and return metric.

        Args:
            val_loader: Validation data loader.
            task_id: Task ID.

        Returns:
            Validation metric value.
        """
        set_logger(self.log_path)
        result = self.manager.validate(val_loader, 0, task_id)
        remove_logger()
        return result