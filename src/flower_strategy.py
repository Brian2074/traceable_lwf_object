"""Flower server strategy for FedRep body-weight aggregation.

Replaces TagFCL's logit-based aggregation with standard FedAvg
on the YOLOv8 body parameters only. Head parameters remain local.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as flw
import numpy as np
import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)


class FedRepStrategy(flw.server.strategy.FedAvg):
    """FedAvg strategy operating only on YOLOv8 body parameters.

    Clients send body weights; the server averages them and sends
    the aggregated body back.  Head parameters are never exchanged.

    Attributes:
        args: Parsed arguments.
    """

    def __init__(
        self,
        args: object,
        initial_parameters: Optional[Parameters] = None,
        **kwargs,
    ) -> None:
        """Initialise the strategy.

        Args:
            args: Parsed arguments.
            initial_parameters: Optional initial model parameters.
            **kwargs: Passed to FedAvg.
        """
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.args = args

        logger.info("FedRepStrategy initialised (body-only FedAvg).")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: flw.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of client training.

        Injects task metadata into the config so clients know which
        task to train on.

        Args:
            server_round: Current server round.
            parameters: Current global parameters.
            client_manager: Server's client manager.

        Returns:
            List of (client, FitIns) pairs.
        """
        # Calculate current task ID from round number
        current_task_id = ((server_round - 1) // self.args.tasks_epoch) + 1
        current_task_id = min(current_task_id, self.args.tasks_global)

        num_classes = current_task_id * self.args.numclass

        config = {
            "task_id": str(current_task_id),
            "num_classes": str(num_classes),
        }

        clients = client_manager.sample(
            num_clients=self.args.num_clients,
            min_num_clients=self.args.num_clients,
        )

        logger.info(
            "Round %d → Task %d (%d classes), %d clients",
            server_round,
            current_task_id,
            num_classes,
            len(clients),
        )

        return [
            (client, FitIns(parameters, config))
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate body parameters from clients using FedAvg.

        Args:
            server_round: Current server round.
            results: List of client results.
            failures: List of failures.

        Returns:
            (aggregated_parameters, metrics) tuple.
        """
        if not results:
            return None, {}

        if failures:
            logger.warning(
                "Round %d: %d failures out of %d",
                server_round,
                len(failures),
                len(results) + len(failures),
            )

        # Standard FedAvg aggregation (inherited from parent)
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        logger.info(
            "Round %d: aggregated %d client results.",
            server_round,
            len(results),
        )

        return aggregated_params, metrics
