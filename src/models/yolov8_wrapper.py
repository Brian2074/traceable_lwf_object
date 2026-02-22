"""YOLOv8 wrapper with FedRep body/head split and dynamic head expansion.

This module wraps the Ultralytics YOLOv8-nano model and provides:
- Body (layers 0-21) / Head (Detect, layer 22) split for FedRep.
- Freeze / unfreeze utilities for alternating training.
- Dynamic expansion of the classification branch (cv3) for new classes.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOv8Wrapper(nn.Module):
    """Wrapper around YOLOv8n for FedRep-based federated continual learning.

    Attributes:
        yolo: The underlying Ultralytics YOLO object.
        inner_model: The raw ``nn.Module`` inside ``yolo.model``.
        body: ``nn.Sequential`` containing backbone + neck layers (0-21).
        head: The ``Detect`` module at layer 22.
        num_classes: Current number of detection classes.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, weights: str = "weights/yolov8n.pt") -> None:
        """Load a YOLOv8n model and split it into body and head.

        Args:
            weights: Path to pretrained YOLO weights or model name.
        """
        super().__init__()
        self.yolo = YOLO(weights)
        self.inner_model: nn.Module = self.yolo.model.model  # nn.Sequential

        # Validate expected layer count (YOLOv8n has 23 layers: 0-22)
        num_layers = len(self.inner_model)
        if num_layers < 23:
            raise ValueError(
                f"Expected ≥ 23 layers in YOLOv8n, got {num_layers}. "
                "Check the YOLO model architecture."
            )

        # Split: body = layers 0..21, head = layer 22 (Detect)
        self.body = nn.Sequential(*list(self.inner_model.children())[:22])
        self.head = self.inner_model[-1]  # Detect module
        self.num_classes: int = self.head.nc

        logger.info(
            "YOLOv8Wrapper initialised: %d body layers, head nc=%d",
            len(self.body),
            self.num_classes,
        )

    # ------------------------------------------------------------------ #
    #  FedRep freeze / unfreeze
    # ------------------------------------------------------------------ #
    def freeze_body(self) -> None:
        """Freeze all parameters in the global body (layers 0-21)."""
        for param in self.body.parameters():
            param.requires_grad = False

    def unfreeze_body(self) -> None:
        """Unfreeze all parameters in the global body (layers 0-21)."""
        for param in self.body.parameters():
            param.requires_grad = True

    def freeze_head(self) -> None:
        """Freeze all parameters in the local head (Detect module)."""
        for param in self.head.parameters():
            param.requires_grad = False

    def unfreeze_head(self) -> None:
        """Unfreeze all parameters in the local head (Detect module)."""
        for param in self.head.parameters():
            param.requires_grad = True

    def get_body_params(self) -> List[nn.Parameter]:
        """Return a list of all body parameters (for optimizer).

        Returns:
            List of ``nn.Parameter`` from the body sub-module.
        """
        return [p for p in self.body.parameters() if p.requires_grad]

    def get_head_params(self) -> List[nn.Parameter]:
        """Return a list of all head parameters (for optimizer).

        Returns:
            List of ``nn.Parameter`` from the head sub-module.
        """
        return [p for p in self.head.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> "YOLOv8Wrapper":
        """Set training mode, propagating to the inner YOLO model.

        The inner ``DetectionModel`` is **not** a registered submodule,
        so ``nn.Module.train()`` does not reach it.  We propagate
        manually to ensure ``DetectionModel.forward()`` dispatches to
        ``_predict_once()`` (eval) vs ``loss()`` (train) correctly.

        Args:
            mode: ``True`` for training, ``False`` for eval.

        Returns:
            Self.
        """
        super().train(mode)
        self.yolo.model.train(mode)
        return self

    def eval(self) -> "YOLOv8Wrapper":
        """Set eval mode, propagating to the inner YOLO model."""
        return self.train(False)

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through body then head.

        Note:
            YOLOv8's ``Detect`` head expects a *list* of multi-scale
            feature maps.  The body's intermediate layers produce these
            via skip connections.  For a proper forward we delegate to
            the raw ``nn.Module`` so that the skip/concat topology is
            preserved — without triggering the YOLO trainer pipeline.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Detection output tensor.
        """
        # Call .forward() explicitly — Ultralytics DetectionModel overrides
        # __call__ and may trigger training pipeline via nn.Module.__call__.
        return self.yolo.model.forward(x)

    # ------------------------------------------------------------------ #
    #  Dynamic head expansion
    # ------------------------------------------------------------------ #
    def expand_detect_head(self, new_num_classes: int) -> None:
        """Expand the Detect head's classification branch for more classes.

        For each detection scale, the last ``Conv2d`` in ``cv3[i]`` is
        replaced with a wider one.  Old class weights are copied; new
        channels are Kaiming-initialised.

        Args:
            new_num_classes: The new (larger) number of classes.

        Raises:
            ValueError: If *new_num_classes* ≤ current number of classes.
        """
        old_nc = self.num_classes
        if new_num_classes <= old_nc:
            raise ValueError(
                f"new_num_classes ({new_num_classes}) must be > "
                f"current nc ({old_nc})"
            )

        detect = self.head  # Detect module

        for scale_idx, cv3_branch in enumerate(detect.cv3):
            # cv3_branch is an nn.Sequential; last child is Conv2d (or
            # an ``nn.Sequential(Conv2d, ...)``).  We need the final
            # Conv2d whose out_channels == old_nc.
            last_conv = self._find_last_conv2d(cv3_branch)
            if last_conv is None:
                raise RuntimeError(
                    f"Could not locate final Conv2d in cv3[{scale_idx}]"
                )

            old_out = last_conv.out_channels
            channels_per_class = old_out // old_nc
            new_out = new_num_classes * channels_per_class

            # Build replacement conv
            new_conv = nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=new_out,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride,
                padding=last_conv.padding,
                bias=last_conv.bias is not None,
            )

            # Copy old weights
            with torch.no_grad():
                new_conv.weight[:old_out].copy_(last_conv.weight)
                if last_conv.bias is not None:
                    new_conv.bias[:old_out].copy_(last_conv.bias)

                # Kaiming init for new channels
                nn.init.kaiming_normal_(
                    new_conv.weight[old_out:], mode="fan_out", nonlinearity="relu"
                )
                if last_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias[old_out:])

            # Move to same device
            new_conv = new_conv.to(last_conv.weight.device)

            # Replace in-place
            self._replace_last_conv2d(cv3_branch, new_conv)

            logger.info(
                "cv3[%d]: expanded %d → %d out_channels", scale_idx, old_out, new_out
            )

        # Update Detect meta-attributes
        detect.nc = new_num_classes
        detect.no = new_num_classes + detect.reg_max * 4
        self.num_classes = new_num_classes

        logger.info("Head expanded: nc=%d, no=%d", detect.nc, detect.no)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_last_conv2d(sequential: nn.Module) -> Optional[nn.Conv2d]:
        """Walk a sequential and return the last ``nn.Conv2d``."""
        last: Optional[nn.Conv2d] = None
        for module in sequential.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        return last

    @staticmethod
    def _replace_last_conv2d(sequential: nn.Module, new_conv: nn.Conv2d) -> None:
        """Replace the last ``nn.Conv2d`` found inside *sequential*."""
        # Walk children at each level; replace when found.
        children = list(sequential.named_children())
        for i in range(len(children) - 1, -1, -1):
            name, child = children[i]
            if isinstance(child, nn.Conv2d):
                setattr(sequential, name, new_conv)
                return
            # Recurse into sub-Sequential / modules
            grandchildren = list(child.named_children())
            for j in range(len(grandchildren) - 1, -1, -1):
                gname, gchild = grandchildren[j]
                if isinstance(gchild, nn.Conv2d):
                    setattr(child, gname, new_conv)
                    return

    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #
    def snapshot_teacher_state(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the inner model's state dict.

        Snapshots ``yolo.model`` (DetectionModel) rather than the
        outer wrapper, so keys align when used by ``FeatureKDLoss``.

        Returns:
            A dict mapping parameter names to detached CPU tensors.
        """
        return {k: v.cpu().clone() for k, v in self.yolo.model.state_dict().items()}
