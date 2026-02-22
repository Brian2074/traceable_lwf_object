"""Feature-based Knowledge Distillation for YOLOv8 continual learning.

Uses ``register_forward_hook`` to capture intermediate feature maps from
the student model at a specified layer.  A frozen snapshot of teacher
weights is stored as a ``state_dict``, and KD loss is computed by
temporarily swapping the student's weights with the teacher's.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureKDLoss(nn.Module):
    """Compute feature-based KD loss between a teacher snapshot and student.

    Instead of maintaining a separate teacher model object (which
    triggers Ultralytics internal machinery), this module stores the
    teacher's ``state_dict`` and temporarily swaps the student's
    weights during KD loss computation.

    Attributes:
        teacher_state: Frozen teacher state dict snapshot.
        student: Reference to the trainable student model.
        layer_idx: The layer index to hook into (default 21).
    """

    def __init__(
        self,
        teacher_state: Dict[str, torch.Tensor],
        student: nn.Module,
        layer_idx: int = 21,
    ) -> None:
        """Initialise the KD loss module.

        Args:
            teacher_state: Frozen teacher state dict.
            student: Trainable student model (``YOLOv8Wrapper``).
            layer_idx: Index of the layer to extract features from.
        """
        super().__init__()
        self.teacher_state = teacher_state
        self.student = student
        self.layer_idx = layer_idx

        self.teacher_features: Optional[torch.Tensor] = None
        self.student_features: Optional[torch.Tensor] = None

        self._student_hook = None
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Hook management
    # ------------------------------------------------------------------ #
    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Resolve the target layer from the model's sequential backbone.

        Supports:
        - ``YOLOv8Wrapper``: accesses ``model.inner_model[idx]``
        - Raw ``DetectionModel``: accesses ``model.model[idx]``

        Args:
            model: A model whose inner sequential is accessible.

        Returns:
            The ``nn.Module`` at ``layer_idx``.
        """
        if hasattr(model, "yolo"):
            return model.yolo.model.model[self.layer_idx]
        if hasattr(model, "inner_model"):
            return model.inner_model[self.layer_idx]
        if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
            return model.model[self.layer_idx]
        children = list(model.children())
        if len(children) > self.layer_idx:
            return children[self.layer_idx]
        raise AttributeError(
            "Cannot locate inner sequential model for hook registration."
        )

    def setup_hooks(self) -> None:
        """Register forward hook on the target layer of the student."""
        self.remove_hooks()

        student_layer = self._get_target_layer(self.student)

        def _hook_fn(
            _module: nn.Module,
            _input: Any,
            output: torch.Tensor,
        ) -> None:
            self.student_features = output

        self._student_hook = student_layer.register_forward_hook(_hook_fn)
        logger.info("KD hook registered on layer %d of student", self.layer_idx)

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        if self._student_hook is not None:
            self._student_hook.remove()
            self._student_hook = None
        self.student_features = None
        self.teacher_features = None

    # ------------------------------------------------------------------ #
    #  Loss computation
    # ------------------------------------------------------------------ #
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KD loss via weight-swap approach.

        1. Save the student's current (training) state dict.
        2. Load the frozen teacher weights into the student.
        3. Forward pass → capture teacher features via hook.
        4. Restore student weights.
        5. Return MSE(student_features, teacher_features).

        The student's feature map should already exist from the training
        forward pass that occurred *before* this call.

        Args:
            x: Input tensor (same batch used for student forward pass).

        Returns:
            Scalar MSE loss between student and teacher features.

        Raises:
            RuntimeError: If hooks are not set up or features are missing.
        """
        if self._student_hook is None:
            raise RuntimeError("Hooks not set up. Call setup_hooks() first.")

        if self.student_features is None:
            raise RuntimeError(
                "Student features not captured. Run a student forward pass first."
            )

        # Save student features captured during training forward
        student_feat = self.student_features.detach().clone()

        # Get inner YOLO model reference for state_dict swap
        inner = self.student.yolo.model

        # Save current student weights
        student_state = {
            k: v.clone() for k, v in inner.state_dict().items()
        }

        # Swap in teacher weights
        inner.load_state_dict(self.teacher_state)
        was_training = inner.training
        inner.eval()

        # Forward with teacher weights — hook captures teacher features
        # Call .forward() explicitly to avoid Ultralytics __call__ pipeline
        with torch.no_grad():
            inner.forward(x)
        teacher_feat = self.student_features.detach().clone()

        # Restore student weights and training state
        inner.load_state_dict(student_state)
        if was_training:
            inner.train()

        # Store for inspection in tests
        self.teacher_features = teacher_feat
        self.student_features = student_feat

        loss = self.mse(student_feat.float(), teacher_feat.float())
        return loss

    def __del__(self) -> None:
        """Cleanup hooks on garbage collection."""
        self.remove_hooks()
