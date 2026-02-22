"""Tests for Feature-based Knowledge Distillation hooks and loss.

Uses the state_dict swap approach: no separate teacher model object,
only a snapshot of weights that are temporarily loaded into the student.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.yolov8_wrapper import YOLOv8Wrapper
from models.feature_kd import FeatureKDLoss


@pytest.fixture(scope="module")
def student() -> YOLOv8Wrapper:
    """Create a student model (shared across module for speed)."""
    return YOLOv8Wrapper(weights="yolov8n.pt")


@pytest.fixture(scope="module")
def teacher_state(student: YOLOv8Wrapper) -> dict:
    """Snapshot the student's state dict as teacher."""
    return student.snapshot_teacher_state()


@pytest.fixture()
def kd(student: YOLOv8Wrapper, teacher_state: dict) -> FeatureKDLoss:
    """Create a FeatureKDLoss instance with hooks set up."""
    kd_loss = FeatureKDLoss(
        teacher_state=teacher_state, student=student, layer_idx=21
    )
    kd_loss.setup_hooks()
    yield kd_loss
    kd_loss.remove_hooks()


class TestHookCapture:
    """Forward hook feature capture."""

    def test_hooks_capture_student_features(
        self,
        student: YOLOv8Wrapper,
        kd: FeatureKDLoss,
    ) -> None:
        """After a student forward pass, student_features should be filled."""
        dummy = torch.randn(1, 3, 640, 640)
        student.eval()
        with torch.no_grad():
            student.forward(dummy)

        assert kd.student_features is not None, "Student features not captured"

    def test_features_are_tensors(
        self,
        student: YOLOv8Wrapper,
        kd: FeatureKDLoss,
    ) -> None:
        """Captured student features should be torch.Tensor."""
        dummy = torch.randn(1, 3, 640, 640)
        student.eval()
        with torch.no_grad():
            student.forward(dummy)

        assert isinstance(kd.student_features, torch.Tensor)


class TestKDLoss:
    """KD loss computation via state_dict swap."""

    def test_identical_models_zero_loss(
        self,
        student: YOLOv8Wrapper,
        kd: FeatureKDLoss,
    ) -> None:
        """Identical teacher and student should produce ~0 KD loss."""
        dummy = torch.randn(1, 3, 640, 640)
        student.eval()

        with torch.no_grad():
            student.forward(dummy)
        loss = kd.compute_loss(dummy)

        assert loss.item() < 1e-5, f"Expected ~0 loss, got {loss.item()}"

    def test_different_weights_positive_loss(
        self,
        student: YOLOv8Wrapper,
        teacher_state: dict,
    ) -> None:
        """After perturbing student weights, KD loss should be > 0."""
        kd = FeatureKDLoss(
            teacher_state=teacher_state, student=student, layer_idx=21
        )
        kd.setup_hooks()

        # Perturb student weights
        with torch.no_grad():
            for param in student.body.parameters():
                param.add_(torch.randn_like(param) * 0.1)
                break  # perturb just one param

        dummy = torch.randn(1, 3, 640, 640)
        student.eval()
        with torch.no_grad():
            student.forward(dummy)
        loss = kd.compute_loss(dummy)

        assert loss.item() > 0, "Expected positive KD loss after perturbation"
        kd.remove_hooks()

    def test_compute_loss_returns_features(
        self,
        student: YOLOv8Wrapper,
        kd: FeatureKDLoss,
    ) -> None:
        """After compute_loss, both teacher and student features should be stored."""
        dummy = torch.randn(1, 3, 640, 640)
        student.eval()
        with torch.no_grad():
            student.forward(dummy)
        kd.compute_loss(dummy)

        assert kd.student_features is not None
        assert kd.teacher_features is not None
        assert kd.student_features.shape == kd.teacher_features.shape


class TestHookCleanup:
    """Hook lifecycle management."""

    def test_hooks_removed(
        self,
        student: YOLOv8Wrapper,
        teacher_state: dict,
    ) -> None:
        """After remove_hooks(), hook handles should be None."""
        kd = FeatureKDLoss(
            teacher_state=teacher_state, student=student, layer_idx=21
        )
        kd.setup_hooks()
        assert kd._student_hook is not None

        kd.remove_hooks()
        assert kd._student_hook is None
        assert kd.student_features is None
        assert kd.teacher_features is None

    def test_compute_loss_fails_without_hooks(
        self,
        student: YOLOv8Wrapper,
        teacher_state: dict,
    ) -> None:
        """compute_loss should raise RuntimeError if hooks are not set up."""
        kd = FeatureKDLoss(
            teacher_state=teacher_state, student=student, layer_idx=21
        )
        dummy = torch.randn(1, 3, 640, 640)

        with pytest.raises(RuntimeError, match="Hooks not set up"):
            kd.compute_loss(dummy)

    def test_double_setup_is_safe(
        self,
        student: YOLOv8Wrapper,
        teacher_state: dict,
    ) -> None:
        """Calling setup_hooks() twice should not leak hooks."""
        kd = FeatureKDLoss(
            teacher_state=teacher_state, student=student, layer_idx=21
        )
        kd.setup_hooks()
        kd.setup_hooks()  # should remove old hooks first

        dummy = torch.randn(1, 3, 640, 640)
        student.eval()
        with torch.no_grad():
            student.forward(dummy)
        loss = kd.compute_loss(dummy)
        assert loss is not None

        kd.remove_hooks()
