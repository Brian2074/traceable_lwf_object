"""Tests for YOLOv8Wrapper — model loading, FedRep split, and freeze/unfreeze."""

import sys
import os

import pytest
import torch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.yolov8_wrapper import YOLOv8Wrapper


# ── Shared fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def wrapper() -> YOLOv8Wrapper:
    """Create a YOLOv8Wrapper once for all tests in this module."""
    return YOLOv8Wrapper(weights="yolov8n.pt")


# ── Tests ───────────────────────────────────────────────────────────────

class TestModelLoading:
    """Model loading and structural validation."""

    def test_model_loads(self, wrapper: YOLOv8Wrapper) -> None:
        """Wrapper loads YOLOv8n with the expected number of layers."""
        assert wrapper.inner_model is not None
        assert len(wrapper.inner_model) >= 23, (
            f"Expected ≥ 23 layers, got {len(wrapper.inner_model)}"
        )

    def test_body_has_22_layers(self, wrapper: YOLOv8Wrapper) -> None:
        """Body should contain layers 0..21 (22 modules)."""
        assert len(wrapper.body) == 22

    def test_head_is_detect(self, wrapper: YOLOv8Wrapper) -> None:
        """Head should be the Detect module (has attribute `nc`)."""
        assert hasattr(wrapper.head, "nc"), "Head should have `nc` (num_classes)"

    def test_num_classes_matches(self, wrapper: YOLOv8Wrapper) -> None:
        """Wrapper's num_classes equals the Detect head's nc."""
        assert wrapper.num_classes == wrapper.head.nc


class TestFreezeUnfreeze:
    """FedRep freeze/unfreeze mechanism."""

    def test_freeze_body(self, wrapper: YOLOv8Wrapper) -> None:
        """After freeze_body(), all body params should have requires_grad=False."""
        wrapper.freeze_body()
        for name, param in wrapper.body.named_parameters():
            assert not param.requires_grad, f"Body param {name} still trainable"
        # Head should be unaffected (restore state for next tests)
        wrapper.unfreeze_body()

    def test_freeze_head(self, wrapper: YOLOv8Wrapper) -> None:
        """After freeze_head(), all head params should have requires_grad=False."""
        wrapper.freeze_head()
        for name, param in wrapper.head.named_parameters():
            assert not param.requires_grad, f"Head param {name} still trainable"
        wrapper.unfreeze_head()

    def test_freeze_body_leaves_head_trainable(self, wrapper: YOLOv8Wrapper) -> None:
        """Freezing body should not affect head."""
        wrapper.unfreeze_body()
        wrapper.unfreeze_head()
        wrapper.freeze_body()

        # Body frozen
        for param in wrapper.body.parameters():
            assert not param.requires_grad

        # Head still trainable
        head_params = list(wrapper.head.parameters())
        assert len(head_params) > 0, "Head should have parameters"
        for param in head_params:
            assert param.requires_grad

        wrapper.unfreeze_body()

    def test_freeze_head_leaves_body_trainable(self, wrapper: YOLOv8Wrapper) -> None:
        """Freezing head should not affect body."""
        wrapper.unfreeze_body()
        wrapper.unfreeze_head()
        wrapper.freeze_head()

        # Head frozen
        for param in wrapper.head.parameters():
            assert not param.requires_grad

        # Body still trainable
        body_params = list(wrapper.body.parameters())
        assert len(body_params) > 0
        for param in body_params:
            assert param.requires_grad

        wrapper.unfreeze_head()

    def test_unfreeze_all(self, wrapper: YOLOv8Wrapper) -> None:
        """After unfreezing both, all params should be trainable."""
        wrapper.freeze_body()
        wrapper.freeze_head()
        wrapper.unfreeze_body()
        wrapper.unfreeze_head()

        for name, param in wrapper.named_parameters():
            assert param.requires_grad, f"Param {name} not trainable after unfreeze"

    def test_get_body_params_empty_when_frozen(
        self, wrapper: YOLOv8Wrapper
    ) -> None:
        """get_body_params() should return empty list when body is frozen."""
        wrapper.freeze_body()
        assert len(wrapper.get_body_params()) == 0
        wrapper.unfreeze_body()

    def test_get_head_params_empty_when_frozen(
        self, wrapper: YOLOv8Wrapper
    ) -> None:
        """get_head_params() should return empty list when head is frozen."""
        wrapper.freeze_head()
        assert len(wrapper.get_head_params()) == 0
        wrapper.unfreeze_head()

    def test_get_body_params_nonempty_when_unfrozen(
        self, wrapper: YOLOv8Wrapper
    ) -> None:
        """get_body_params() should return params when body is trainable."""
        wrapper.unfreeze_body()
        assert len(wrapper.get_body_params()) > 0

    def test_get_head_params_nonempty_when_unfrozen(
        self, wrapper: YOLOv8Wrapper
    ) -> None:
        """get_head_params() should return params when head is trainable."""
        wrapper.unfreeze_head()
        assert len(wrapper.get_head_params()) > 0
