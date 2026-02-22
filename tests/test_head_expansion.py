"""Tests for dynamic YOLOv8 Detect head expansion."""

import copy
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.yolov8_wrapper import YOLOv8Wrapper


@pytest.fixture()
def wrapper() -> YOLOv8Wrapper:
    """Create a fresh YOLOv8Wrapper for each test."""
    return YOLOv8Wrapper(weights="yolov8n.pt")


class TestHeadExpansion:
    """Dynamic cv3 classification branch expansion."""

    def test_expand_increases_nc(self, wrapper: YOLOv8Wrapper) -> None:
        """After expansion, detect.nc should equal new_num_classes."""
        old_nc = wrapper.num_classes
        new_nc = old_nc + 5
        wrapper.expand_detect_head(new_nc)

        assert wrapper.head.nc == new_nc
        assert wrapper.num_classes == new_nc

    def test_expand_updates_no(self, wrapper: YOLOv8Wrapper) -> None:
        """detect.no should be updated to nc + reg_max * 4."""
        old_nc = wrapper.num_classes
        new_nc = old_nc + 5
        wrapper.expand_detect_head(new_nc)

        expected_no = new_nc + wrapper.head.reg_max * 4
        assert wrapper.head.no == expected_no

    def test_old_weights_preserved(self, wrapper: YOLOv8Wrapper) -> None:
        """Old class weights must be copied exactly into the expanded conv."""
        # Snapshot old weights for all cv3 branches
        old_weights = {}
        old_biases = {}
        for i, cv3 in enumerate(wrapper.head.cv3):
            last_conv = wrapper._find_last_conv2d(cv3)
            old_weights[i] = last_conv.weight.data.clone()
            if last_conv.bias is not None:
                old_biases[i] = last_conv.bias.data.clone()

        old_nc = wrapper.num_classes
        new_nc = old_nc + 3
        wrapper.expand_detect_head(new_nc)

        for i, cv3 in enumerate(wrapper.head.cv3):
            new_conv = wrapper._find_last_conv2d(cv3)
            old_out = old_weights[i].size(0)

            # Old channels must match exactly
            assert torch.allclose(
                new_conv.weight[:old_out], old_weights[i]
            ), f"cv3[{i}] old weights diverged"

            if i in old_biases and new_conv.bias is not None:
                assert torch.allclose(
                    new_conv.bias[:old_out], old_biases[i]
                ), f"cv3[{i}] old biases diverged"

    def test_new_weights_not_zero(self, wrapper: YOLOv8Wrapper) -> None:
        """New channels should be initialised (Kaiming), not all zeros."""
        old_nc = wrapper.num_classes
        new_nc = old_nc + 5
        old_out_per_scale = {}
        for i, cv3 in enumerate(wrapper.head.cv3):
            last_conv = wrapper._find_last_conv2d(cv3)
            old_out_per_scale[i] = last_conv.weight.size(0)

        wrapper.expand_detect_head(new_nc)

        for i, cv3 in enumerate(wrapper.head.cv3):
            new_conv = wrapper._find_last_conv2d(cv3)
            old_out = old_out_per_scale[i]
            new_part = new_conv.weight[old_out:]
            assert not torch.all(new_part == 0), (
                f"cv3[{i}] new channels are all zeros — init failed"
            )

    def test_forward_after_expansion(self, wrapper: YOLOv8Wrapper) -> None:
        """Model should run a forward pass without errors after expansion."""
        old_nc = wrapper.num_classes
        new_nc = old_nc + 5
        wrapper.expand_detect_head(new_nc)

        # Dummy input (batch=1, 3 channels, 640x640)
        dummy = torch.randn(1, 3, 640, 640)
        if next(wrapper.parameters()).is_cuda:
            dummy = dummy.cuda()

        # Use the raw nn.Module forward (yolo.model) directly to avoid
        # triggering the YOLO trainer/dataset validation pipeline.
        wrapper.yolo.model.eval()
        with torch.no_grad():
            output = wrapper.yolo.model(dummy)
        # Should not raise; output should exist
        assert output is not None

    def test_expand_fails_if_not_larger(self, wrapper: YOLOv8Wrapper) -> None:
        """Expansion should raise ValueError if new_nc <= old_nc."""
        with pytest.raises(ValueError, match="must be >"):
            wrapper.expand_detect_head(wrapper.num_classes)

        with pytest.raises(ValueError, match="must be >"):
            wrapper.expand_detect_head(wrapper.num_classes - 1)

    def test_double_expansion(self, wrapper: YOLOv8Wrapper) -> None:
        """Head can be expanded multiple times sequentially."""
        nc0 = wrapper.num_classes
        nc1 = nc0 + 3
        nc2 = nc1 + 4

        wrapper.expand_detect_head(nc1)
        assert wrapper.num_classes == nc1

        wrapper.expand_detect_head(nc2)
        assert wrapper.num_classes == nc2
        assert wrapper.head.nc == nc2
