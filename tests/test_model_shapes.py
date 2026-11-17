"""
Unit tests for 1D U-Net++ model forward pass and tensor dimensions.
"""

import pytest
import torch

from earthquake_detection.models.unetplusplus_1d import UNetPlusPlus1D
from earthquake_detection.training.losses import build_loss


def test_baseline_unetplusplus_shapes():
    B, C, T = 2, 3, 500
    model = UNetPlusPlus1D(
        in_channels=C,
        num_classes=1,
        base_channels=16,
        depth=3,
        dilated_convolutions=False,
        attention=False,
        deep_supervision=False,
    )
    x = torch.randn(B, C, T)
    out = model(x)
    assert out.shape == (B, 1, T)


def test_dilated_and_attention_model_shapes():
    B, C, T = 2, 3, 400
    model = UNetPlusPlus1D(
        in_channels=C,
        num_classes=1,
        base_channels=16,
        depth=3,
        dilated_convolutions=True,
        dilation_rates=[1, 2, 4],
        attention=True,
        attention_heads=4,
        deep_supervision=False,
    )
    x = torch.randn(B, C, T)
    out = model(x)
    assert out.shape == (B, 1, T)


def test_deep_supervision_shapes():
    B, C, T = 2, 3, 300
    model = UNetPlusPlus1D(
        in_channels=C,
        num_classes=1,
        base_channels=8,
        depth=3,
        deep_supervision=True,
    )
    x = torch.randn(B, C, T)

    # During training mode -> list of outputs
    model.train()
    out_train = model(x)
    assert isinstance(out_train, list)
    assert len(out_train) == 2  # heads at X^{0, 1} and X^{0, 2}
    for head_out in out_train:
        assert head_out.shape == (B, 1, T)

    # During eval mode -> single tensor from deepest head
    model.eval()
    out_eval = model(x)
    assert isinstance(out_eval, torch.Tensor)
    assert out_eval.shape == (B, 1, T)


def test_gradient_backward_pass():
    B, C, T = 2, 3, 200
    model = UNetPlusPlus1D(
        in_channels=C,
        num_classes=1,
        base_channels=8,
        depth=3,
        dilated_convolutions=True,
        attention=True,
        attention_heads=2,
    )
    criterion = build_loss("mse")
    x = torch.randn(B, C, T)
    target = torch.randn(B, 1, T)

    model.train()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Verify gradients computed for model weights
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grads
