"""
Unit tests for Temporal Multi-Head Self-Attention (MHSA).
"""

import pytest
import torch

from earthquake_detection.models.attention import TemporalMHSA1D


def test_attention_shapes():
    B, C, L = 2, 64, 150
    attn = TemporalMHSA1D(embed_dim=C, num_heads=4, dropout=0.0)
    x = torch.randn(B, C, L)
    out = attn(x)

    assert out.shape == (B, C, L)
    assert not torch.isnan(out).any()


def test_attention_invalid_heads():
    with pytest.raises(ValueError):
        # 65 not divisible by 4
        TemporalMHSA1D(embed_dim=65, num_heads=4)


def test_attention_gradient_flow():
    B, C, L = 2, 32, 50
    attn = TemporalMHSA1D(embed_dim=C, num_heads=2)
    x = torch.randn(B, C, L, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
