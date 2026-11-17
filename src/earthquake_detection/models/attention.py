"""
Multi-Head Self-Attention (MHSA) for temporal 1D seismic representations.

Applies scaled dot-product self-attention across low-resolution temporal feature maps:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMHSA1D(nn.Module):
    """Temporal Multi-Head Self-Attention with residual connection and LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize MHSA module.

        Args:
            embed_dim: Channel dimension C of feature map. Must be divisible by num_heads.
            num_heads: Number of parallel attention heads.
            dropout: Attention and projection dropout probability.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, L), where L is temporal sequence length.

        Returns:
            Output tensor of shape (B, C, L).
        """
        B, C, L = x.shape
        if C != self.embed_dim:
            raise ValueError(f"Input channel dim {C} != expected embed_dim {self.embed_dim}")

        # Permute (B, C, L) -> (B, L, C) for sequence attention
        residual = x
        x_seq = x.permute(0, 2, 1)  # shape (B, L, C)
        normed = self.norm(x_seq)

        # Linear projections & split into heads: (B, L, heads, head_dim) -> (B, heads, L, head_dim)
        Q = self.q_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context output: (B, heads, L, head_dim) -> (B, L, C)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, C)
        out = self.proj_dropout(self.out_proj(context))

        # Permute back to (B, C, L) and add residual
        out_conv = out.permute(0, 2, 1) + residual
        return out_conv
