"""
Multi-scale 1D Dilated Convolution module.

Expands temporal receptive fields exponentially across seismic waveforms
without losing temporal resolution or subsampling high-frequency phase onsets.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDilatedConv1D(nn.Module):
    """Parallel multi-branch dilated convolutions with feature fusion and residual skip.

    Branch k computes:
        Y_k = Conv1D(X, dilation=d_k, padding=d_k*(kernel_size-1)//2)
    Output:
        Y = Conv1x1(Concat([Y_1, Y_2, ..., Y_M])) + X
    """

    def __init__(
        self,
        channels: int,
        dilation_rates: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize multi-scale dilated block.

        Args:
            channels: Input and output channel dimension.
            dilation_rates: List of dilation rates (e.g. [1, 2, 4, 8]).
            kernel_size: Kernel size per branch (default 3, odd integer required for symmetric padding).
            dropout: Dropout probability.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer for symmetric padding, got {kernel_size}")

        self.dilation_rates = dilation_rates or [1, 2, 4, 8]
        self.branches = nn.ModuleList()

        for d in self.dilation_rates:
            padding = d * (kernel_size - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        num_branches = len(self.dilation_rates)
        # Fusion 1x1 convolution
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * num_branches, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, T)

        Returns:
            Tensor of shape (B, C, T) with expanded receptive field.
        """
        branch_outs = [branch(x) for branch in self.branches]
        concat_feats = torch.cat(branch_outs, dim=1)
        fused = self.fusion(concat_feats)
        return fused + x  # Residual connection
