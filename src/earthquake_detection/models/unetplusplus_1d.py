"""
1D U-Net++ (Nested U-Net) for temporal seismic phase picking.

Implements the complete X^{i,j} dense nested topology with configurable depth,
multi-scale dilated convolutions, bottleneck Multi-Head Self-Attention, and deep supervision.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from earthquake_detection.models.dilated_blocks import MultiScaleDilatedConv1D
from earthquake_detection.models.attention import TemporalMHSA1D


class ConvBlock1D(nn.Module):
    """Standard double 1D convolutional node with BatchNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.act2(self.bn2(self.conv2(out)))
        return out


class UNetPlusPlus1D(nn.Module):
    """1D U-Net++ architecture with nested skip pathways.

    Notation:
        X^{i,j}: node at depth level i and nested skip column j.
        i in [0, depth-1] (downsampling levels)
        j in [0, depth-1-i] (nested connections)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dilated_convolutions: bool = False,
        dilation_rates: Optional[List[int]] = None,
        attention: bool = False,
        attention_heads: int = 4,
        deep_supervision: bool = False,
        dropout: float = 0.1,
    ):
        """Initialize 1D U-Net++.

        Args:
            in_channels: Number of input seismic channels (e.g. 3 for 3-component seismograms).
            num_classes: Number of output channels (1 for continuous Gaussian P-wave score).
            base_channels: Channels at depth level 0.
            depth: Number of downsampling resolution levels (typically 4).
            dilated_convolutions: If True, integrates multi-scale dilated convolutions.
            dilation_rates: Dilation rates for multi-scale blocks (e.g. [1, 2, 4, 8]).
            attention: If True, integrates bottleneck Multi-Head Self-Attention.
            attention_heads: Number of attention heads for bottleneck MHSA.
            deep_supervision: If True, computes auxiliary predictions at all X^{0,j} heads.
            dropout: Dropout probability throughout convolutional blocks.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        self.dilated_convolutions = dilated_convolutions
        self.attention = attention
        self.deep_supervision = deep_supervision

        # Calculate channels per depth level: [C_0, C_1, C_2, ..., C_{D-1}]
        self.channels = [base_channels * (2 ** i) for i in range(depth)]

        # Downsampling via MaxPool1d
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dictionary of nested convolutional blocks: key is (i, j)
        self.nodes = nn.ModuleDict()

        for i in range(depth):
            for j in range(depth - i):
                c_out = self.channels[i]

                if j == 0:
                    # Encoder node X^{i, 0}
                    c_in = in_channels if i == 0 else self.channels[i - 1]
                    self.nodes[f"node_{i}_{j}"] = ConvBlock1D(c_in, c_out, dropout=dropout)
                else:
                    # Nested node X^{i, j} receives:
                    # - all previous nodes at level i: X^{i, 0}, ..., X^{i, j-1} (j * c_out channels)
                    # - upsampled node from level below: X^{i+1, j-1} (channels[i+1])
                    c_in = (j * c_out) + self.channels[i + 1]
                    self.nodes[f"node_{i}_{j}"] = ConvBlock1D(c_in, c_out, dropout=dropout)

        # Optional multi-scale dilated block at the bottleneck
        if dilated_convolutions:
            bottleneck_channels = self.channels[-1]
            self.dilated_block = MultiScaleDilatedConv1D(
                channels=bottleneck_channels,
                dilation_rates=dilation_rates or [1, 2, 4, 8],
                dropout=dropout,
            )
        else:
            self.dilated_block = None

        # Optional bottleneck Multi-Head Self-Attention (MHSA)
        if attention:
            bottleneck_channels = self.channels[-1]
            self.attention_block = TemporalMHSA1D(
                embed_dim=bottleneck_channels,
                num_heads=attention_heads,
                dropout=dropout,
            )
        else:
            self.attention_block = None

        # Final prediction heads at top resolution level i=0
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv1d(self.channels[0], num_classes, kernel_size=1)
                for _ in range(1, depth)
            ])
        else:
            self.final_head = nn.Conv1d(self.channels[0], num_classes, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Robust temporal linear interpolation matching target sequence length."""
        if x.shape[-1] == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through U-Net++.

        Args:
            x: Input seismic waveform tensor of shape (B, in_channels, T).

        Returns:
            If deep_supervision is False:
                torch.Tensor of shape (B, num_classes, T) with continuous scores.
            If deep_supervision is True and training:
                List of torch.Tensor corresponding to predictions from each X^{0, j} head.
        """
        B, C, T = x.shape
        x_map: Dict[Tuple[int, int], torch.Tensor] = {}

        # 1. Compute encoder backbone nodes X^{i, 0}
        for i in range(self.depth):
            if i == 0:
                inp = x
            else:
                inp = self.pool(x_map[(i - 1, 0)])

            feat = self.nodes[f"node_{i}_0"](inp)

            # Apply bottleneck enhancements at the lowest resolution level
            if i == self.depth - 1:
                if self.dilated_block is not None:
                    feat = self.dilated_block(feat)
                if self.attention_block is not None:
                    feat = self.attention_block(feat)

            x_map[(i, 0)] = feat

        # 2. Compute nested dense nodes X^{i, j} column by column
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                # Gather all horizontal skip connections from same depth level
                horizontal_skips = [x_map[(i, k)] for k in range(j)]

                # Upsample feature map from lower level (i+1, j-1)
                lower_feat = x_map[(i + 1, j - 1)]
                target_len = horizontal_skips[0].shape[-1]
                upsampled = self._upsample_to(lower_feat, target_len)

                # Concatenate along channel dimension
                concat_input = torch.cat(horizontal_skips + [upsampled], dim=1)
                x_map[(i, j)] = self.nodes[f"node_{i}_{j}"](concat_input)

        # 3. Output prediction
        if self.deep_supervision and self.training:
            outputs = []
            for j in range(1, self.depth):
                out_j = self.heads[j - 1](x_map[(0, j)])
                outputs.append(out_j)
            return outputs
        elif self.deep_supervision:
            # During evaluation, return prediction from deepest dense head X^{0, depth-1}
            return self.heads[-1](x_map[(0, self.depth - 1)])
        else:
            return self.final_head(x_map[(0, self.depth - 1)])
