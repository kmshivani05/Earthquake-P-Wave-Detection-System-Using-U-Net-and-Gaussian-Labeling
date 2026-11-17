"""Models package initialization and model factory."""

from typing import Union
import torch.nn as nn

from earthquake_detection.config import ModelConfig
from earthquake_detection.models.unetplusplus_1d import UNetPlusPlus1D
from earthquake_detection.models.dilated_blocks import MultiScaleDilatedConv1D
from earthquake_detection.models.attention import TemporalMHSA1D


def build_model(config: ModelConfig) -> nn.Module:
    """Build seismic phase picking model based on configuration."""
    if config.architecture.lower() in ["unetplusplus_1d", "unet++", "unetpp"]:
        return UNetPlusPlus1D(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            depth=config.depth,
            dilated_convolutions=config.dilated_convolutions,
            dilation_rates=config.dilation_rates,
            attention=config.attention,
            attention_heads=config.attention_heads,
            deep_supervision=config.deep_supervision,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown architecture: {config.architecture}")
