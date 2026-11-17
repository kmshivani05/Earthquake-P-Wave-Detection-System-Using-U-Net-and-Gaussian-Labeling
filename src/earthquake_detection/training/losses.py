"""
Loss functions for continuous Gaussian seismic target regression.

Supports Mean Squared Error (MSE), Peak-Weighted MSE, and Deep Supervision loss wrappers.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """Mean Squared Error with positive peak weight enhancement.

    Because seismic time-series are predominantly zeros (background noise),
    standard MSE can be dominated by trivial zero-prediction. Weighted MSE applies
    an extra weighting factor to samples where the Gaussian target is non-zero:
        loss = mean( (1 + peak_weight * target) * (pred - target)^2 )
    """

    def __init__(self, peak_weight: float = 5.0):
        super().__init__()
        self.peak_weight = peak_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + (self.peak_weight * target)
        squared_errors = (pred - target) ** 2
        return torch.mean(weights * squared_errors)


class DeepSupervisionLoss(nn.Module):
    """Computes total loss across multiple nested heads during deep supervision."""

    def __init__(self, base_criterion: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        self.base_criterion = base_criterion
        self.weights = weights

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return self.base_criterion(preds, target)

        num_heads = len(preds)
        if self.weights is None:
            w = [1.0 / num_heads] * num_heads
        else:
            w = self.weights

        total_loss = 0.0
        for p, weight in zip(preds, w):
            total_loss = total_loss + weight * self.base_criterion(p, target)
        return total_loss


def build_loss(loss_name: str, deep_supervision: bool = False) -> nn.Module:
    """Build loss function based on configured name.

    Args:
        loss_name: 'mse', 'weighted_mse', or 'bce_with_logits'.
        deep_supervision: Whether to wrap criterion in DeepSupervisionLoss.

    Returns:
        PyTorch nn.Module loss criterion.
    """
    name = loss_name.lower()
    if name == "mse":
        crit = nn.MSELoss()
    elif name == "weighted_mse":
        crit = WeightedMSELoss(peak_weight=5.0)
    elif name in ["bce", "bce_with_logits"]:
        crit = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose from ['mse', 'weighted_mse', 'bce_with_logits']")

    if deep_supervision:
        return DeepSupervisionLoss(crit)
    return crit
