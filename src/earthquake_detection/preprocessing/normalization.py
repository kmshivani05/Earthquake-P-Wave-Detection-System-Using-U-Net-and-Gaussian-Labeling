"""
Waveform normalization routines for single-station 3-channel seismic records.

All normalization methods operate independently per-record or per-window to strictly
prevent data leakage across dataset splits.
"""

from typing import Literal
import numpy as np


NormalizationType = Literal["per_channel", "per_record", "robust", "peak", "none"]


def normalize_waveform(
    data: np.ndarray,
    method: NormalizationType = "per_channel",
    eps: float = 1e-8,
    channel_axis: int = 0,
) -> np.ndarray:
    """Normalize seismic waveform array.

    Args:
        data: Waveform array, typically shaped (C, T) or (T, C).
        method: Normalization strategy:
            - 'per_channel': Zero-mean, unit-variance standardized per channel.
            - 'per_record': Zero-mean, unit-variance standardized across all channels.
            - 'robust': Median centered and scaled by Interquartile Range (IQR).
            - 'peak': Scaled by maximum absolute amplitude to [-1, 1].
            - 'none': Return unmodified copy.
        eps: Small constant to prevent division by zero in quiet or dead channels.
        channel_axis: Axis denoting channels (0 for (C, T), 1 for (T, C)).

    Returns:
        Normalized array of identical shape and float32 dtype.
    """
    if method == "none" or method is None:
        return data.astype(np.float32, copy=True)

    arr = np.array(data, dtype=np.float32, copy=True)
    temporal_axis = 1 if channel_axis == 0 else 0

    if method == "per_channel":
        mean = np.mean(arr, axis=temporal_axis, keepdims=True)
        std = np.std(arr, axis=temporal_axis, keepdims=True)
        std = np.where(std < eps, eps, std)
        normalized = (arr - mean) / std

    elif method == "per_record":
        mean = np.mean(arr)
        std = np.std(arr)
        if std < eps:
            std = eps
        normalized = (arr - mean) / std

    elif method == "robust":
        median = np.median(arr, axis=temporal_axis, keepdims=True)
        q75 = np.percentile(arr, 75, axis=temporal_axis, keepdims=True)
        q25 = np.percentile(arr, 25, axis=temporal_axis, keepdims=True)
        iqr = q75 - q25
        iqr = np.where(iqr < eps, eps, iqr)
        normalized = (arr - median) / iqr

    elif method == "peak":
        max_abs = np.max(np.abs(arr), axis=temporal_axis, keepdims=True)
        max_abs = np.where(max_abs < eps, eps, max_abs)
        normalized = arr / max_abs

    else:
        raise ValueError(
            f"Unsupported normalization method '{method}'. "
            f"Choose from ['per_channel', 'per_record', 'robust', 'peak', 'none']."
        )

    return normalized.astype(np.float32)
