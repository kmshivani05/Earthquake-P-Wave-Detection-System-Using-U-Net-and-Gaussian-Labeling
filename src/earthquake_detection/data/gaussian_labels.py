"""
Gaussian target label generation for seismic phase arrival detection.

Implements the continuous temporal Gaussian distribution target:
    Y(t) = exp( - (t - t_p)^2 / (2 * sigma^2) )
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_gaussian_target(
    waveform_length: int,
    arrival_index: Optional[int],
    sigma: float = 20.0,
    normalize_peak: bool = True,
    allow_missing: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generate a 1D continuous Gaussian distribution target around arrival_index.

    Args:
        waveform_length: Total number of temporal samples T.
        arrival_index: The annotated arrival sample index t_p. If None or negative,
                       treated as missing label.
        sigma: Standard deviation of Gaussian bell curve in samples. Must be > 0.
        normalize_peak: If True, peak amplitude is strictly 1.0 at t = arrival_index.
        allow_missing: If True, returns an all-zero array when arrival_index is None
                       or negative. If False, raises ValueError.
        dtype: Output numpy array data type.

    Returns:
        np.ndarray of shape (waveform_length,) representing continuous target scores [0, 1].

    Raises:
        ValueError: If waveform_length <= 0, sigma <= 0, or arrival_index is invalid
                    when allow_missing is False.
    """
    if waveform_length <= 0:
        raise ValueError(f"waveform_length must be positive, got {waveform_length}")
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    # Handle missing or unannotated phase arrival
    if arrival_index is None or arrival_index < 0:
        if allow_missing:
            return np.zeros(waveform_length, dtype=dtype)
        raise ValueError(f"Missing arrival index provided with allow_missing=False: {arrival_index}")

    # Validate boundary conditions
    if arrival_index >= waveform_length:
        if allow_missing:
            return np.zeros(waveform_length, dtype=dtype)
        raise ValueError(
            f"arrival_index ({arrival_index}) exceeds waveform_length ({waveform_length})"
        )

    # Vectorized temporal index calculation
    t = np.arange(waveform_length, dtype=np.float64)
    squared_diff = (t - float(arrival_index)) ** 2
    variance = 2.0 * (float(sigma) ** 2)
    target = np.exp(-squared_diff / variance)

    if normalize_peak:
        peak_val = target[arrival_index]
        if peak_val > 0:
            target = target / peak_val

    return target.astype(dtype)


def generate_gaussian_target_torch(
    waveform_length: int,
    arrival_index: Optional[Union[int, torch.Tensor]],
    sigma: float = 20.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a 1D Gaussian target tensor directly on PyTorch device.

    Args:
        waveform_length: Length of sequence T.
        arrival_index: Arrival index scalar or 0D tensor.
        sigma: Standard deviation in samples.
        device: PyTorch device.
        dtype: Output tensor data type.

    Returns:
        torch.Tensor of shape (1, waveform_length) with values in [0, 1].
    """
    if arrival_index is None:
        return torch.zeros((1, waveform_length), dtype=dtype, device=device)

    if isinstance(arrival_index, torch.Tensor):
        idx = arrival_index.item()
    else:
        idx = arrival_index

    if idx < 0 or idx >= waveform_length:
        return torch.zeros((1, waveform_length), dtype=dtype, device=device)

    t = torch.arange(waveform_length, dtype=torch.float32, device=device)
    target = torch.exp(-((t - float(idx)) ** 2) / (2.0 * (float(sigma) ** 2)))
    return target.unsqueeze(0).to(dtype)


def plot_gaussian_label_alignment(
    waveform: np.ndarray,
    target: np.ndarray,
    arrival_index: int,
    channel_idx: int = 0,
    window: Optional[Tuple[int, int]] = None,
    title: str = "P-Wave Arrival Annotation & Gaussian Target Alignment",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize waveform channel together with aligned continuous Gaussian target.

    Args:
        waveform: Array of shape (T,) or (T, C) or (C, T).
        target: 1D array of shape (T,).
        arrival_index: Annotated arrival sample index.
        channel_idx: Channel to display if waveform is multi-channel.
        window: Optional tuple of (start_sample, end_sample) to zoom in.
        title: Title of figure.
        save_path: Optional filepath to save the figure.

    Returns:
        matplotlib Figure object.
    """
    # Normalize waveform orientation to (T,)
    if waveform.ndim == 2:
        if waveform.shape[0] < waveform.shape[1]:
            sig = waveform[channel_idx, :]
        else:
            sig = waveform[:, channel_idx]
    else:
        sig = waveform

    T = len(sig)
    t = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=150)

    # Plot seismic waveform on left y-axis
    ax1.plot(t, sig, color="#2b5c8f", lw=0.9, alpha=0.85, label=f"Seismic Trace (Ch {channel_idx})")
    ax1.axvline(
        x=arrival_index,
        color="#d9381e",
        linestyle="--",
        lw=1.8,
        label=f"Annotated Arrival (t={arrival_index})",
    )
    ax1.set_xlabel("Time (Samples)")
    ax1.set_ylabel("Amplitude", color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")

    # Plot continuous Gaussian target on secondary right y-axis
    ax2 = ax1.twinx()
    ax2.plot(t, target, color="#e67e22", lw=2.2, label=f"Gaussian Target (Peak={target.max():.2f})")
    ax2.set_ylabel("Target Probability Density", color="#e67e22")
    ax2.set_ylim(-0.05, 1.15)
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    if window is not None:
        ax1.set_xlim(window[0], window[1])

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig
