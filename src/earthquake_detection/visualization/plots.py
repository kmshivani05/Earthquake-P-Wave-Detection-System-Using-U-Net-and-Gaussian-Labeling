"""
Scientific visualization module for earthquake waveforms, predictions, and metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific plotting style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_three_channel_waveform(
    waveform: np.ndarray,
    p_arrival: Optional[int] = None,
    s_arrival: Optional[int] = None,
    sampling_rate: Optional[float] = 100.0,
    channel_names: Optional[List[str]] = None,
    title: str = "3-Channel Seismic Seismogram",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot 3-channel seismic waveform with P and S phase picks.

    Args:
        waveform: Shape (3, T) or (T, 3).
        p_arrival: Sample index of P-wave arrival.
        s_arrival: Sample index of S-wave arrival.
        sampling_rate: Sampling frequency in Hz. If None, plots against samples.
        channel_names: Channel labels, default ['Component 1 (Z)', 'Component 2 (N)', 'Component 3 (E)'].
        title: Plot title.
        save_path: Optional filepath to save the figure.
    """
    arr = np.array(waveform, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T

    C, T = arr.shape
    names = channel_names or ["Vertical (Z)", "North-South (N)", "East-West (E)"]
    if len(names) < C:
        names = [f"Channel {i+1}" for i in range(C)]

    if sampling_rate is not None and sampling_rate > 0:
        time = np.arange(T) / sampling_rate
        time_label = "Time (seconds)"
        p_time = (p_arrival / sampling_rate) if (p_arrival is not None and p_arrival >= 0) else None
        s_time = (s_arrival / sampling_rate) if (s_arrival is not None and s_arrival >= 0) else None
    else:
        time = np.arange(T)
        time_label = "Sample Index"
        p_time = p_arrival if (p_arrival is not None and p_arrival >= 0) else None
        s_time = s_arrival if (s_arrival is not None and s_arrival >= 0) else None

    fig, axes = plt.subplots(C, 1, figsize=(12, 2.2 * C), sharex=True, dpi=150)
    if C == 1:
        axes = [axes]

    colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    for idx, (ax, name) in enumerate(zip(axes, names)):
        c = colors[idx % len(colors)]
        ax.plot(time, arr[idx], color=c, lw=0.9, alpha=0.9, label=name)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        if p_time is not None:
            ax.axvline(
                p_time,
                color="#d62728",
                linestyle="--",
                lw=1.6,
                label="P-arrival" if idx == 0 else "",
            )
        if s_time is not None:
            ax.axvline(
                s_time,
                color="#ff7f0e",
                linestyle=":",
                lw=1.6,
                label="S-arrival" if idx == 0 else "",
            )

        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(time_label, fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_prediction_alignment(
    waveform: np.ndarray,
    target: Optional[np.ndarray],
    predicted_scores: np.ndarray,
    actual_p_arrival: Optional[int] = None,
    predicted_p_arrival: Optional[int] = None,
    sampling_rate: Optional[float] = 100.0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot seismic trace, true Gaussian target, and predicted temporal probability density."""
    arr = np.array(waveform, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    z_trace = arr[0] if arr.ndim == 2 else arr
    T = len(z_trace)

    if sampling_rate is not None and sampling_rate > 0:
        time = np.arange(T) / sampling_rate
        time_label = "Time (seconds)"
        act_t = (actual_p_arrival / sampling_rate) if (actual_p_arrival is not None and actual_p_arrival >= 0) else None
        pred_t = (predicted_p_arrival / sampling_rate) if (predicted_p_arrival is not None and predicted_p_arrival >= 0) else None
    else:
        time = np.arange(T)
        time_label = "Sample Index"
        act_t = actual_p_arrival if (actual_p_arrival is not None and actual_p_arrival >= 0) else None
        pred_t = predicted_p_arrival if (predicted_p_arrival is not None and predicted_p_arrival >= 0) else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=150)

    # Subplot 1: Waveform & Arrival Markers
    ax1.plot(time, z_trace, color="#2c3e50", lw=0.9, label="Normalized Seismic Trace (Z)")
    if act_t is not None:
        ax1.axvline(act_t, color="#27ae60", linestyle="-", lw=2.0, label=f"Ground Truth P ({act_t:.2f})")
    if pred_t is not None:
        ax1.axvline(pred_t, color="#e74c3c", linestyle="--", lw=2.0, label=f"Predicted P ({pred_t:.2f})")

    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Seismic Waveform and P-Wave Phase Picking", fontsize=11, fontweight="bold")

    # Subplot 2: Gaussian Target vs Predicted Curve
    if target is not None:
        t_arr = np.squeeze(target)
        ax2.plot(time, t_arr, color="#27ae60", lw=1.8, alpha=0.7, label="Ground Truth Gaussian Target")
    p_arr = np.squeeze(predicted_scores)
    ax2.plot(time, p_arr, color="#e74c3c", lw=2.0, label="1D U-Net++ Predicted Score")

    ax2.set_xlabel(time_label, fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Temporal Gaussian Target vs Model Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    ax.plot(epochs, train_losses, color="#2980b9", lw=1.8, marker="o", ms=3, label="Training Loss")
    ax.plot(epochs, val_losses, color="#e67e22", lw=1.8, marker="s", ms=3, label="Validation Loss")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_loss = val_losses[best_epoch - 1]
    ax.scatter([best_epoch], [best_loss], color="#c0392b", s=70, zorder=5, label=f"Best Val (Epoch {best_epoch})")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("1D U-Net++ Training & Validation Loss", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_error_distribution(
    errors_samples: List[float],
    tolerance_samples: int = 20,
    sampling_rate: Optional[float] = 100.0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot arrival-time residual error histogram with tolerance window."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    if sampling_rate is not None and sampling_rate > 0:
        errs = np.array(errors_samples) * (1000.0 / sampling_rate)
        tol = tolerance_samples * (1000.0 / sampling_rate)
        unit = "ms"
    else:
        errs = np.array(errors_samples)
        tol = tolerance_samples
        unit = "samples"

    bins = np.linspace(-tol * 2, tol * 2, 40)
    ax.hist(errs, bins=bins, color="#3498db", edgecolor="#2980b9", alpha=0.8, density=True)
    ax.axvline(0, color="#2c3e50", linestyle="-", lw=1.2, label="Zero Error")
    ax.axvline(-tol, color="#e74c3c", linestyle="--", lw=1.4, label=f"-Tolerance (±{tol:.1f} {unit})")
    ax.axvline(tol, color="#e74c3c", linestyle="--", lw=1.4)

    mean_err = np.mean(errs) if len(errs) > 0 else 0.0
    med_err = np.median(errs) if len(errs) > 0 else 0.0

    ax.set_xlabel(f"Arrival Time Error ({unit}) [Predicted - True]", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_title(
        f"Arrival Time Residuals (Mean: {mean_err:.2f} {unit}, Median: {med_err:.2f} {unit})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_noise_robustness_curve(
    benchmark_results: Dict[str, Dict[str, Any]],
    metric_key: str = "f1_score",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot performance metric degradation across noise levels."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    markers = ["o-", "s--", "^:"]
    colors = ["#2980b9", "#27ae60", "#e67e22"]

    for idx, (n_type, snr_dict) in enumerate(benchmark_results.items()):
        labels = list(snr_dict.keys())
        values = [getattr(snr_dict[k], metric_key) if hasattr(snr_dict[k], metric_key) else snr_dict[k][metric_key] for k in labels]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.plot(labels, values, m, color=c, lw=2.0, ms=6, label=f"Noise: {n_type.capitalize()}")

    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10)
    ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=10)
    ax.set_title(f"Noise Robustness Benchmark ({metric_key.replace('_', ' ').title()})", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig
