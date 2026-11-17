"""
Butterworth bandpass filtering for 3-channel seismic waveforms.

Supports both zero-phase filtering (offline mode) and causal filtering (real-time mode).
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal


def design_butterworth_bandpass(
    lowcut: float = 0.5,
    highcut: float = 10.0,
    sampling_rate: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """Design second-order sections (SOS) for Butterworth bandpass filter.

    Args:
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        sampling_rate: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        sos: Second-order sections representation of filter.

    Raises:
        ValueError: If cutoffs violate Nyquist criterion (0 < lowcut < highcut < nyquist).
    """
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    nyquist = 0.5 * sampling_rate

    if lowcut <= 0:
        raise ValueError(f"lowcut frequency must be positive, got {lowcut}")
    if highcut <= lowcut:
        raise ValueError(f"highcut ({highcut}) must be greater than lowcut ({lowcut})")
    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be less than Nyquist frequency ({nyquist} Hz)"
        )

    # Use second-order sections (SOS) for superior numerical stability over transfer function (ba)
    sos = signal.butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return sos


def apply_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 10.0,
    sampling_rate: float = 100.0,
    order: int = 4,
    zero_phase: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """Apply Butterworth bandpass filter across seismic temporal dimension.

    NOTE ON ZERO-PHASE FILTERING:
        When `zero_phase=True` (`scipy.signal.sosfiltfilt`), the filter is applied
        in both forward and reverse directions. This produces zero phase distortion,
        which preserves phase arrival onset positions without temporal delay.
        However, it uses future samples and is strictly non-causal (valid for offline
        training and benchmark evaluation). For real-time causal simulation, set
        `zero_phase=False` (`scipy.signal.sosfilt`).

    Args:
        data: Seismic waveform array of shape (..., T) or (T, C).
        lowcut: Low cutoff in Hz.
        highcut: High cutoff in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.
        zero_phase: If True, uses forward-backward zero-phase filtering; if False, causal.
        axis: Temporal axis along which to filter.

    Returns:
        Filtered waveform array of same shape and float32 dtype.
    """
    sos = design_butterworth_bandpass(
        lowcut=lowcut,
        highcut=highcut,
        sampling_rate=sampling_rate,
        order=order,
    )

    if zero_phase:
        filtered = signal.sosfiltfilt(sos, data, axis=axis)
    else:
        filtered = signal.sosfilt(sos, data, axis=axis)

    return filtered.astype(np.float32)
