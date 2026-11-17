"""
Physics-informed acceleration-to-displacement processing.

Performs baseline detrending, numerical integration (cumulative trapezoid),
high-pass drift filtering, and 3D vector displacement magnitude estimation:
    D(t) = sqrt(dx(t)^2 + dy(t)^2 + dz(t)^2)

CRITICAL PHYSICAL CONSIDERATION:
    Numerical double integration of acceleration records inherently amplifies
    low-frequency baseline offsets and sensor drift. Consequently, high-pass filtering
    and baseline correction are mandatory. This module is an experimental extension
    and should be compared against direct raw/filtered acceleration waveforms.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal
from scipy import integrate


def integrate_with_drift_correction(
    signal_arr: np.ndarray,
    sampling_rate: float = 100.0,
    highpass_cutoff: float = 0.1,
    filter_order: int = 2,
    axis: int = -1,
) -> np.ndarray:
    """Perform numerical integration with baseline detrending and high-pass drift correction.

    Args:
        signal_arr: Input waveform array (acceleration or velocity).
        sampling_rate: Sampling frequency in Hz.
        highpass_cutoff: High-pass corner frequency in Hz to remove integration baseline drift.
        filter_order: Butterworth filter order for high-pass filtering.
        axis: Temporal axis along which to integrate.

    Returns:
        Integrated signal array of same shape and float32 dtype.
    """
    dt = 1.0 / sampling_rate

    # Step 1: Linear baseline detrending
    detrended = signal.detrend(signal_arr, axis=axis, type="linear")

    # Step 2: Numerical integration using cumulative trapezoid rule
    # initial=0 preserves exact sample length
    integrated = integrate.cumulative_trapezoid(detrended, dx=dt, axis=axis, initial=0.0)

    # Step 3: High-pass filter to attenuate low-frequency integration drift
    nyquist = 0.5 * sampling_rate
    if highpass_cutoff > 0 and highpass_cutoff < nyquist:
        sos = signal.butter(
            N=filter_order,
            Wn=highpass_cutoff,
            btype="highpass",
            fs=sampling_rate,
            output="sos",
        )
        integrated = signal.sosfiltfilt(sos, integrated, axis=axis)

    return integrated.astype(np.float32)


def compute_displacement_and_magnitude(
    acceleration_3c: np.ndarray,
    sampling_rate: float = 100.0,
    highpass_cutoff: float = 0.1,
    channel_axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive 3-channel velocity, 3-channel displacement, and scalar displacement magnitude.

    Args:
        acceleration_3c: 3-channel acceleration array shaped (3, T) or (T, 3).
        sampling_rate: Sampling frequency in Hz.
        highpass_cutoff: Frequency in Hz for low-frequency drift filtering.
        channel_axis: Axis denoting 3 seismic channels (0 or 1).

    Returns:
        velocity_3c: 3-channel velocity array.
        displacement_3c: 3-channel displacement array.
        displacement_magnitude: 1D array of 3D displacement vector magnitude D(t).
    """
    temporal_axis = 1 if channel_axis == 0 else 0

    # Ensure shape (3, T) for uniform processing
    if channel_axis == 1:
        acc = np.swapaxes(acceleration_3c, 0, 1)
    else:
        acc = acceleration_3c

    # 1st integration: acceleration -> velocity
    vel = integrate_with_drift_correction(
        acc,
        sampling_rate=sampling_rate,
        highpass_cutoff=highpass_cutoff,
        axis=1,
    )

    # 2nd integration: velocity -> displacement
    disp = integrate_with_drift_correction(
        vel,
        sampling_rate=sampling_rate,
        highpass_cutoff=highpass_cutoff,
        axis=1,
    )

    # 3D vector displacement magnitude: D(t) = sqrt(dx^2 + dy^2 + dz^2)
    disp_mag = np.sqrt(np.sum(disp ** 2, axis=0)).astype(np.float32)

    if channel_axis == 1:
        vel = np.swapaxes(vel, 0, 1)
        disp = np.swapaxes(disp, 0, 1)

    return vel, disp, disp_mag
