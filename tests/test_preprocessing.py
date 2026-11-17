"""
Unit tests for signal preprocessing, filtering, normalization, and displacement modules.
"""

import numpy as np
import pytest

from earthquake_detection.preprocessing.filtering import apply_bandpass_filter, design_butterworth_bandpass
from earthquake_detection.preprocessing.normalization import normalize_waveform
from earthquake_detection.preprocessing.displacement import compute_displacement_and_magnitude
from earthquake_detection.preprocessing.windowing import segment_waveform


def test_bandpass_filtering_nyquist_validation():
    fs = 100.0  # Nyquist = 50.0 Hz
    # Valid filter design
    sos = design_butterworth_bandpass(lowcut=0.5, highcut=10.0, sampling_rate=fs, order=4)
    assert sos is not None

    # Invalid highcut exceeding Nyquist
    with pytest.raises(ValueError):
        design_butterworth_bandpass(lowcut=0.5, highcut=55.0, sampling_rate=fs)

    # Invalid inverted cutoffs (lowcut >= highcut)
    with pytest.raises(ValueError):
        design_butterworth_bandpass(lowcut=10.0, highcut=5.0, sampling_rate=fs)


def test_apply_bandpass_filter(sample_waveform):
    filtered = apply_bandpass_filter(
        data=sample_waveform,
        lowcut=0.5,
        highcut=10.0,
        sampling_rate=100.0,
        zero_phase=True,
    )
    assert filtered.shape == sample_waveform.shape
    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))


def test_normalization_methods(sample_waveform):
    # Per-channel standardization
    norm_ch = normalize_waveform(sample_waveform, method="per_channel", channel_axis=0)
    assert norm_ch.shape == sample_waveform.shape
    for c in range(norm_ch.shape[0]):
        assert np.isclose(np.mean(norm_ch[c]), 0.0, atol=1e-4)
        assert np.isclose(np.std(norm_ch[c]), 1.0, atol=1e-4)

    # Peak normalization
    norm_peak = normalize_waveform(sample_waveform, method="peak", channel_axis=0)
    assert np.all(norm_peak >= -1.0 - 1e-5)
    assert np.all(norm_peak <= 1.0 + 1e-5)

    # Robust normalization
    norm_rob = normalize_waveform(sample_waveform, method="robust", channel_axis=0)
    assert norm_rob.shape == sample_waveform.shape
    assert np.all(np.isfinite(norm_rob))


def test_displacement_processing(sample_waveform):
    vel, disp, disp_mag = compute_displacement_and_magnitude(
        acceleration_3c=sample_waveform,
        sampling_rate=100.0,
        highpass_cutoff=0.1,
        channel_axis=0,
    )
    assert vel.shape == sample_waveform.shape
    assert disp.shape == sample_waveform.shape
    assert disp_mag.shape == (sample_waveform.shape[1],)
    assert np.all(disp_mag >= 0.0)  # Magnitude is non-negative
    assert np.all(np.isfinite(disp_mag))


def test_sliding_window_segmentation(sample_waveform):
    segments = segment_waveform(
        waveform=sample_waveform,
        window_length=400,
        step_size=200,
        global_p_arrival=250,
        channel_axis=0,
    )
    assert len(segments) > 0
    # First window covers [0, 400], arrival 250 falls in it with local arrival 250
    seg0 = segments[0]
    assert seg0.global_start == 0
    assert seg0.global_end == 400
    assert seg0.local_p_arrival == 250

    # Second window covers [200, 600], arrival 250 falls in it with local arrival 50
    seg1 = segments[1]
    assert seg1.global_start == 200
    assert seg1.global_end == 600
    assert seg1.local_p_arrival == 50

    # Third window covers [400, 800], arrival 250 does not fall in it
    seg2 = segments[2]
    assert seg2.local_p_arrival is None
