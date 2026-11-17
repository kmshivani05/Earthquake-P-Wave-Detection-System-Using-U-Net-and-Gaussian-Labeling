"""
Noise robustness experiments and SNR degradation benchmarks.

Injects calibrated levels of Gaussian white noise, ambient seismic colored noise,
and anthropogenic cultural burst noise to evaluate phase picking degradation.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal
import torch

from earthquake_detection.evaluation.metrics import EvaluationMetrics, calculate_metrics
from earthquake_detection.inference.predict import SeismicPredictor

logger = logging.getLogger(__name__)


def add_snr_noise(
    waveform: np.ndarray,
    snr_db: Optional[float],
    noise_type: str = "gaussian",
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Add controlled noise to waveform matching target Signal-to-Noise Ratio (SNR) in dB.

    Args:
        waveform: Waveform array of shape (C, T) or (T, C).
        snr_db: Target SNR in decibels. If None or inf, returns unmodified copy.
        noise_type: 'gaussian', 'pink' (ambient), or 'burst' (anthropogenic).
        rng: Optional seeded numpy random state.

    Returns:
        Noisy waveform of identical shape and float32 dtype.
    """
    if snr_db is None or np.isinf(snr_db):
        return waveform.copy().astype(np.float32)

    if rng is None:
        rng = np.random.RandomState(42)

    arr = np.array(waveform, dtype=np.float32, copy=True)
    signal_power = np.mean(arr ** 2)

    if signal_power <= 0:
        return arr

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    if noise_type == "gaussian":
        raw_noise = rng.randn(*arr.shape).astype(np.float32)

    elif noise_type == "pink":
        # 1/f ambient seismic noise approximation
        raw_noise = rng.randn(*arr.shape).astype(np.float32)
        # Low-pass filter to simulate ambient seismic background hum
        b, a = signal.butter(2, 0.2, btype="lowpass")
        raw_noise = signal.filtfilt(b, a, raw_noise, axis=-1).astype(np.float32)

    elif noise_type == "burst":
        # Anthropogenic cultural spikes
        raw_noise = rng.randn(*arr.shape).astype(np.float32)
        spikes = (rng.rand(*arr.shape) > 0.98).astype(np.float32) * (rng.randn(*arr.shape) * 3.0)
        raw_noise = raw_noise + spikes.astype(np.float32)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    curr_noise_power = np.mean(raw_noise ** 2)
    if curr_noise_power > 0:
        scaling = np.sqrt(target_noise_power / curr_noise_power)
        scaled_noise = raw_noise * scaling
    else:
        scaled_noise = raw_noise

    noisy = arr + scaled_noise
    return noisy.astype(np.float32)


def run_noise_benchmark(
    predictor: SeismicPredictor,
    test_waveforms: List[np.ndarray],
    ground_truth_arrivals: List[int],
    snr_levels_db: Optional[List[Optional[float]]] = None,
    noise_types: Optional[List[str]] = None,
    tolerance_samples: int = 20,
    sampling_rate: Optional[float] = 100.0,
    seed: int = 42,
) -> Dict[str, Dict[str, EvaluationMetrics]]:
    """Run comprehensive noise robustness evaluation across SNRs and noise archetypes.

    Args:
        predictor: Initialized SeismicPredictor.
        test_waveforms: List of raw test waveforms.
        ground_truth_arrivals: Ground truth arrival sample indices.
        snr_levels_db: List of SNR values to benchmark (e.g. [None, 20.0, 10.0, 0.0, -5.0]).
        noise_types: List of noise types ('gaussian', 'pink', 'burst').
        tolerance_samples: Timing tolerance window.
        sampling_rate: Sampling frequency in Hz.
        seed: Random seed for noise reproducibility.

    Returns:
        Nested dict: {noise_type: {snr_label: EvaluationMetrics}}
    """
    levels = snr_levels_db if snr_levels_db is not None else [None, 20.0, 10.0, 0.0, -5.0]
    types = noise_types if noise_types is not None else ["gaussian", "pink"]

    results: Dict[str, Dict[str, EvaluationMetrics]] = {}
    rng = np.random.RandomState(seed)

    for n_type in types:
        results[n_type] = {}
        for snr in levels:
            snr_str = "clean" if snr is None else f"{int(snr)}dB"
            logger.info(f"Evaluating Noise Type: {n_type} | SNR: {snr_str}")

            predicted_arrivals: List[Optional[int]] = []
            for wf in test_waveforms:
                noisy_wf = add_snr_noise(wf, snr_db=snr, noise_type=n_type, rng=rng)
                det_res, _ = predictor.predict(noisy_wf)
                predicted_arrivals.append(det_res.arrival_index)

            metrics = calculate_metrics(
                ground_truth_arrivals=ground_truth_arrivals,
                predicted_arrivals=predicted_arrivals,
                timing_tolerance_samples=tolerance_samples,
                sampling_rate_hz=sampling_rate,
            )
            results[n_type][snr_str] = metrics

    return results
