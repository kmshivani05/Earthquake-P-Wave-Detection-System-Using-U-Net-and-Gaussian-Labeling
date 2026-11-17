"""
Unit tests for evaluation metrics and timing error calculations.
"""

import pytest
from earthquake_detection.evaluation.metrics import calculate_metrics


def test_perfect_metrics():
    gt = [100, 250, 400, 500]
    pred = [100, 250, 400, 500]

    metrics = calculate_metrics(
        ground_truth_arrivals=gt,
        predicted_arrivals=pred,
        timing_tolerance_samples=10,
        sampling_rate_hz=100.0,
    )
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.true_positives == 4
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.mae_samples == 0.0
    assert metrics.mae_ms == 0.0


def test_timing_tolerance_and_error():
    gt = [100, 200, 300]
    # Errors: +5 (within tol 10), -8 (within tol 10), +15 (outside tol 10)
    pred = [105, 192, 315]

    metrics = calculate_metrics(
        ground_truth_arrivals=gt,
        predicted_arrivals=pred,
        timing_tolerance_samples=10,
        sampling_rate_hz=100.0,
    )
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    # MAE over True Positives: mean(5, 8) = 6.5 samples
    assert metrics.mae_samples == 6.5
    # At 100 Hz, 6.5 samples = 65.0 ms
    assert metrics.mae_ms == 65.0


def test_missed_detections_and_false_alarms():
    gt = [100, 200, None]
    pred = [102, None, 450]  # Record 0: TP, Record 1: FN (missed), Record 2: FP (noise false alarm)

    metrics = calculate_metrics(
        ground_truth_arrivals=gt,
        predicted_arrivals=pred,
        timing_tolerance_samples=10,
        sampling_rate_hz=100.0,
    )
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5
