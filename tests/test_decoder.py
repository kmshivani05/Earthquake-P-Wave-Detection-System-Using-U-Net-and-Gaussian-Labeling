"""
Unit tests for arrival decoding and peak extraction.
"""

import numpy as np
import pytest
import torch

from earthquake_detection.inference.decoder import decode_p_arrival, reconcile_windowed_predictions


def test_decode_prominent_arrival():
    T = 1000
    scores = np.zeros(T, dtype=np.float32)
    # Put a prominent peak at sample 450
    t = np.arange(T)
    scores = np.exp(-((t - 450) ** 2) / (2 * 10 ** 2)).astype(np.float32)

    res = decode_p_arrival(scores, threshold=0.5, prominence=0.2)
    assert res.has_detection is True
    assert res.arrival_index == 450
    assert np.isclose(res.confidence, 1.0)


def test_decode_low_confidence_no_detection():
    T = 1000
    # Background noise with max below threshold 0.5
    rng = np.random.RandomState(42)
    scores = rng.uniform(0.01, 0.25, size=T).astype(np.float32)

    res = decode_p_arrival(scores, threshold=0.5, prominence=0.2)
    assert res.has_detection is False
    assert res.arrival_index is None


def test_reconcile_windowed_predictions():
    total_len = 1000
    # Window 1: [0, 600], peak at 200 (val 0.8)
    w1 = np.zeros(600, dtype=np.float32)
    w1[200] = 0.8

    # Window 2: [400, 1000], peak at 500 global -> 100 local (val 0.9)
    w2 = np.zeros(600, dtype=np.float32)
    w2[100] = 0.9

    reconciled_max = reconcile_windowed_predictions(
        window_scores=[w1, w2],
        global_starts=[0, 400],
        total_length=total_len,
        aggregation="max",
    )
    assert reconciled_max.shape == (total_len,)
    assert np.isclose(reconciled_max[200], 0.8)
    assert np.isclose(reconciled_max[500], 0.9)
