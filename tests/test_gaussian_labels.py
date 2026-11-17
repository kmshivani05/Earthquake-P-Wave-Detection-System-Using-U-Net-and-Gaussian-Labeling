"""
Unit tests for Gaussian target label generation.
"""

import numpy as np
import pytest
import torch

from earthquake_detection.data.gaussian_labels import (
    generate_gaussian_target,
    generate_gaussian_target_torch,
)


def test_gaussian_target_peak_and_length():
    T = 6000
    t_p = 900
    sigma = 20.0
    target = generate_gaussian_target(T, t_p, sigma=sigma)

    assert isinstance(target, np.ndarray)
    assert target.shape == (T,)
    assert target.dtype == np.float32
    # Peak must be strictly 1.0 at arrival sample
    assert np.isclose(target[t_p], 1.0)
    # Peak index should be argmax
    assert np.argmax(target) == t_p
    # Bounded strictly within [0, 1]
    assert np.all(target >= 0.0)
    assert np.all(target <= 1.0)


def test_gaussian_sigma_broadening():
    T = 1000
    t_p = 500
    target_10 = generate_gaussian_target(T, t_p, sigma=10.0)
    target_40 = generate_gaussian_target(T, t_p, sigma=40.0)

    # Broader sigma should have higher integral and higher values away from peak
    offset = 20
    assert target_40[t_p + offset] > target_10[t_p + offset]
    assert np.sum(target_40) > np.sum(target_10)


def test_missing_and_invalid_labels():
    T = 1000

    # None arrival -> all zeros
    target_none = generate_gaussian_target(T, None, allow_missing=True)
    assert np.all(target_none == 0.0)
    assert target_none.shape == (T,)

    # Negative arrival -> all zeros
    target_neg = generate_gaussian_target(T, -1, allow_missing=True)
    assert np.all(target_neg == 0.0)

    # Disallowing missing should raise ValueError
    with pytest.raises(ValueError):
        generate_gaussian_target(T, None, allow_missing=False)

    with pytest.raises(ValueError):
        generate_gaussian_target(T, -5, allow_missing=False)

    # Out of bounds arrival
    with pytest.raises(ValueError):
        generate_gaussian_target(T, 1500, allow_missing=False)


def test_boundary_arrivals():
    T = 500
    # Left boundary (t=0)
    t_start = generate_gaussian_target(T, 0, sigma=15.0)
    assert np.isclose(t_start[0], 1.0)
    assert np.argmax(t_start) == 0

    # Right boundary (t=T-1)
    t_end = generate_gaussian_target(T, T - 1, sigma=15.0)
    assert np.isclose(t_end[T - 1], 1.0)
    assert np.argmax(t_end) == T - 1


def test_gaussian_target_torch():
    T = 1000
    t_p = 300
    tensor = generate_gaussian_target_torch(T, t_p, sigma=25.0)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, T)
    assert torch.isclose(tensor[0, t_p], torch.tensor(1.0))
    assert torch.argmax(tensor[0]).item() == t_p
