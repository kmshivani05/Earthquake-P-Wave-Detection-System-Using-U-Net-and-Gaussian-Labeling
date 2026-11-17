"""
Pytest configuration, fixtures, and synthetic seismic HDF5 generator.
"""

from pathlib import Path
import tempfile
from typing import Generator
import h5py
import numpy as np
import pytest
import torch


@pytest.fixture
def synthetic_hdf5_path() -> Generator[Path, None, None]:
    """Create a temporary synthetic seismic HDF5 archive with known wave packets."""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    N = 20           # 20 synthetic records
    T = 1000         # 1000 temporal samples
    C = 3            # 3 channels (Z, N, E)
    fs = 100.0       # 100 Hz

    traces = np.zeros((N, T, C), dtype=np.float32)
    p_arrivals = np.zeros(N, dtype=np.int64)
    s_arrivals = np.zeros(N, dtype=np.int64)
    metadata = np.empty(N, dtype=object)

    t = np.arange(T) / fs
    rng = np.random.RandomState(42)

    for i in range(N):
        # Baseline noise
        traces[i] = rng.randn(T, C) * 0.05

        # P-wave packet (onset around sample 200 - 300)
        p_idx = rng.randint(200, 350)
        s_idx = p_idx + rng.randint(150, 300)

        p_arrivals[i] = p_idx
        s_arrivals[i] = s_idx
        metadata[i] = f"SYNTH_STA_{i:03d}_EV"

        # Synthetic P-wave Ricker-like wavelet on Z channel
        p_t = t - (p_idx / fs)
        p_wave = (1.0 - 2.0 * (np.pi * 5.0 * p_t) ** 2) * np.exp(-((np.pi * 5.0 * p_t) ** 2))
        traces[i, :, 0] += p_wave.astype(np.float32) * 2.0

        # Synthetic S-wave packet on horizontal channels N, E
        s_t = t - (s_idx / fs)
        s_wave = np.sin(2 * np.pi * 3.0 * s_t) * np.exp(-((s_t / 0.5) ** 2))
        traces[i, :, 1] += s_wave.astype(np.float32) * 3.0
        traces[i, :, 2] += s_wave.astype(np.float32) * 2.5

    with h5py.File(tmp_path, "w") as f:
        f.attrs["Author"] = "Synthetic Test Suite"
        f.create_dataset("traces", data=traces)
        f.create_dataset("p_arrival", data=p_arrivals)
        f.create_dataset("s_arrival", data=s_arrivals)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("metadata", data=metadata, dtype=dt)

    yield tmp_path

    # Cleanup temporary file
    if tmp_path.is_file():
        tmp_path.unlink()


@pytest.fixture
def sample_waveform() -> np.ndarray:
    """Single 3-channel synthetic seismic waveform of shape (3, 1000)."""
    rng = np.random.RandomState(42)
    wf = rng.randn(3, 1000).astype(np.float32)
    # Inject synthetic arrival pulse at index 250
    t = np.arange(1000)
    pulse = np.exp(-((t - 250) ** 2) / (2 * 15 ** 2)) * 5.0
    wf[0] += pulse
    return wf
