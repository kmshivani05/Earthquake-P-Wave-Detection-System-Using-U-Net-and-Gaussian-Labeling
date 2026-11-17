"""
PyTorch Dataset for HDF5 seismic waveform archives.

Provides lazy, memory-efficient record streaming, on-the-fly bandpass filtering,
per-record normalization, and continuous Gaussian target generation.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from earthquake_detection.data.gaussian_labels import generate_gaussian_target
from earthquake_detection.preprocessing.filtering import apply_bandpass_filter
from earthquake_detection.preprocessing.normalization import normalize_waveform
from earthquake_detection.preprocessing.displacement import compute_displacement_and_magnitude


class SeismicHDF5Dataset(Dataset):
    """Memory-efficient PyTorch Dataset for HDF5 seismic archives."""

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        indices: Optional[List[int]] = None,
        traces_key: str = "traces",
        p_arrival_key: str = "p_arrival",
        s_arrival_key: str = "s_arrival",
        metadata_key: str = "metadata",
        sampling_rate: float = 100.0,
        gaussian_sigma: float = 20.0,
        apply_filter: bool = True,
        lowcut: float = 0.5,
        highcut: float = 10.0,
        filter_order: int = 4,
        zero_phase: bool = True,
        normalization: str = "per_channel",
        compute_displacement: bool = False,
    ):
        """Initialize dataset.

        Args:
            hdf5_path: Path to HDF5 file.
            indices: Optional list of record indices to include (for train/val/test splits).
            traces_key: Key for 3-channel waveforms.
            p_arrival_key: Key for P-arrival indices.
            s_arrival_key: Key for S-arrival indices.
            metadata_key: Key for record metadata.
            sampling_rate: Waveform sampling rate in Hz.
            gaussian_sigma: Standard deviation in samples for Gaussian target.
            apply_filter: Whether to apply Butterworth bandpass filter.
            lowcut: Bandpass low cut in Hz.
            highcut: Bandpass high cut in Hz.
            filter_order: Butterworth filter order.
            zero_phase: True for forward-backward non-causal filtering; False for causal.
            normalization: 'per_channel', 'per_record', 'robust', 'peak', or 'none'.
            compute_displacement: If True, derives displacement magnitude as extra channel.
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.hdf5_path}")

        self.traces_key = traces_key
        self.p_arrival_key = p_arrival_key
        self.s_arrival_key = s_arrival_key
        self.metadata_key = metadata_key

        self.sampling_rate = sampling_rate
        self.gaussian_sigma = gaussian_sigma
        self.apply_filter = apply_filter
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.zero_phase = zero_phase
        self.normalization = normalization
        self.compute_displacement = compute_displacement

        # Inspect length and shapes without keeping file handle open across processes
        with h5py.File(self.hdf5_path, "r") as f:
            total_records = len(f[self.traces_key])
            shape = f[self.traces_key].shape
            self.total_samples = shape[1]
            self.num_channels = shape[2]

        if indices is not None:
            self.indices = [i for i in indices if 0 <= i < total_records]
        else:
            self.indices = list(range(total_records))

        self._h5_file: Optional[h5py.File] = None

    def _get_h5_handle(self) -> h5py.File:
        """Worker-safe lazy HDF5 handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch and preprocess a single seismic record.

        Returns:
            Dictionary containing:
                - 'waveform': torch.FloatTensor of shape (3, T) or (4, T) if displacement enabled
                - 'target': torch.FloatTensor of shape (1, T)
                - 'p_arrival': int arrival index
                - 's_arrival': int arrival index
                - 'metadata': str identifier
                - 'record_index': int original archive index
        """
        record_idx = self.indices[idx]
        h5 = self._get_h5_handle()

        # Read trace: expected raw shape (T, 3)
        raw_trace = np.array(h5[self.traces_key][record_idx], dtype=np.float32)

        # Transpose from (T, 3) to (3, T) for standard PyTorch 1D convolutions
        if raw_trace.ndim == 2 and raw_trace.shape[1] == self.num_channels:
            trace = raw_trace.T.copy()  # shape (3, T)
        else:
            trace = raw_trace.copy()

        # Read phase arrivals
        p_arrival = int(h5[self.p_arrival_key][record_idx]) if self.p_arrival_key in h5 else -1
        s_arrival = int(h5[self.s_arrival_key][record_idx]) if self.s_arrival_key in h5 else -1

        # Read metadata safely
        meta_val = ""
        if self.metadata_key in h5:
            raw_meta = h5[self.metadata_key][record_idx]
            if isinstance(raw_meta, (bytes, bytearray)):
                meta_val = raw_meta.decode("utf-8", errors="replace")
            else:
                meta_val = str(raw_meta)

        # 1. Bandpass Filtering
        if self.apply_filter and self.sampling_rate is not None:
            trace = apply_bandpass_filter(
                data=trace,
                lowcut=self.lowcut,
                highcut=self.highcut,
                sampling_rate=self.sampling_rate,
                order=self.filter_order,
                zero_phase=self.zero_phase,
                axis=-1,
            )

        # 2. Normalization
        trace = normalize_waveform(
            data=trace,
            method=self.normalization,
            channel_axis=0,
        )

        # 3. Optional Displacement Magnitude Feature
        if self.compute_displacement and self.sampling_rate is not None:
            _, _, disp_mag = compute_displacement_and_magnitude(
                trace,
                sampling_rate=self.sampling_rate,
                channel_axis=0,
            )
            # Add displacement magnitude as 4th channel
            disp_mag = normalize_waveform(disp_mag[np.newaxis, :], method="robust", channel_axis=0)
            trace = np.concatenate([trace, disp_mag], axis=0)

        # 4. Continuous Gaussian Target Generation
        T = trace.shape[-1]
        gaussian_target = generate_gaussian_target(
            waveform_length=T,
            arrival_index=p_arrival if (0 <= p_arrival < T) else None,
            sigma=self.gaussian_sigma,
            normalize_peak=True,
            allow_missing=True,
            dtype=np.float32,
        )

        return {
            "waveform": torch.from_numpy(trace).float(),
            "target": torch.from_numpy(gaussian_target).unsqueeze(0).float(),
            "p_arrival": p_arrival,
            "s_arrival": s_arrival,
            "metadata": meta_val,
            "record_index": record_idx,
        }

    def close(self) -> None:
        """Safely close open HDF5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()


def create_dataset_splits(
    hdf5_path: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_records: Optional[int] = None,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[SeismicHDF5Dataset, SeismicHDF5Dataset, SeismicHDF5Dataset]:
    """Create reproducible train, validation, and test dataset splits.

    Args:
        hdf5_path: Path to HDF5 file.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        num_records: Optional cap on total records used (for fast development).
        seed: Random seed for shuffle reproducibility.
        **dataset_kwargs: Arguments forwarded to SeismicHDF5Dataset.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    with h5py.File(hdf5_path, "r") as f:
        traces_key = dataset_kwargs.get("traces_key", "traces")
        total_available = len(f[traces_key])

    n_use = total_available if num_records is None else min(total_available, num_records)
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(total_available)[:n_use].tolist()

    n_train = int(n_use * train_ratio)
    n_val = int(n_use * val_ratio)

    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train: n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val:]

    train_ds = SeismicHDF5Dataset(hdf5_path, indices=train_idx, **dataset_kwargs)
    val_ds = SeismicHDF5Dataset(hdf5_path, indices=val_idx, **dataset_kwargs)
    test_ds = SeismicHDF5Dataset(hdf5_path, indices=test_idx, **dataset_kwargs)

    return train_ds, val_ds, test_ds
