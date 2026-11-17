"""
Unit tests for SeismicHDF5Dataset and dataset splitting.
"""

from pathlib import Path
import pytest
import torch

from earthquake_detection.data.dataset import SeismicHDF5Dataset, create_dataset_splits


def test_dataset_item_structure(synthetic_hdf5_path):
    dataset = SeismicHDF5Dataset(
        hdf5_path=synthetic_hdf5_path,
        sampling_rate=100.0,
        gaussian_sigma=15.0,
        apply_filter=True,
        normalization="per_channel",
    )
    assert len(dataset) == 20

    item = dataset[0]
    assert "waveform" in item
    assert "target" in item
    assert "p_arrival" in item
    assert "s_arrival" in item
    assert "metadata" in item
    assert "record_index" in item

    wf = item["waveform"]
    tg = item["target"]
    assert isinstance(wf, torch.Tensor)
    assert isinstance(tg, torch.Tensor)
    assert wf.shape == (3, 1000)
    assert tg.shape == (1, 1000)
    assert item["p_arrival"] >= 0

    dataset.close()


def test_dataset_splits(synthetic_hdf5_path):
    train_ds, val_ds, test_ds = create_dataset_splits(
        hdf5_path=synthetic_hdf5_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 20

    # Ensure no index overlap among splits
    train_idx = set(train_ds.indices)
    val_idx = set(val_ds.indices)
    test_idx = set(test_ds.indices)

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0
