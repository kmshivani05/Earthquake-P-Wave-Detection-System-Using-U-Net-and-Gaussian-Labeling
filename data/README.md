# Dataset Management & Provenance Guide

## 1. Overview
This project processes 3-channel temporal seismic waveforms to detect P-wave arrival times.

## 2. Expected Dataset Format
Datasets are stored in standard **HDF5** files with the following layout:

```text
/
├── metadata    (N,)      - String / object identifiers (e.g., station, event ID)
├── p_arrival   (N,)      - Integer: sample index of annotated P-wave arrival
├── s_arrival   (N,)      - Integer: sample index of annotated S-wave arrival
└── traces      (N, T, 3) - Float32: 3-component seismic waveform signals
```

Where:
- `N`: Number of seismic waveform records (e.g. 6,000).
- `T`: Number of time samples per record (e.g. 6,000 samples).
- `3`: Seismic channels (typically vertical Z, north-south N, east-west E).

## 3. Local Dataset Provenance Notice
- A local dataset is provided at `STEAD/test.hdf5`.
- Audit attributes report: `Author: Daniele Trappolini`.
- This file is treated strictly as an **evaluation / held-out test split**.
- **Important**: In rigorous research, never train on held-out test splits. Training should be conducted on verified training splits or controlled subsets.

## 4. STEAD Benchmark Acquisition
The full Stanford Earthquake Dataset (STEAD) can be downloaded from:
- Primary repository: [https://github.com/smousavi05/STEAD](https://github.com/smousavi05/STEAD)
- Paper: Mousavi et al., *IEEE Access* (2019), DOI: 10.1109/ACCESS.2019.2959182.

To acquire and extract:
1. Download `merged.hdf5` or the chunked HDF5 files from the official repository.
2. Place the training split in `data/raw/train.hdf5` and test split in `data/raw/test.hdf5`.
3. Configure the path in `configs/config.yaml` or pass `--dataset-path` on the CLI.

## 5. Synthetic Dataset Generation
For automated CI/CD testing and environments without the multi-gigabyte STEAD archive, the test suite includes a synthetic HDF5 generator (`tests/conftest.py`) that generates physically plausible seismic wave packets with known P-wave arrival times, noise, and valid shapes.
