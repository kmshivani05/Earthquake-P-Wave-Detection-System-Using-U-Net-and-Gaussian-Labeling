# Earthquake P-Wave Detection System Using U-Net++ and Gaussian Labeling

A physics-informed deep learning framework for seismic P-wave arrival detection using 1D U-Net++, continuous Gaussian-distribution labeling, multi-scale dilated convolutions, and bottleneck multi-head self-attention.

## Project Overview

This project implements an advanced deep learning framework for detecting primary seismic (P-wave) arrivals in three-component earthquake waveforms.

The system addresses limitations of traditional STA/LTA-based detection methods by combining deep learning, continuous Gaussian target generation, and temporal feature extraction.

## Key Features

- 1D U-Net++ architecture with dense nested skip connections.
- Continuous Gaussian labeling for P-wave arrival targets.
- Multi-scale dilated convolutions.
- Multi-Head Self-Attention (MHSA).
- Three-component seismic waveform processing.
- Peak prominence and threshold-based arrival decoding.
- Model ablation experiments (Models A–D).

## System Architecture

The processing pipeline consists of:

1. Raw three-channel seismic waveform input.
2. Butterworth bandpass filtering (0.5–10 Hz).
3. Per-channel standardization.
4. 1D U-Net++ feature extraction.
5. Multi-scale dilated convolution and attention.
6. Continuous temporal score prediction.
7. Peak detection and P-wave arrival estimation.

## Mathematical Formulation

### Gaussian Target Generation

The continuous Gaussian target is defined as:

\[
Y(t)=\exp\left(-\frac{(t-t_p)^2}{2\sigma^2}\right)
\]

Where:
- \(t\) is the temporal sample index.
- \(t_p\) is the ground-truth P-wave arrival.
- \(\sigma\) controls the Gaussian width.

### U-Net++ Architecture

The network uses dense nested skip connections to combine encoder and decoder features at multiple resolutions.

### Multi-Scale Dilated Convolutions

Dilation rates:

\[
d \in \{1,2,4,8\}
\]

### Multi-Head Self-Attention

Attention is applied at the bottleneck to capture long-range temporal dependencies.

## Model Configurations

| Model | Architecture | Dilated Convolutions | Attention |
|---|---|---|---|
| Model A | Baseline U-Net++ | No | No |
| Model B | Dilated U-Net++ | Yes | No |
| Model C | Attention U-Net++ | No | Yes |
| Model D | Enhanced U-Net++ | Yes | Yes |

## Dataset Structure

The project supports HDF5 seismic datasets with the following structure:

```text
/
├── metadata
├── p_arrival
├── s_arrival
└── traces
```

The traces contain three-component seismic waveforms.

**Note:** The local STEAD/test.hdf5 archive is reserved for testing, validation, and benchmarking, not training.

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- h5py
- Matplotlib
- PyYAML

### Setup

```bash
git clone https://github.com/kmshivani05/Earthquake-P-Wave-Detection-System-Using-U-Net-and-Gaussian-Labeling-.git

cd Earthquake-P-Wave-Detection-System-Using-U-Net-and-Gaussian-Labeling-

python -m venv venv
```

Activate the environment:

**Windows:**
```powershell
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Dataset Inspection

```bash
python scripts/inspect_dataset.py --config configs/config.yaml
```

### Waveform Visualization

```bash
python scripts/visualize_waveform.py --record-index 0
```

### Model Training

```bash
python scripts/train_model.py --config configs/baseline.yaml
```

For the enhanced architecture:

```bash
python scripts/train_model.py --config configs/enhanced.yaml --epochs 50 --batch-size 16
```

### P-Wave Prediction

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --record-index 0
```

### Model Evaluation

```bash
python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt --num-records 100
```

### Testing

```bash
pytest tests
```

## Research Notebooks

The `notebooks/` directory contains interactive notebooks for:

1. Dataset auditing.
2. Waveform analysis.
3. Signal preprocessing.
4. Gaussian labeling.
5. Model experiments.
6. Error analysis.

## Phase 2: Bayesian Magnitude Estimation

An optional research extension explores Bayesian earthquake magnitude estimation using P-wave displacement, cumulative velocity energy, and Markov Chain Monte Carlo sampling.

## Scientific Limitations

This project is an independent research implementation and is not a certified operational earthquake early warning system.

- Zero-phase filtering uses future samples and is intended for offline analysis.
- Single-station measurements cannot independently determine complete fault geometry.
- Double integration of acceleration signals can introduce low-frequency drift.

## References

- Zhou et al. (2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
- Mousavi et al. (2019), STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI.
- Research framework DOI: [10.21203/rs.3.rs-7479899/v1](https://doi.org/10.21203/rs.3.rs-7479899/v1).

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
