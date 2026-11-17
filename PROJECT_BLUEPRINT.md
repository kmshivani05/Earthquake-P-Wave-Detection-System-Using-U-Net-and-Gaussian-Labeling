# Project Blueprint: Earthquake P-Wave Detection System

## 1. System Architecture Pipeline

```text
                SEISMIC DATASET (HDF5)
                          |
                          v
                   DATASET AUDIT
                 (dataset_audit.py)
                          |
                          v
                DATA LOADER & SPLITTING
                      (dataset.py)
                          |
                          v
                SIGNAL PREPROCESSING
         +----------------+----------------+
         |                |                |
         v                v                v
     Filtering      Normalization     Optional Disp.
    (filtering.py) (normalization.py)(displacement.py)
         |                |                |
         +----------------+----------------+
                          |
                          v
                  GAUSSIAN LABELING
                 (gaussian_labels.py)
                          |
                          v
                  MODEL INPUT TENSOR
                     (B, 3, 6000)
                          |
                          v
                  1D U-NET++ ENCODER
                 (unetplusplus_1d.py)
                          |
                          v
                 DILATED CONV BLOCKS
                  (dilated_blocks.py)
                          |
                          v
                 MHSA ATTENTION BLOCK
                    (attention.py)
                          |
                          v
                  NESTED DENSE NODES
                   X^{i,j} Pathways
                          |
                          v
                    DECODER HEAD
                          |
                          v
                  TEMPORAL PREDICTIONS
                     (B, 1, 6000)
                          |
                          v
                    PHASE DECODER
                     (decoder.py)
                          |
                          v
                 P-WAVE ARRIVAL INDEX
                          |
                          v
                 EVALUATION & METRICS
                     (metrics.py)
                          |
                          v
                    VISUALIZATION
                      (plots.py)
```

---

## 2. Directory Structure & File Map

```text
earthquake-p-wave-detection/
├── README.md                      # Comprehensive research presentation
├── LICENSE                        # MIT License
├── CITATION.cff                   # Research citation metadata
├── requirements.txt               # Python package dependencies
├── pyproject.toml                 # Package definition and pytest config
├── .gitignore                     # Git ignore rules for HDF5, checkpoints, caches
├── PROJECT_BLUEPRINT.md           # Engineering architecture specification
│
├── configs/
│   ├── config.yaml                # Default configuration
│   ├── baseline.yaml              # Model A: Baseline 1D U-Net++
│   ├── dilated.yaml               # Model B: Dilated 1D U-Net++
│   ├── attention.yaml             # Model C: Attention 1D U-Net++
│   └── enhanced.yaml              # Model D: Proposed Enhanced Architecture
│
├── data/
│   ├── README.md                  # Dataset acquisition and provenance
│   ├── raw/                       # Raw HDF5 archives (.gitkeep)
│   └── processed/                 # Processed dataset cache (.gitkeep)
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_waveform_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_gaussian_labeling.ipynb
│   ├── 05_model_experiments.ipynb
│   └── 06_error_analysis.ipynb
│
├── src/earthquake_detection/
│   ├── __init__.py
│   ├── config.py                  # Dataclass config loading & validation
│   ├── data/
│   │   ├── dataset.py             # Lazy HDF5 PyTorch Dataset
│   │   ├── dataset_audit.py       # Comprehensive dataset audit
│   │   └── gaussian_labels.py     # Continuous Gaussian target generator
│   ├── preprocessing/
│   │   ├── filtering.py           # 4th-order Butterworth bandpass filter
│   │   ├── normalization.py       # Per-channel / per-record / robust normalization
│   │   ├── displacement.py        # Double integration & 3D vector displacement
│   │   └── windowing.py           # Sliding-window segmentation
│   ├── models/
│   │   ├── __init__.py            # Model factory
│   │   ├── unetplusplus_1d.py     # 1D U-Net++ with nested skip pathways
│   │   ├── dilated_blocks.py      # Multi-scale parallel dilated convolutions
│   │   └── attention.py           # Bottleneck Multi-Head Self-Attention
│   ├── training/
│   │   ├── losses.py              # MSE, Weighted MSE, Deep Supervision losses
│   │   ├── trainer.py             # Optimization loop, early stopping, checkpointing
│   │   └── train.py               # Main training entrypoint
│   ├── inference/
│   │   ├── decoder.py             # Prominence peak detection & arrival decoder
│   │   └── predict.py             # Single/batch inference predictor
│   ├── evaluation/
│   │   └── metrics.py             # MAE, RMSE, Precision, Recall, F1
│   ├── experiments/
│   │   └── noise_robustness.py    # SNR-based noise benchmark
│   ├── magnitude/
│   │   ├── README.md              # Phase 2 magnitude theory & equations
│   │   └── mcmc_magnitude.py      # 32-walker ensemble MCMC Bayesian sampler
│   └── visualization/
│       └── plots.py               # Scientific publication plotting routines
│
├── scripts/
│   ├── inspect_dataset.py         # CLI: dataset audit report generator
│   ├── visualize_waveform.py      # CLI: waveform & target visualizer
│   ├── train_model.py             # CLI: training runner
│   ├── predict.py                 # CLI: single-trace inference runner
│   └── evaluate_model.py          # CLI: test split evaluator
│
├── tests/
│   ├── conftest.py                # Synthetic seismic HDF5 generator fixture
│   ├── test_gaussian_labels.py    # Tests for continuous Gaussian targets
│   ├── test_preprocessing.py      # Tests for filters, normalization, displacement
│   ├── test_dataset.py            # Tests for dataset loading & splits
│   ├── test_model_shapes.py       # Tests for U-Net++ forward/backward & shapes
│   ├── test_attention.py          # Tests for MHSA module
│   ├── test_decoder.py            # Tests for peak detection & reconciliation
│   ├── test_metrics.py            # Tests for phase picking metrics & tolerances
│   └── test_training_pipeline.py  # End-to-end integration training smoke test
│
├── checkpoints/                   # Trained weights (.pt) and histories (.gitkeep)
├── results/                       # Plots, JSON reports, and evaluations (.gitkeep)
└── docs/
    ├── architecture.md            # Topology and mathematical node formulation
    ├── methodology.md             # Seismology and ML principles
    ├── experiments.md             # Ablation matrix & benchmark protocol
    └── limitations.md             # Research notice and operational caveats
```
