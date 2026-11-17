# Scientific Limitations & Research Notice

## 1. Safety & Operational Notice
> [!CAUTION]
> **This software is an experimental scientific research implementation and is NOT a certified or operational earthquake early warning (EEW) system.**
> It must not be used for life-safety-critical automation, civil protection alerts, or operational emergency response infrastructure without rigorous independent seismological validation, hardware certification, and agency approval.

---

## 2. Dataset Assumptions & Provenance
- The local archive `STEAD/test.hdf5` is a held-out test split curated by external researchers.
- Although standard STEAD records are 60-second traces sampled at 100 Hz, local datasets should always be verified using `scripts/inspect_dataset.py` before inferring physical time units.
- Under rigorous scientific practice, held-out test splits must never be used for model training.

---

## 3. Signal Preprocessing & Causality
- **Zero-Phase Filtering**: When `zero_phase=True` (`scipy.signal.sosfiltfilt`), filtering operates bidirectionally (forward and backward). While this completely eliminates phase distortion and preserves exact arrival times for offline benchmark evaluation, it utilizes future samples and is strictly non-causal. For real-time streaming simulations, set `zero_phase=False` (`scipy.signal.sosfilt`).
- **Numerical Double Integration**: Integrating acceleration records to estimate displacement amplifies low-frequency sensor baseline errors and Brownian-like drift. While high-pass filtering attenuates drift, displacement features remain sensitive to baseline offsets.

---

## 4. Single-Station Observation Limits
- A single station cannot definitively resolve earthquake hypocentral depth, fault mechanism, or rupture directivity without multi-station network cross-correlation.
- Magnitude estimation from initial 3-second P-wave windows requires calibrated regional empirical attenuation coefficients $(\alpha, \beta, \gamma)$. Using generic parameters yields relative rather than absolute moment magnitudes.
