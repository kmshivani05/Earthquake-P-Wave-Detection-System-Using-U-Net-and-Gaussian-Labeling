# Experimental Protocol & Ablation Framework

## 1. Architecture Ablation Matrix

To isolate the contributions of each architectural component, this repository establishes four standardized model configurations:

| Model ID | Architecture | Dilated Convolutions | Attention (MHSA) | Deep Supervision | Config File |
|---|---|---|---|---|---|
| **Model A** | Baseline 1D U-Net++ | False | False | False | `configs/baseline.yaml` |
| **Model B** | Dilated 1D U-Net++ | True (`[1, 2, 4, 8]`) | False | False | `configs/dilated.yaml` |
| **Model C** | Attention 1D U-Net++ | False | True (4 heads) | False | `configs/attention.yaml` |
| **Model D** | Proposed Enhanced Architecture | True (`[1, 2, 4, 8]`) | True (4 heads) | True | `configs/enhanced.yaml` |

---

## 2. Evaluation Metrics Protocol

The evaluation protocol measures picking accuracy within calibrated tolerances:
- **True Positive (TP)**: Detection $\hat{t}_p$ with $|\hat{t}_p - t_p| \le \delta$, where $\delta = 20\text{ samples} = 0.20\text{ s}$ at 100 Hz.
- **False Positive (FP)**: Detection made but $|\hat{t}_p - t_p| > \delta$, or detection made on background noise without earthquake.
- **False Negative (FN)**: Ground-truth arrival exists, but no detection is made above the calibrated confidence threshold ($\tau = 0.40$), or detection was made outside tolerance.

Primary metrics:
1. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
2. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
3. **F1-Score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
4. **Mean Absolute Error (MAE)**: Calculated exclusively over True Positives in samples and milliseconds:
   $$\text{MAE} = \frac{1}{|\text{TP}|} \sum_{i \in \text{TP}} |\hat{t}_{p, i} - t_{p, i}|$$

---

## 3. Noise Robustness Benchmarking

Models are evaluated across varying Signal-to-Noise Ratios (SNRs) by injecting synthetic noise into test waveforms:
- **SNR Levels**: Clean ($\infty$ dB), $+20\text{ dB}$, $+10\text{ dB}$, $0\text{ dB}$, $-5\text{ dB}$.
- **Noise Archetypes**:
  1. *Gaussian White Noise*: Uncorrelated thermal sensor noise.
  2. *Ambient Seismic Noise*: Colored $1/f$ noise approximating microseisms.
  3. *Anthropogenic Cultural Noise*: Sporadic non-stationary burst spikes.
