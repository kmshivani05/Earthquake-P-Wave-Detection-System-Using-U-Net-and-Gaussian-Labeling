# Seismological & Machine Learning Methodology

## 1. Seismic Phase Arrival Detection

The initial identification of the primary (P) seismic wave arrival is critical for earthquake early warning (EEW). Because P-waves propagate longitudinally at velocities typically between 5 and 8 km/s, their arrival at a recording station precedes destructive transverse S-waves and surface waves.

Traditional manual picking or energy-ratio heuristics (such as STA/LTA or Baer-Ulrich filters) suffer from high sensitivity to background cultural noise, emergent wave onsets, and varying signal-to-noise ratios. This project frames phase picking as a continuous temporal regression task.

---

## 2. Continuous Gaussian Target Distribution

Instead of treating arrival picking as a hard binary point annotation:
$$Y_{\text{binary}}(t) = \begin{cases} 1 & \text{if } t = t_p \\ 0 & \text{otherwise} \end{cases}$$
which imposes extreme class imbalance (a single 1 in 6,000 samples) and severely penalizes predictions off by just 1 sample, we model arrival uncertainty as a continuous Gaussian probability density:

$$Y(t) = \exp\left( -\frac{(t - t_p)^2}{2\sigma^2} \right)$$

Where:
- $t \in [0, T-1]$ is the temporal sample index.
- $t_p$ is the ground-truth annotated arrival index.
- $\sigma$ is the standard deviation in samples (e.g., $\sigma = 20$ samples $\approx 0.20$ seconds at 100 Hz).

### Advantages:
1. **Physical Uncertainty Modeling**: In real seismology, phase picking has inherent human analyst or instrument uncertainty ($\pm 0.05$ to $0.20$ s).
2. **Smooth Gradient Landscape**: Backpropagation receives non-zero gradient signals when predicted peaks approach the true arrival.
3. **Calibrated Confidence**: The peak height and width directly reflect model confidence and temporal certainty.

---

## 3. Physics-Informed Signal Preprocessing

### A. Butterworth Bandpass Filtering
Seismic signals of local and regional earthquakes exhibit dominant energy in the 0.5–10 Hz band. Ambient microseisms (e.g. oceanic waves) predominate below 0.5 Hz, while anthropogenic cultural noise dominates above 15–20 Hz.
- A 4th-order Butterworth bandpass filter (0.5–10.0 Hz) is designed using Second-Order Sections (SOS) for numerical stability.
- **Zero-Phase Filtering (`sosfiltfilt`)**: Non-causal forward-backward filtering removes all phase distortion, ensuring the onset time $t_p$ is not artificially delayed. This is appropriate for retrospective benchmark evaluation.
- **Causal Filtering (`sosfilt`)**: Single-pass forward filtering for real-time simulation without using future samples.

### B. Per-Channel Standardization
To remove station amplification differences and amplitude variations across three components:
$$x_{\text{norm}}(t) = \frac{x(t) - \mu_c}{\sigma_c + \epsilon}$$
Computed independently per trace to eliminate information leakage across dataset splits.

---

## 4. Phase 2: Bayesian Magnitude Estimation

Following detection of $t_p$, the initial 3-second P-wave window is integrated to derive peak displacement $P_d$ and velocity energy $E_p$:
$$P_d = \max_{t \in [t_p, t_p + 3.0\text{s}]} \sqrt{d_x(t)^2 + d_y(t)^2 + d_z(t)^2}$$
$$E_p = \int_{t_p}^{t_p + 3.0\text{s}} \left( v_x(t)^2 + v_y(t)^2 + v_z(t)^2 \right) dt$$

Using empirical seismological scaling:
$$\log_{10}(P_d) = \alpha M + \beta \log_{10}(R) + \gamma + \epsilon$$

Posterior magnitude distributions are sampled using an ensemble of 32 MCMC walkers under a Gutenberg-Richter prior $P(M) \propto 10^{-b M}$, providing full Bayesian credible intervals rather than overconfident point estimates.
