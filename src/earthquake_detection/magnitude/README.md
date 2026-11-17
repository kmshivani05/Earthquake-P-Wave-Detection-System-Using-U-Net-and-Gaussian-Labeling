# Phase 2 Extension: Physics-Informed Probabilistic Magnitude Estimation

## 1. Scientific Context
In earthquake early warning (EEW), magnitude estimation from initial P-wave windows (typically the first 1 to 3 seconds following P-wave onset $t_p$) is an inverse problem subject to high observational uncertainty.

The reference paper:
> *"A Physics-Informed Deep Learning and Probabilistic Inference Framework for Real-Time Single-Station Earthquake Detection and Magnitude Estimation"*
> (DOI: 10.21203/rs.3.rs-7479899/v1)

proposes estimating single-station magnitude by coupling physical displacement energy evolution with Bayesian probabilistic inference and Markov Chain Monte Carlo (MCMC) sampling.

## 2. Mathematical Formulation

### A. Initial P-Wave Energy & Peak Displacement
Following detected P-wave arrival $t_p$, the seismic displacement $d(t)$ and velocity $v(t)$ are integrated over a short duration $\tau$ (e.g., $\tau = 3.0\text{ s}$):
$$P_d = \max_{t \in [t_p, t_p + \tau]} \sqrt{d_x(t)^2 + d_y(t)^2 + d_z(t)^2}$$
$$E_p = \int_{t_p}^{t_p + \tau} \left( v_x(t)^2 + v_y(t)^2 + v_z(t)^2 \right) dt$$

### B. Empirical Scaling Relationship
Seismological scaling laws relate $P_d$ and $E_p$ to earthquake magnitude $M$ and hypocentral distance $R$:
$$\log_{10}(P_d) = \alpha M + \beta \log_{10}(R) + \gamma + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\epsilon}^2)$$

### C. Bayesian Posterior Inference
Under Bayes' theorem:
$$P(M, R \mid \text{obs}) \propto P(\text{obs} \mid M, R) \cdot P(M) \cdot P(R)$$
- **Prior $P(M)$**: Gutenberg-Richter distribution $P(M) \propto 10^{-b M}$, bounded between $M_{\min}$ and $M_{\max}$.
- **Likelihood $P(\text{obs} \mid M, R)$**: Gaussian error model on logarithmic energy/displacement observations.
- **MCMC Sampling**: An ensemble of 32 walkers explores the posterior parameter space to infer the marginal magnitude distribution, median estimate, and 95% Bayesian credible interval.

## 3. Important Scientific Qualification
> **Warning**: Empirical regression coefficients $(\alpha, \beta, \gamma, \sigma_{\epsilon})$ depend heavily on tectonic regional attenuation, station site response, and instrument calibration. This module provides the mathematical and computational framework, but should not be deployed operationally without regional calibration against an authoritative seismic catalog.
