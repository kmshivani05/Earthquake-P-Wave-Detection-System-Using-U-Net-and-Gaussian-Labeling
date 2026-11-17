"""
Probabilistic magnitude estimation using initial P-wave displacement and MCMC sampling.

Implements Bayesian inference on initial P-wave energy/displacement with an ensemble
of 32 Markov Chain Monte Carlo walkers to yield full posterior magnitude distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MagnitudeEstimate:
    """Bayesian magnitude inference results."""
    median_magnitude: float
    mean_magnitude: float
    std_magnitude: float
    credible_interval_95: Tuple[float, float]
    samples: np.ndarray
    initial_peak_displacement: float
    energy_integral: float
    chain_acceptance_fraction: float


def compute_initial_pwave_features(
    displacement_3c: np.ndarray,
    velocity_3c: np.ndarray,
    p_arrival_sample: int,
    sampling_rate: float = 100.0,
    window_duration_seconds: float = 3.0,
) -> Tuple[float, float]:
    """Compute initial peak displacement (Pd) and cumulative velocity energy (Ep).

    Args:
        displacement_3c: 3-channel displacement array (3, T).
        velocity_3c: 3-channel velocity array (3, T).
        p_arrival_sample: Detected P-wave sample index.
        sampling_rate: Sampling frequency in Hz.
        window_duration_seconds: Duration of post-P window (typically 3s).

    Returns:
        (Pd, Ep): Peak 3D displacement and cumulative squared velocity integral.
    """
    dt = 1.0 / sampling_rate
    num_samples = int(window_duration_seconds * sampling_rate)
    start = max(0, p_arrival_sample)
    end = min(displacement_3c.shape[-1], start + num_samples)

    if start >= end:
        return 0.0, 0.0

    # 3D vector displacement magnitude in window
    disp_win = displacement_3c[:, start:end]
    disp_mag = np.sqrt(np.sum(disp_win ** 2, axis=0))
    pd = float(np.max(disp_mag)) if len(disp_mag) > 0 else 0.0

    # Velocity squared integral (Ep)
    vel_win = velocity_3c[:, start:end]
    vel_sq = np.sum(vel_win ** 2, axis=0)
    ep = float(np.sum(vel_sq) * dt)

    return pd, ep


class BayesianMagnitudeSampler:
    """32-Walker Ensemble Markov Chain Monte Carlo (MCMC) sampler for magnitude inference."""

    def __init__(
        self,
        alpha: float = 0.95,
        beta: float = -1.30,
        gamma: float = -3.50,
        sigma_obs: float = 0.35,
        b_value: float = 1.0,
        num_walkers: int = 32,
    ):
        """Initialize Bayesian magnitude model.

        Forward model:
            log10(Pd) = alpha * M + beta * log10(R) + gamma + epsilon
            epsilon ~ Normal(0, sigma_obs^2)

        Priors:
            M ~ Gutenberg-Richter: P(M) proportional to 10^(-b * M) for M in [1.0, 9.0]
            R ~ LogNormal or uniform prior over [5.0, 200.0] km

        Args:
            alpha: Scaling slope for magnitude.
            beta: Attenuation slope for hypocentral distance R (km).
            gamma: Empirical intercept.
            sigma_obs: Logarithmic observation uncertainty.
            b_value: Seismological Gutenberg-Richter b-value.
            num_walkers: Number of parallel MCMC walkers (paper specifies 32).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_obs = sigma_obs
        self.b_value = b_value
        self.num_walkers = num_walkers

    def log_prior(self, M: float, R: float) -> float:
        """Log prior probability."""
        if not (1.0 <= M <= 9.0) or not (2.0 <= R <= 300.0):
            return -np.inf
        # Gutenberg-Richter prior: ln(P(M)) = -b * ln(10) * M
        log_p_m = -self.b_value * np.log(10.0) * M
        # Jeffrey's uninformative log-uniform prior on distance: P(R) proportional to 1/R
        log_p_r = -np.log(R)
        return log_p_m + log_p_r

    def log_likelihood(self, M: float, R: float, log_pd_obs: float) -> float:
        """Gaussian log-likelihood given observation."""
        log_pd_model = self.alpha * M + self.beta * np.log10(R) + self.gamma
        diff = log_pd_obs - log_pd_model
        return -0.5 * (diff / self.sigma_obs) ** 2 - np.log(self.sigma_obs * np.sqrt(2.0 * np.pi))

    def log_posterior(self, params: np.ndarray, log_pd_obs: float) -> float:
        M, R = params[0], params[1]
        lp = self.log_prior(M, R)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(M, R, log_pd_obs)

    def sample(
        self,
        pd_obs: float,
        ep_obs: float,
        num_steps: int = 1000,
        burn_in: int = 200,
        seed: int = 42,
    ) -> MagnitudeEstimate:
        """Execute 32-walker ensemble MCMC sampling.

        Args:
            pd_obs: Observed peak displacement.
            ep_obs: Observed energy integral.
            num_steps: Steps per walker.
            burn_in: Initial burn-in steps to discard.
            seed: Random seed.

        Returns:
            MagnitudeEstimate with posterior distribution statistics.
        """
        rng = np.random.RandomState(seed)
        log_pd_obs = np.log10(max(1e-9, pd_obs))

        # Initialize 32 walkers around prior centroid
        # Parameters: [Magnitude M, Distance R (km)]
        init_M = rng.uniform(3.5, 5.5, size=self.num_walkers)
        init_R = rng.uniform(20.0, 80.0, size=self.num_walkers)
        walkers = np.column_stack([init_M, init_R])

        accepted = 0
        total_proposals = 0
        chain_M: List[float] = []

        # Affine-invariant stretch-move algorithm (Goodman & Weare)
        a = 2.0  # standard scale factor
        for step in range(num_steps):
            for k in range(self.num_walkers):
                # Pick a random complementary walker j != k
                other_idx = rng.choice([i for i in range(self.num_walkers) if i != k])
                walker_k = walkers[k]
                walker_j = walkers[other_idx]

                # Stretch factor z drawn from g(z) proportional to 1/sqrt(z) on [1/a, a]
                z = ((a - 1.0) * rng.rand() + 1.0) ** 2 / a
                proposed = walker_j + z * (walker_k - walker_j)

                curr_post = self.log_posterior(walker_k, log_pd_obs)
                prop_post = self.log_posterior(proposed, log_pd_obs)

                # Acceptance probability: min(1, z^(dim - 1) * exp(prop_post - curr_post))
                # dim = 2 (M, R) -> z^(2-1) = z
                log_accept = np.log(z) + prop_post - curr_post
                total_proposals += 1

                if np.log(rng.rand()) < log_accept:
                    walkers[k] = proposed
                    accepted += 1

                if step >= burn_in:
                    chain_M.append(float(walkers[k, 0]))

        samples_arr = np.array(chain_M)
        q2_5, q50, q97_5 = np.percentile(samples_arr, [2.5, 50.0, 97.5])
        acceptance_fraction = accepted / max(1, total_proposals)

        return MagnitudeEstimate(
            median_magnitude=float(q50),
            mean_magnitude=float(np.mean(samples_arr)),
            std_magnitude=float(np.std(samples_arr)),
            credible_interval_95=(float(q2_5), float(q97_5)),
            samples=samples_arr,
            initial_peak_displacement=pd_obs,
            energy_integral=ep_obs,
            chain_acceptance_fraction=float(acceptance_fraction),
        )
