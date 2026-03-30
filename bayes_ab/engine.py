"""Bayesian A/B testing engine."""

from __future__ import annotations

from typing import Optional

import numpy as np

from bayes_ab.models import ArmStats, BayesABResult


def _beta_posterior_samples(
    n: int, successes: int, prior_alpha: float, prior_beta: float, n_samples: int
) -> np.ndarray:
    """Sample from Beta posterior: Beta(alpha + successes, beta + failures)."""
    alpha = prior_alpha + successes
    beta = prior_beta + (n - successes)
    return np.random.beta(alpha, beta, n_samples)


def _normal_posterior_samples(
    n: int, mean: float, std: float, prior_mean: float, prior_var: float, n_samples: int
) -> np.ndarray:
    """
    Sample from Normal posterior with known variance (normal-normal conjugate).
    Uses an uninformative prior (large prior_var) by default.
    """
    sample_var = (std ** 2) / n
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sample_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + mean / sample_var)
    return np.random.normal(posterior_mean, np.sqrt(posterior_var), n_samples)


def analyze_binary(
    control_n: int,
    control_successes: int,
    treatment_n: int,
    treatment_successes: int,
    control_name: str = "control",
    treatment_name: str = "treatment",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rope_low: float = -0.01,
    rope_high: float = 0.01,
    n_samples: int = 50_000,
    credible_interval: float = 0.95,
    random_seed: Optional[int] = None,
) -> BayesABResult:
    """
    Bayesian A/B test for binary (conversion) metrics.

    Uses Beta-Binomial conjugate model.
    Prior: Beta(prior_alpha, prior_beta) — default is uniform Beta(1,1).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    ctrl_samples = _beta_posterior_samples(control_n, control_successes, prior_alpha, prior_beta, n_samples)
    trt_samples = _beta_posterior_samples(treatment_n, treatment_successes, prior_alpha, prior_beta, n_samples)

    return _compute_result(
        ctrl_samples=ctrl_samples,
        trt_samples=trt_samples,
        metric_type="binary",
        control_name=control_name,
        treatment_name=treatment_name,
        control_arm=ArmStats(name=control_name, n=control_n, successes=control_successes),
        treatment_arm=ArmStats(name=treatment_name, n=treatment_n, successes=treatment_successes),
        rope_low=rope_low,
        rope_high=rope_high,
        n_samples=n_samples,
        credible_interval=credible_interval,
    )


def analyze_continuous(
    control_n: int,
    control_mean: float,
    control_std: float,
    treatment_n: int,
    treatment_mean: float,
    treatment_std: float,
    control_name: str = "control",
    treatment_name: str = "treatment",
    rope_low: Optional[float] = None,
    rope_high: Optional[float] = None,
    n_samples: int = 50_000,
    credible_interval: float = 0.95,
    random_seed: Optional[int] = None,
) -> BayesABResult:
    """
    Bayesian A/B test for continuous metrics.

    Uses Normal-Normal conjugate model with uninformative prior.
    ROPE defaults to ±5% of the control mean if not specified.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    prior_mean = control_mean
    prior_var = 1e6  # uninformative

    ctrl_samples = _normal_posterior_samples(
        control_n, control_mean, control_std, prior_mean, prior_var, n_samples
    )
    trt_samples = _normal_posterior_samples(
        treatment_n, treatment_mean, treatment_std, prior_mean, prior_var, n_samples
    )

    rope_magnitude = abs(control_mean) * 0.05
    rope_low = rope_low if rope_low is not None else -rope_magnitude
    rope_high = rope_high if rope_high is not None else rope_magnitude

    return _compute_result(
        ctrl_samples=ctrl_samples,
        trt_samples=trt_samples,
        metric_type="continuous",
        control_name=control_name,
        treatment_name=treatment_name,
        control_arm=ArmStats(name=control_name, n=control_n, mean=control_mean, std=control_std),
        treatment_arm=ArmStats(name=treatment_name, n=treatment_n, mean=treatment_mean, std=treatment_std),
        rope_low=rope_low,
        rope_high=rope_high,
        n_samples=n_samples,
        credible_interval=credible_interval,
    )


def _compute_result(
    ctrl_samples: np.ndarray,
    trt_samples: np.ndarray,
    metric_type: str,
    control_name: str,
    treatment_name: str,
    control_arm: ArmStats,
    treatment_arm: ArmStats,
    rope_low: float,
    rope_high: float,
    n_samples: int,
    credible_interval: float,
) -> BayesABResult:
    diff_samples = trt_samples - ctrl_samples
    alpha = (1.0 - credible_interval) / 2.0

    ci_lo = np.quantile(ctrl_samples, alpha)
    ci_hi = np.quantile(ctrl_samples, 1.0 - alpha)
    trt_ci_lo = np.quantile(trt_samples, alpha)
    trt_ci_hi = np.quantile(trt_samples, 1.0 - alpha)

    prob_better = float(np.mean(diff_samples > 0))
    expected_loss = float(np.mean(np.maximum(ctrl_samples - trt_samples, 0)))
    ctrl_mean = float(np.mean(ctrl_samples))
    trt_mean = float(np.mean(trt_samples))
    rope_prob = float(np.mean((diff_samples >= rope_low) & (diff_samples <= rope_high)))

    relative_uplift = (trt_mean - ctrl_mean) / ctrl_mean if ctrl_mean != 0 else 0.0

    # Decision rule
    if rope_prob > 0.95:
        decision = "equivalent"
    elif prob_better > 0.95:
        decision = "prefer_treatment"
    elif prob_better < 0.05:
        decision = "prefer_control"
    else:
        decision = "inconclusive"

    return BayesABResult(
        metric_type=metric_type,
        arms=[control_arm, treatment_arm],
        control_name=control_name,
        treatment_name=treatment_name,
        control_mean=ctrl_mean,
        treatment_mean=trt_mean,
        control_ci=(float(ci_lo), float(ci_hi)),
        treatment_ci=(float(trt_ci_lo), float(trt_ci_hi)),
        prob_treatment_better=prob_better,
        expected_loss=expected_loss,
        relative_uplift=relative_uplift,
        rope_probability=rope_prob,
        decision=decision,
        rope_low=rope_low,
        rope_high=rope_high,
        n_samples=n_samples,
        credible_interval=credible_interval,
    )
