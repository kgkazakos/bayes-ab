"""Core data models for bayes-ab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ArmStats:
    name: str
    n: int
    successes: Optional[int] = None   # for binary metrics
    mean: Optional[float] = None      # for continuous metrics
    std: Optional[float] = None       # for continuous metrics


@dataclass
class BayesABResult:
    metric_type: str          # "binary" | "continuous"
    arms: list[ArmStats]
    control_name: str
    treatment_name: str

    # Posterior summaries
    control_mean: float = 0.0
    treatment_mean: float = 0.0
    control_ci: tuple[float, float] = (0.0, 0.0)
    treatment_ci: tuple[float, float] = (0.0, 0.0)

    # Inference
    prob_treatment_better: float = 0.0
    expected_loss: float = 0.0
    relative_uplift: float = 0.0
    rope_probability: float = 0.0     # P(effect in ROPE — practically equivalent)

    # Decision
    decision: str = "inconclusive"   # "prefer_treatment" | "prefer_control" | "equivalent" | "inconclusive"
    rope_low: float = -0.01
    rope_high: float = 0.01

    n_samples: int = 50_000
    credible_interval: float = 0.95

    # Optional LLM narrative
    narrative: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_type": self.metric_type,
            "control_name": self.control_name,
            "treatment_name": self.treatment_name,
            "control_mean": round(self.control_mean, 6),
            "treatment_mean": round(self.treatment_mean, 6),
            "control_ci": [round(v, 6) for v in self.control_ci],
            "treatment_ci": [round(v, 6) for v in self.treatment_ci],
            "prob_treatment_better": round(self.prob_treatment_better, 4),
            "expected_loss": round(self.expected_loss, 6),
            "relative_uplift": round(self.relative_uplift, 4),
            "rope_probability": round(self.rope_probability, 4),
            "decision": self.decision,
            "rope_low": self.rope_low,
            "rope_high": self.rope_high,
            "n_samples": self.n_samples,
            "credible_interval": self.credible_interval,
            "narrative": self.narrative,
        }
