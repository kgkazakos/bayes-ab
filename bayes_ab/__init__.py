"""bayes-ab — Bayesian A/B testing for product researchers."""

from bayes_ab.engine import analyze_binary, analyze_continuous
from bayes_ab.models import BayesABResult

__version__ = "0.1.0"
__all__ = ["analyze_binary", "analyze_continuous", "BayesABResult"]
