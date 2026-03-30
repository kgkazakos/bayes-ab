"""Tests for bayes-ab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bayes_ab.engine import analyze_binary, analyze_continuous
from bayes_ab.models import BayesABResult
from bayes_ab.narrator import BayesNarrator, build_prompt


SEED = 42


# --- analyze_binary ---

def test_binary_prefer_treatment():
    result = analyze_binary(1000, 100, 1000, 150, random_seed=SEED)
    assert result.prob_treatment_better > 0.95
    assert result.decision == "prefer_treatment"


def test_binary_prefer_control():
    result = analyze_binary(1000, 150, 1000, 80, random_seed=SEED)
    assert result.prob_treatment_better < 0.05
    assert result.decision == "prefer_control"


def test_binary_inconclusive():
    result = analyze_binary(100, 50, 100, 52, random_seed=SEED)
    assert result.decision == "inconclusive"


def test_binary_equivalent_rope():
    # Nearly identical arms — should land in ROPE
    result = analyze_binary(10000, 1000, 10000, 1001, rope_low=-0.005, rope_high=0.005, random_seed=SEED)
    assert result.rope_probability > 0.5


def test_binary_relative_uplift_direction():
    result = analyze_binary(1000, 100, 1000, 150, random_seed=SEED)
    assert result.relative_uplift > 0


def test_binary_ci_ordering():
    result = analyze_binary(1000, 100, 1000, 150, random_seed=SEED)
    assert result.control_ci[0] < result.control_ci[1]
    assert result.treatment_ci[0] < result.treatment_ci[1]


def test_binary_prob_bounded():
    result = analyze_binary(500, 50, 500, 75, random_seed=SEED)
    assert 0.0 <= result.prob_treatment_better <= 1.0


def test_binary_to_dict_serializable():
    import json
    result = analyze_binary(500, 50, 500, 75, random_seed=SEED)
    d = result.to_dict()
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed["metric_type"] == "binary"


def test_binary_custom_names():
    result = analyze_binary(500, 50, 500, 75, control_name="v1", treatment_name="v2", random_seed=SEED)
    assert result.control_name == "v1"
    assert result.treatment_name == "v2"


def test_binary_expected_loss_non_negative():
    result = analyze_binary(500, 50, 500, 75, random_seed=SEED)
    assert result.expected_loss >= 0.0


# --- analyze_continuous ---

def test_continuous_prefer_treatment():
    result = analyze_continuous(500, 10.0, 2.0, 500, 11.5, 2.0, random_seed=SEED)
    assert result.prob_treatment_better > 0.95
    assert result.decision == "prefer_treatment"


def test_continuous_prefer_control():
    result = analyze_continuous(500, 11.5, 2.0, 500, 10.0, 2.0, random_seed=SEED)
    assert result.prob_treatment_better < 0.05
    assert result.decision == "prefer_control"


def test_continuous_relative_uplift():
    result = analyze_continuous(500, 10.0, 2.0, 500, 11.0, 2.0, random_seed=SEED)
    assert abs(result.relative_uplift - 0.1) < 0.05


def test_continuous_ci_ordering():
    result = analyze_continuous(500, 10.0, 2.0, 500, 11.0, 2.0, random_seed=SEED)
    assert result.control_ci[0] < result.control_ci[1]
    assert result.treatment_ci[0] < result.treatment_ci[1]


def test_continuous_rope_defaults_to_5pct():
    result = analyze_continuous(500, 100.0, 10.0, 500, 101.0, 10.0, random_seed=SEED)
    assert result.rope_low == pytest.approx(-5.0, abs=0.01)
    assert result.rope_high == pytest.approx(5.0, abs=0.01)


def test_continuous_custom_rope():
    result = analyze_continuous(500, 10.0, 2.0, 500, 10.05, 2.0, rope_low=-0.1, rope_high=0.1, random_seed=SEED)
    assert result.rope_low == -0.1


# --- Narrator (mocked) ---

def test_narrator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        BayesNarrator(backend="cohere")


def test_narrator_uses_default_models():
    assert "gpt" in BayesNarrator(backend="openai").model
    assert "claude" in BayesNarrator(backend="anthropic").model
    assert "gemini" in BayesNarrator(backend="gemini").model


def test_narrator_openai_mocked():
    narrator = BayesNarrator(backend="openai")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "The treatment performs better."
    with patch.object(narrator, "_call_openai", return_value="The treatment performs better."):
        result = analyze_binary(1000, 100, 1000, 150, random_seed=SEED)
        narrative = narrator.narrate(result.to_dict())
    assert "treatment" in narrative.lower()


def test_narrator_returns_empty_on_error():
    narrator = BayesNarrator(backend="openai")
    with patch.object(narrator, "_call_openai", side_effect=ConnectionError("down")):
        result = analyze_binary(500, 50, 500, 75, random_seed=SEED)
        narrative = narrator.narrate(result.to_dict())
    assert narrative == ""


def test_build_prompt_contains_key_fields():
    result = analyze_binary(500, 50, 500, 75, random_seed=SEED)
    prompt = build_prompt(result.to_dict())
    assert "prob_treatment_better" in prompt
    assert "decision" in prompt
