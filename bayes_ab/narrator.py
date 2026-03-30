"""LLM narrative interpretation for Bayesian A/B results."""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.1-pro-preview",
}

SYSTEM_PROMPT = """You are a senior data scientist who specialises in Bayesian A/B testing for product research.

Given a JSON summary of a Bayesian A/B test result, write a clear, concise narrative interpretation for a product researcher or stakeholder audience. Cover:
1. What the test found (plain numbers, no jargon)
2. What the probability of superiority means in practice
3. Whether the decision is clear or requires more data
4. One concrete recommendation

Keep it to 3-4 sentences. Do not repeat the numbers verbatim — interpret them. Do not use the word "significant"."""


def build_prompt(result_dict: dict) -> str:
    summary = {
        "metric_type": result_dict["metric_type"],
        "control": result_dict["control_name"],
        "treatment": result_dict["treatment_name"],
        "control_mean": result_dict["control_mean"],
        "treatment_mean": result_dict["treatment_mean"],
        "prob_treatment_better": result_dict["prob_treatment_better"],
        "relative_uplift": result_dict["relative_uplift"],
        "rope_probability": result_dict["rope_probability"],
        "decision": result_dict["decision"],
        "credible_interval": result_dict["credible_interval"],
        "control_ci": result_dict["control_ci"],
        "treatment_ci": result_dict["treatment_ci"],
    }
    return f"Interpret this Bayesian A/B test result:\n\n{json.dumps(summary, indent=2)}"


class BayesNarrator:
    def __init__(self, backend: str, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(DEFAULT_MODELS)}")
        self.backend = backend
        self.model = model or DEFAULT_MODELS[backend]
        self.api_key = api_key

    def narrate(self, result_dict: dict) -> str:
        prompt = build_prompt(result_dict)
        try:
            if self.backend == "openai":
                return self._call_openai(prompt)
            elif self.backend == "anthropic":
                return self._call_anthropic(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("LLM narration failed: %s", exc)
            return ""

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install 'bayes-ab[openai]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install 'bayes-ab[anthropic]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = anthropic.Anthropic(**kwargs)
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.content[0].text if response.content else ""

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("pip install 'bayes-ab[gemini]'") from exc
        if self.api_key:
            genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model, system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 300})
        return response.text or ""
