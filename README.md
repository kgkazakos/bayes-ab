# bayes-ab

Bayesian A/B testing for product researchers — posterior distributions, credible intervals, and clear decision rules. No p-values required.

```
$ bayes-ab binary --control-n 1000 --control-successes 87 \
    --treatment-n 1000 --treatment-successes 112

  Decision: PREFER TREATMENT  |  P(treatment better) = 97.3%  |  Relative uplift = +28.7%

  Arm         Posterior mean   95% CI
  control     0.0872           [0.0703, 0.1054]
  treatment   0.1121           [0.0930, 0.1323]

  ROPE [-0.01, 0.01]: P(equivalent) = 0.1%
  Expected loss: 0.000023
```

## The problem

Frequentist A/B testing gives you a p-value and a binary "significant / not significant" answer. It doesn't tell you *how much* better the treatment is likely to be, or what the probability of a wrong decision is. Bayesian testing does. `bayes-ab` gives you a posterior distribution over the true effect, a credible interval you can actually interpret, and a decision framework that handles "probably equivalent" as a first-class outcome via the ROPE.

## What it does

**Binary metrics (conversion rates, click-through, completion)**
Uses a Beta-Binomial conjugate model. Prior defaults to Beta(1, 1) — uniform and uninformative. Posterior is updated analytically from observed successes and trials.

**Continuous metrics (revenue, time-on-task, NPS, session length)**
Uses a Normal-Normal conjugate model with an uninformative prior. Posterior is sampled via Monte Carlo (default 50,000 draws).

**For both metric types:**
- Posterior mean and credible interval per arm
- Probability that treatment beats control
- Expected loss (cost of choosing wrong arm)
- Relative uplift
- ROPE probability — P(effect is practically negligible)
- Decision: `prefer_treatment` | `prefer_control` | `equivalent` | `inconclusive`
- Optional LLM narrative interpretation

## Install

```bash
pip install bayes-ab
```

With optional LLM narrative layer:

```bash
pip install "bayes-ab[openai]"
pip install "bayes-ab[anthropic]"
pip install "bayes-ab[gemini]"
pip install "bayes-ab[all]"
```

## Usage

### CLI

```bash
# Binary metric (conversion rate)
bayes-ab binary \
  --control-n 1000 --control-successes 87 \
  --treatment-n 1000 --treatment-successes 112 \
  --control-name "v1" --treatment-name "v2"

# Continuous metric (e.g. session length in seconds)
bayes-ab continuous \
  --control-n 500 --control-mean 142.3 --control-std 38.1 \
  --treatment-n 500 --treatment-mean 157.8 --treatment-std 41.2

# With custom ROPE and LLM narrative
bayes-ab binary \
  --control-n 2000 --control-successes 180 \
  --treatment-n 2000 --treatment-successes 210 \
  --rope-low -0.005 --rope-high 0.005 \
  --llm-backend anthropic \
  --output result.json
```

### Python API

```python
from bayes_ab import analyze_binary, analyze_continuous

# Binary
result = analyze_binary(
    control_n=1000, control_successes=87,
    treatment_n=1000, treatment_successes=112,
    control_name="v1", treatment_name="v2",
    rope_low=-0.01, rope_high=0.01,
)

print(result.decision)                   # "prefer_treatment"
print(result.prob_treatment_better)      # 0.973
print(result.relative_uplift)            # 0.287
print(result.control_ci)                 # (0.0703, 0.1054)

# Continuous
result = analyze_continuous(
    control_n=500, control_mean=142.3, control_std=38.1,
    treatment_n=500, treatment_mean=157.8, treatment_std=41.2,
)

import json
print(json.dumps(result.to_dict(), indent=2))
```

## Decision rules

| Condition | Decision |
|-----------|---------|
| P(treatment better) > 95% | `prefer_treatment` |
| P(treatment better) < 5% | `prefer_control` |
| P(effect in ROPE) > 95% | `equivalent` |
| Otherwise | `inconclusive` |

Exit code is `1` when decision is `inconclusive` — useful as a CI gate to prevent shipping undecided experiments.

## ROPE

The Region of Practical Equivalence defines what counts as a meaninglessly small difference. For binary metrics, the default ROPE is ±1 percentage point (`rope_low=-0.01, rope_high=0.01`). For continuous metrics, it defaults to ±5% of the control mean. Both are fully configurable.

## Part of the CPR Ecosystem

`bayes-ab` is Tool 4.6 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools in this series:**

| Tool | What it does |
|------|-------------|
| [`survey-cleaner`](https://github.com/kgkazakos/survey-cleaner) | Remove speeders, straight-liners and gibberish open-ends |
| [`regression-diagnostics`](https://github.com/kgkazakos/regression-diagnostics) | OLS assumption checks, VIF, Cook's distance |
| [`scale-validator`](https://github.com/kgkazakos/scale-validator) | Cronbach's alpha, item analysis, factor retention |
| [`power-analyst`](https://github.com/kgkazakos/power-analyst) | Sample size and power curves for common test types |

Part of the broader **Computational Product Research (CPR)** discipline. Core projects — CausalTrack, Synthetic User Council, Co-Reasoning Canvas — are developed separately and target CHI 2027/2028.

## License

MIT
