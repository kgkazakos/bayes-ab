"""Click CLI for bayes-ab."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

if TYPE_CHECKING:
    from bayes_ab.models import BayesABResult

try:
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    _rich = True
except ImportError:
    _rich = False


@click.group()
@click.version_option(package_name="bayes-ab")
def cli() -> None:
    """Bayesian A/B testing — posteriors, credible intervals, and decision rules."""


@cli.command()
@click.option("--control-n", required=True, type=int, help="Control group size.")
@click.option("--control-successes", required=True, type=int, help="Control group conversions.")
@click.option("--treatment-n", required=True, type=int, help="Treatment group size.")
@click.option("--treatment-successes", required=True, type=int, help="Treatment group conversions.")
@click.option("--control-name", default="control", show_default=True)
@click.option("--treatment-name", default="treatment", show_default=True)
@click.option("--rope-low", default=-0.01, show_default=True, help="ROPE lower bound (absolute difference).")
@click.option("--rope-high", default=0.01, show_default=True, help="ROPE upper bound (absolute difference).")
@click.option("--credible-interval", default=0.95, show_default=True)
@click.option("--n-samples", default=50000, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None, help="Write JSON result to this path.")
@click.option("--quiet", "-q", is_flag=True)
def binary(
    control_n: int, control_successes: int, treatment_n: int, treatment_successes: int,
    control_name: str, treatment_name: str, rope_low: float, rope_high: float,
    credible_interval: float, n_samples: int, seed: Optional[int],
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Bayesian A/B test for binary (conversion) metrics."""
    from bayes_ab.engine import analyze_binary
    from bayes_ab.narrator import BayesNarrator

    result = analyze_binary(
        control_n=control_n, control_successes=control_successes,
        treatment_n=treatment_n, treatment_successes=treatment_successes,
        control_name=control_name, treatment_name=treatment_name,
        rope_low=rope_low, rope_high=rope_high,
        n_samples=n_samples, credible_interval=credible_interval,
        random_seed=seed,
    )

    if llm_backend:
        narrator = BayesNarrator(backend=llm_backend)
        result.narrative = narrator.narrate(result.to_dict())

    if not quiet:
        _print_result(result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        if not quiet:
            click.echo(f"✓ Result written to {output}")

    if result.decision == "inconclusive":
        sys.exit(1)


@cli.command()
@click.option("--control-n", required=True, type=int)
@click.option("--control-mean", required=True, type=float)
@click.option("--control-std", required=True, type=float)
@click.option("--treatment-n", required=True, type=int)
@click.option("--treatment-mean", required=True, type=float)
@click.option("--treatment-std", required=True, type=float)
@click.option("--control-name", default="control", show_default=True)
@click.option("--treatment-name", default="treatment", show_default=True)
@click.option("--rope-low", default=None, type=float)
@click.option("--rope-high", default=None, type=float)
@click.option("--credible-interval", default=0.95, show_default=True)
@click.option("--n-samples", default=50000, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def continuous(
    control_n: int, control_mean: float, control_std: float,
    treatment_n: int, treatment_mean: float, treatment_std: float,
    control_name: str, treatment_name: str,
    rope_low: Optional[float], rope_high: Optional[float],
    credible_interval: float, n_samples: int, seed: Optional[int],
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Bayesian A/B test for continuous metrics (revenue, time-on-task, etc.)."""
    from bayes_ab.engine import analyze_continuous
    from bayes_ab.narrator import BayesNarrator

    result = analyze_continuous(
        control_n=control_n, control_mean=control_mean, control_std=control_std,
        treatment_n=treatment_n, treatment_mean=treatment_mean, treatment_std=treatment_std,
        control_name=control_name, treatment_name=treatment_name,
        rope_low=rope_low, rope_high=rope_high,
        n_samples=n_samples, credible_interval=credible_interval,
        random_seed=seed,
    )

    if llm_backend:
        narrator = BayesNarrator(backend=llm_backend)
        result.narrative = narrator.narrate(result.to_dict())

    if not quiet:
        _print_result(result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


def _print_result(result: "BayesABResult") -> None:
    if _rich:
        _rich_print(result)
    else:
        _plain_print(result)


def _fmt_ci(ci: tuple) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _rich_print(result: "BayesABResult") -> None:
    from rich.console import Console
    console = Console()
    decision_colour = {
        "prefer_treatment": "green", "prefer_control": "red",
        "equivalent": "yellow", "inconclusive": "dim",
    }.get(result.decision, "white")

    console.print()
    console.print(Panel(
        f"[bold]Decision:[/] [{decision_colour}]{result.decision.replace('_', ' ').upper()}[/]  "
        f"| P(treatment better) = [bold]{result.prob_treatment_better:.1%}[/]  "
        f"| Relative uplift = [bold]{result.relative_uplift:+.2%}[/]",
        title=f"[bold]bayes-ab[/] · {result.control_name} vs {result.treatment_name}",
        expand=False,
    ))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    t.add_column("Arm")
    t.add_column("Posterior mean")
    t.add_column(f"{result.credible_interval:.0%} CI")
    t.add_row(result.control_name, f"{result.control_mean:.4f}", _fmt_ci(result.control_ci))
    t.add_row(result.treatment_name, f"{result.treatment_mean:.4f}", _fmt_ci(result.treatment_ci))
    console.print(t)
    console.print(f"  ROPE [{result.rope_low}, {result.rope_high}]: P(equivalent) = {result.rope_probability:.1%}")
    console.print(f"  Expected loss: {result.expected_loss:.6f}")
    if result.narrative:
        console.print(Panel(result.narrative, title="Narrative", expand=False))
    console.print()


def _plain_print(result: "BayesABResult") -> None:
    click.echo(f"\nbayes-ab: {result.control_name} vs {result.treatment_name}")
    click.echo(f"  Decision: {result.decision}")
    click.echo(f"  P(treatment better): {result.prob_treatment_better:.1%}")
    click.echo(f"  Relative uplift: {result.relative_uplift:+.2%}")
    click.echo(f"  Control posterior mean: {result.control_mean:.4f}")
    click.echo(f"  Treatment posterior mean: {result.treatment_mean:.4f}")
    if result.narrative:
        click.echo(f"\n  {result.narrative}")
