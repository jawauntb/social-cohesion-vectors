"""Baseline experiments for pairwise social-cohesion ranking."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    load_pairwise_examples_jsonl,
    load_scored_runs_jsonl,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun

ScoreFn = Callable[[ScoredRun], float]

STRATEGY_PRIOR = {
    "adversarial": 0.0,
    "self_protective": 0.5,
    "cooperative": 1.0,
}

POSITIVE_WORDS = (
    "cooperate",
    "collaborate",
    "repair",
    "amends",
    "apolog",
    "fair",
    "share",
    "honest",
    "truth",
    "choice",
    "consent",
    "respect",
    "listen",
    "trust",
    "mutual",
    "together",
)
NEGATIVE_WORDS = (
    "defect",
    "threat",
    "punish",
    "force",
    "coerce",
    "trick",
    "lie",
    "deceive",
    "manipulat",
    "unfair",
    "exploit",
    "hoard",
    "sabotage",
    "retaliat",
    "humiliat",
)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    accuracy: float
    ci_low: float
    ci_high: float
    n_pairs: int
    ties: int
    description: str


def run_baselines(
    *,
    scored_runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> list[BaselineResult]:
    """Evaluate all built-in baselines on pairwise examples."""

    run_index = {run.run_id: run for run in scored_runs}
    baselines: list[tuple[str, str, ScoreFn]] = [
        ("chance", "Analytic random-choice baseline.", lambda _run: 0.0),
        (
            "strategy_prior",
            "Ranks cooperative > self-protective > adversarial from metadata only.",
            strategy_prior_score,
        ),
        (
            "metrics_only",
            "Ranks using simulation metrics only, ignoring transcript text.",
            metrics_only_score,
        ),
        (
            "lexical_only",
            "Ranks using a small hand-written prosocial vs adversarial lexicon.",
            lexical_only_score,
        ),
        (
            "full_scorer",
            "Sanity check: ranks with the stored combined cohesion score.",
            lambda run: run.cohesion_score,
        ),
    ]

    results: list[BaselineResult] = []
    for name, description, score_fn in baselines:
        if name == "chance":
            results.append(
                BaselineResult(
                    name=name,
                    accuracy=0.5,
                    ci_low=0.5,
                    ci_high=0.5,
                    n_pairs=len(pairs),
                    ties=0,
                    description=description,
                )
            )
            continue
        outcomes, ties = pairwise_outcomes(pairs, run_index=run_index, score_fn=score_fn)
        accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
        ci_low, ci_high = bootstrap_ci(outcomes, samples=bootstrap_samples, seed=seed)
        results.append(
            BaselineResult(
                name=name,
                accuracy=round(accuracy, 6),
                ci_low=round(ci_low, 6),
                ci_high=round(ci_high, 6),
                n_pairs=len(outcomes),
                ties=ties,
                description=description,
            )
        )
    return results


def run_baselines_from_files(
    *,
    scored_runs_path: str | Path,
    pairs_path: str | Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> list[BaselineResult]:
    return run_baselines(
        scored_runs=load_scored_runs_jsonl(scored_runs_path),
        pairs=load_pairwise_examples_jsonl(pairs_path),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def pairwise_outcomes(
    pairs: Sequence[PairwiseExample],
    *,
    run_index: Mapping[str, ScoredRun],
    score_fn: ScoreFn,
) -> tuple[list[float], int]:
    """Return 1/0/.5 outcomes for positive-vs-negative pair rankings."""

    outcomes: list[float] = []
    ties = 0
    for pair in pairs:
        positive = run_index[pair.positive_run_id]
        negative = run_index[pair.negative_run_id]
        positive_score = score_fn(positive)
        negative_score = score_fn(negative)
        if positive_score > negative_score:
            outcomes.append(1.0)
        elif positive_score < negative_score:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)
            ties += 1
    return outcomes, ties


def strategy_prior_score(run: ScoredRun) -> float:
    return STRATEGY_PRIOR.get(run.strategy_profile, 0.0)


def metrics_only_score(run: ScoredRun) -> float:
    metrics = run.metrics
    cooperation = _metric(metrics, "cooperation_rate", "cooperation")
    repair = _metric(metrics, "repair_attempt_rate", "repair")
    fairness = _metric(metrics, "fairness_score", "fairness")
    payoff = _metric(metrics, "joint_payoff")
    hostility_inverse = 1.0 - _metric(metrics, "hostility_rate", "hostility")
    return (
        (0.28 * cooperation)
        + (0.18 * repair)
        + (0.18 * fairness)
        + (0.18 * hostility_inverse)
        + (0.18 * payoff)
    )


def lexical_only_score(run: ScoredRun) -> float:
    text = run.transcript.lower()
    positive = sum(len(re.findall(term, text)) for term in POSITIVE_WORDS)
    negative = sum(len(re.findall(term, text)) for term in NEGATIVE_WORDS)
    total = positive + negative
    if total == 0:
        return 0.5
    return positive / total


def bootstrap_ci(
    outcomes: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not outcomes:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(outcomes)
    for _ in range(samples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo = estimates[int(samples * alpha / 2)]
    hi = estimates[min(samples - 1, int(samples * (1 - alpha / 2)))]
    return lo, hi


def save_baseline_reports(
    results: Sequence[BaselineResult],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(
        json.dumps(
            {
                "baselines": [result.__dict__ for result in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(results), encoding="utf-8")


def render_markdown(results: Sequence[BaselineResult]) -> str:
    lines = [
        "# Baseline Experiments",
        "",
        "Pairwise ranking task: choose the higher-scoring social-cohesion trajectory within each scenario pair.",
        "",
        "| Baseline | Accuracy | 95% bootstrap CI | Pairs | Ties | Description |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.accuracy:.3f} | "
            f"[{result.ci_low:.3f}, {result.ci_high:.3f}] | "
            f"{result.n_pairs} | {result.ties} | {result.description} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `full_scorer` is a sanity check because the current pair labels are derived from the same stored cohesion score.",
            "- `metrics_only` asks how much of the ranking is recoverable from simulated behavior without transcript text.",
            "- `lexical_only` asks how much is recoverable from surface language alone.",
            "- The next non-circular benchmark should use held-out LLM-generated transcripts or human labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int | float):
            return max(0.0, min(1.0, float(value)))
    return 0.0
