"""Generated-trajectory benchmark orchestration and reporting."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    build_pairwise_examples,
    export_pairwise_jsonl,
    load_simulation_runs_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun, SimulationRun
from social_cohesion_vectors.scoring import score_runs

_STYLE_RE = re.compile(r"\bstyle=([^|\n]+)")


def run_generated_benchmark_from_files(
    *,
    input_path: str | Path,
    scored_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report: str | Path | None = None,
    markdown_report: str | Path | None = None,
    min_margin: float = 0.05,
    max_pairs_per_scenario: int | None = None,
) -> dict[str, Any]:
    """Run the generated benchmark pipeline from JSONL input paths."""

    runs = load_simulation_runs_jsonl(input_path)
    report = run_generated_benchmark(
        runs,
        scored_output=scored_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        min_margin=min_margin,
        max_pairs_per_scenario=max_pairs_per_scenario,
        input_path=input_path,
    )
    if json_report is not None or markdown_report is not None:
        write_reports(report, json_path=json_report, markdown_path=markdown_report)
    return report


def run_generated_benchmark(
    runs: Sequence[SimulationRun],
    *,
    scored_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    min_margin: float = 0.05,
    max_pairs_per_scenario: int | None = None,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score generated runs, build pairs/prompts, write artifacts, and summarize."""

    scored_runs = score_runs(runs)
    pairs = build_pairwise_examples(
        scored_runs,
        min_margin=min_margin,
        max_pairs_per_scenario=max_pairs_per_scenario,
    )
    prompts = activation_prompts_from_pairs(pairs)

    scored_count = write_jsonl(scored_runs, scored_output)
    pair_count = export_pairwise_jsonl(pairs, pairs_output)
    prompt_count = write_jsonl(prompts, prompts_output)

    return {
        "experiment": "generated_trajectory_benchmark",
        "paths": {
            "input": str(input_path) if input_path is not None else None,
            "scored_output": str(scored_output),
            "pairs_output": str(pairs_output),
            "prompts_output": str(prompts_output),
        },
        "parameters": {
            "min_margin": min_margin,
            "max_pairs_per_scenario": max_pairs_per_scenario,
        },
        "counts": {
            "input_runs": len(runs),
            "scored_runs": scored_count,
            "pairwise_examples": pair_count,
            "activation_prompts": prompt_count,
        },
        "scores": score_summary(scored_runs),
        "pair_margins": pair_margin_summary(pairs),
        "by_style": group_score_summary(scored_runs, key_fn=run_style),
        "by_strategy": group_score_summary(
            scored_runs,
            key_fn=lambda run: str(run.strategy_profile),
        ),
    }


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path | None,
    markdown_path: str | Path | None,
) -> None:
    """Write optional JSON and markdown reports."""

    if json_path is not None:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        output = Path(markdown_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact generated benchmark markdown report."""

    counts = _mapping(report.get("counts"))
    scores = _mapping(report.get("scores"))
    margins = _mapping(report.get("pair_margins"))
    lines = [
        "# Generated Trajectory Benchmark",
        "",
        "## Summary",
        "",
        f"- Input runs: {counts.get('input_runs', 0)}",
        f"- Scored runs: {counts.get('scored_runs', 0)}",
        f"- Pairwise examples: {counts.get('pairwise_examples', 0)}",
        f"- Activation prompts: {counts.get('activation_prompts', 0)}",
        f"- Mean cohesion score: {_fmt(scores.get('mean'))}",
        f"- Mean pair margin: {_fmt(margins.get('mean'))}",
        "",
        "## By Style",
        "",
        *_group_table(report.get("by_style")),
        "",
        "## By Strategy",
        "",
        *_group_table(report.get("by_strategy")),
        "",
    ]
    return "\n".join(lines)


def score_summary(runs: Sequence[ScoredRun]) -> dict[str, float | int]:
    """Summarize cohesion scores."""

    scores = [run.cohesion_score for run in runs]
    return _number_summary(scores)


def pair_margin_summary(pairs: Sequence[PairwiseExample]) -> dict[str, float | int]:
    """Summarize positive-minus-negative pair margins."""

    margins = [pair.positive_score - pair.negative_score for pair in pairs]
    return _number_summary(margins)


def group_score_summary(
    runs: Sequence[ScoredRun],
    *,
    key_fn: Any,
) -> dict[str, dict[str, float | int]]:
    """Summarize cohesion scores by a run grouping key."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        grouped[str(key_fn(run))].append(run.cohesion_score)
    return {key: _number_summary(values) for key, values in sorted(grouped.items())}


def run_style(run: SimulationRun) -> str:
    """Infer generated trajectory style from transcript metadata."""

    match = _STYLE_RE.search(run.transcript)
    return match.group(1).strip() if match else "unknown"


def _number_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _group_table(value: object) -> list[str]:
    groups = value if isinstance(value, Mapping) else {}
    lines = ["| Group | Count | Mean | Min | Max |", "| --- | ---: | ---: | ---: | ---: |"]
    for group, summary in groups.items():
        item = _mapping(summary)
        lines.append(
            "| "
            f"{group} | "
            f"{item.get('count', 0)} | "
            f"{_fmt(item.get('mean'))} | "
            f"{_fmt(item.get('min'))} | "
            f"{_fmt(item.get('max'))} |"
        )
    return lines


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, int | float) else "n/a"
