"""Component-margin audits for pairwise benchmark failures."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    load_pairwise_examples_jsonl,
    load_scored_runs_jsonl,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun
from social_cohesion_vectors.scoring import COMPONENT_NAMES


def run_component_margin_audit_from_files(
    *,
    scored_runs_path: str | Path,
    pairs_path: str | Path,
    group_metadata_key: str = "primary_fault_class",
) -> dict[str, Any]:
    """Load scored runs and pairs, then summarize pairwise component margins."""

    return run_component_margin_audit(
        scored_runs=load_scored_runs_jsonl(scored_runs_path),
        pairs=load_pairwise_examples_jsonl(pairs_path),
        group_metadata_key=group_metadata_key,
        input_paths={
            "scored_runs": str(scored_runs_path),
            "pairs": str(pairs_path),
        },
    )


def run_component_margin_audit(
    *,
    scored_runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    group_metadata_key: str = "primary_fault_class",
    input_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Summarize how scorer components rank positive-vs-negative pairs."""

    run_index = {run.run_id: run for run in scored_runs}
    pair_rows = [
        _pair_row(pair, run_index=run_index, group_metadata_key=group_metadata_key)
        for pair in pairs
    ]
    return {
        "experiment": "component_margin_audit",
        "description": (
            "Audits positive-minus-negative margins for the stored scorer score "
            "and each rubric component, grouped by pair metadata."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "scored_runs": len(scored_runs),
            "pairs": len(pairs),
            "group_metadata_key": group_metadata_key,
        },
        "summary": _summary_rows(pair_rows),
        "groups": _group_rows(pair_rows),
        "pairs": pair_rows,
    }


def save_component_margin_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown component-margin audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_component_margin_markdown(report), encoding="utf-8")


def render_component_margin_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise component-margin audit."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Component Margin Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Scorer prefers positive: {int(summary.get('score_prefers_positive', 0))}",
        f"- Scorer pairwise accuracy: {float(summary.get('score_accuracy', 0.0)):.3f}",
        f"- Mean score margin: {float(summary.get('mean_score_margin', 0.0)):+.3f}",
        "",
        "## Component Means",
        "",
        "| Component | Mean positive-minus-negative margin |",
        "| --- | ---: |",
    ]
    for component in COMPONENT_NAMES:
        lines.append(
            f"| {component} | "
            f"{float(summary.get(f'mean_{component}_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Groups",
            "",
            "| Group | Pairs | Score acc | Mean score margin | Worst component | Worst margin |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in _sequence(report.get("groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('score_accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_score_margin', 0.0)):+.3f} | "
            f"{row_map.get('worst_component', '')} | "
            f"{float(row_map.get('worst_component_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _pair_row(
    pair: PairwiseExample,
    *,
    run_index: Mapping[str, ScoredRun],
    group_metadata_key: str,
) -> dict[str, Any]:
    positive = run_index[pair.positive_run_id]
    negative = run_index[pair.negative_run_id]
    component_margins = {
        component: round(
            _component_value(positive, component)
            - _component_value(negative, component),
            6,
        )
        for component in COMPONENT_NAMES
    }
    score_margin = round(positive.cohesion_score - negative.cohesion_score, 6)
    return {
        "pair_id": pair.pair_id,
        "group": str(pair.metadata.get(group_metadata_key, "ungrouped")),
        "score_margin": score_margin,
        "score_prefers_positive": score_margin > 0.0,
        "component_margins": component_margins,
    }


def _summary_rows(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_margins = [float(row["score_margin"]) for row in pair_rows]
    summary: dict[str, Any] = {
        "pairs": len(pair_rows),
        "score_prefers_positive": sum(
            1 for row in pair_rows if bool(row["score_prefers_positive"])
        ),
        "score_accuracy": round(
            sum(1 for row in pair_rows if bool(row["score_prefers_positive"]))
            / len(pair_rows),
            6,
        )
        if pair_rows
        else 0.0,
        "mean_score_margin": _mean(score_margins),
    }
    for component in COMPONENT_NAMES:
        summary[f"mean_{component}_margin"] = _mean(
            float(_mapping(row["component_margins"])[component]) for row in pair_rows
        )
    return summary


def _group_rows(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["group"])].append(row)
    rows: list[dict[str, Any]] = []
    for group, rows_for_group in sorted(grouped.items()):
        component_means = {
            component: _mean(
                float(_mapping(row["component_margins"])[component])
                for row in rows_for_group
            )
            for component in COMPONENT_NAMES
        }
        worst_component, worst_margin = min(
            component_means.items(),
            key=lambda item: item[1],
        )
        score_prefers_positive = sum(
            1 for row in rows_for_group if bool(row["score_prefers_positive"])
        )
        rows.append(
            {
                "group": group,
                "pairs": len(rows_for_group),
                "score_accuracy": round(score_prefers_positive / len(rows_for_group), 6),
                "mean_score_margin": _mean(
                    float(row["score_margin"]) for row in rows_for_group
                ),
                "worst_component": worst_component,
                "worst_component_margin": worst_margin,
                "component_means": component_means,
            }
        )
    return rows


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _component_value(run: ScoredRun, component: str) -> float:
    if component in run.score_components:
        return float(run.score_components[component])
    if component == "slack_preservation":
        return float(run.score_components.get("autonomy_safety", 0.0))
    return 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
