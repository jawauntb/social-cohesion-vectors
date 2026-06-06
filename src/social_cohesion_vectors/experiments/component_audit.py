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
    min_score_accuracy: float = 1.0,
    min_slack_margin: float = 0.0,
) -> dict[str, Any]:
    """Load scored runs and pairs, then summarize pairwise component margins."""

    return run_component_margin_audit(
        scored_runs=load_scored_runs_jsonl(scored_runs_path),
        pairs=load_pairwise_examples_jsonl(pairs_path),
        group_metadata_key=group_metadata_key,
        min_score_accuracy=min_score_accuracy,
        min_slack_margin=min_slack_margin,
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
    min_score_accuracy: float = 1.0,
    min_slack_margin: float = 0.0,
    input_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Summarize how scorer components rank positive-vs-negative pairs."""

    run_index = {run.run_id: run for run in scored_runs}
    pair_rows = [
        _pair_row(pair, run_index=run_index, group_metadata_key=group_metadata_key)
        for pair in pairs
    ]
    summary = _summary_rows(pair_rows)
    groups = _group_rows(pair_rows)
    readiness = _component_readiness(
        summary=summary,
        groups=groups,
        min_score_accuracy=min_score_accuracy,
        min_slack_margin=min_slack_margin,
    )
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
            "min_score_accuracy": min_score_accuracy,
            "min_slack_margin": min_slack_margin,
        },
        "summary": {
            **summary,
            "activation_readiness": readiness["status"],
            "ready_for_activation": readiness["ready"],
        },
        "readiness": readiness,
        "groups": groups,
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
        f"- Mean slack-preservation margin: "
        f"{float(summary.get('mean_slack_preservation_margin', 0.0)):+.3f}",
        f"- Min slack-preservation margin: "
        f"{float(summary.get('min_slack_preservation_margin', 0.0)):+.3f}",
        f"- Slack component source: {summary.get('slack_component_source', 'unknown')}",
        f"- Activation readiness: `{summary.get('activation_readiness', 'not_ready')}`",
        f"- Ready for activation: {bool(summary.get('ready_for_activation', False))}",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    readiness = _mapping(report.get("readiness"))
    for gate in _sequence(readiness.get("gates")):
        gate_map = _mapping(gate)
        lines.append(
            "| "
            f"{gate_map.get('gate_id', '')} | "
            f"{float(gate_map.get('value', 0.0)):.3f} | "
            f"{float(gate_map.get('threshold', 0.0)):.3f} | "
            f"{bool(gate_map.get('passed', False))} |"
        )
    failed_groups = _sequence(readiness.get("failed_groups"))
    if failed_groups:
        lines.extend(
            [
                "",
                "Not ready for activation: component margins fail in one or "
                f"more groups ({', '.join(str(group) for group in failed_groups)}).",
            ]
        )
    lines.extend(
        [
            "",
            "## Component Means",
            "",
            "| Component | Mean positive-minus-negative margin |",
            "| --- | ---: |",
        ]
    )
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
    slack_source = _slack_component_source(positive, negative)
    score_margin = round(positive.cohesion_score - negative.cohesion_score, 6)
    return {
        "pair_id": pair.pair_id,
        "group": str(pair.metadata.get(group_metadata_key, "ungrouped")),
        "score_margin": score_margin,
        "score_prefers_positive": score_margin > 0.0,
        "slack_preservation_margin": component_margins["slack_preservation"],
        "slack_component_source": slack_source,
        "component_margins": component_margins,
    }


def _summary_rows(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_margins = [float(row["score_margin"]) for row in pair_rows]
    slack_margins = [
        float(row.get("slack_preservation_margin", 0.0)) for row in pair_rows
    ]
    slack_sources = sorted(
        {str(row.get("slack_component_source", "unknown")) for row in pair_rows}
    )
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
        "mean_slack_preservation_margin": _mean(slack_margins),
        "min_slack_preservation_margin": round(min(slack_margins), 6)
        if slack_margins
        else 0.0,
        "explicit_slack_pair_count": sum(
            1 for row in pair_rows if row.get("slack_component_source") == "explicit"
        ),
        "slack_component_source": slack_sources[0]
        if len(slack_sources) == 1
        else "mixed",
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
                "mean_slack_preservation_margin": _mean(
                    float(row.get("slack_preservation_margin", 0.0))
                    for row in rows_for_group
                ),
                "min_slack_preservation_margin": round(
                    min(
                        float(row.get("slack_preservation_margin", 0.0))
                        for row in rows_for_group
                    ),
                    6,
                ),
                "worst_component": worst_component,
                "worst_component_margin": worst_margin,
                "component_means": component_means,
            }
        )
    return rows


def _component_readiness(
    *,
    summary: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    min_score_accuracy: float,
    min_slack_margin: float,
) -> dict[str, Any]:
    failed_groups = sorted(
        {
            str(row.get("group", ""))
            for row in groups
            if float(row.get("score_accuracy", 0.0)) < min_score_accuracy
            or float(row.get("min_slack_preservation_margin", 0.0)) < min_slack_margin
        }
    )
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(summary.get("pairs", 0)),
            "threshold": 1.0,
            "passed": int(summary.get("pairs", 0)) > 0,
        },
        {
            "gate_id": "score_accuracy",
            "value": float(summary.get("score_accuracy", 0.0)),
            "threshold": min_score_accuracy,
            "passed": float(summary.get("score_accuracy", 0.0))
            >= min_score_accuracy,
        },
        {
            "gate_id": "min_slack_preservation_margin",
            "value": float(summary.get("min_slack_preservation_margin", 0.0)),
            "threshold": min_slack_margin,
            "passed": float(summary.get("min_slack_preservation_margin", 0.0))
            >= min_slack_margin,
        },
        {
            "gate_id": "group_score_and_slack_margins",
            "value": 0.0 if failed_groups else 1.0,
            "threshold": 1.0,
            "passed": not failed_groups,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "activation_ready" if ready else "not_ready_for_activation",
        "ready": ready,
        "min_score_accuracy": min_score_accuracy,
        "min_slack_margin": min_slack_margin,
        "failed_groups": failed_groups,
        "gates": gates,
        "slack_component_source": summary.get("slack_component_source", "unknown"),
    }


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _component_value(run: ScoredRun, component: str) -> float:
    if component in run.score_components:
        return float(run.score_components[component])
    if component == "slack_preservation":
        return float(run.score_components.get("autonomy_safety", 0.0))
    return 0.0


def _slack_component_source(positive: ScoredRun, negative: ScoredRun) -> str:
    positive_explicit = "slack_preservation" in positive.score_components
    negative_explicit = "slack_preservation" in negative.score_components
    if positive_explicit and negative_explicit:
        return "explicit"
    if positive_explicit or negative_explicit:
        return "mixed"
    if (
        "autonomy_safety" in positive.score_components
        and "autonomy_safety" in negative.score_components
    ):
        return "autonomy_safety_fallback"
    return "missing"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
