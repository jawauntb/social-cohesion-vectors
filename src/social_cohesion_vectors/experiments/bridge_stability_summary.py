"""Summarize constructed bridge stability over perturbation diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl


@dataclass(frozen=True)
class BridgeReportInput:
    """Named bridge diagnostic report path."""

    model_id: str
    path: Path


def run_bridge_stability_summary_from_files(
    *,
    bridge_reports: Sequence[BridgeReportInput],
    fresh_source_pairs_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load bridge diagnostics and summarize constructed-bridge failures."""

    pair_metadata = _load_pair_metadata(fresh_source_pairs_path)
    reports = [
        (item.model_id, _load_json(item.path), str(item.path))
        for item in bridge_reports
    ]
    model_rows = [
        _model_summary(model_id=model_id, report=report, report_path=report_path)
        for model_id, report, report_path in reports
    ]
    failure_rows = [
        row
        for model_id, report, _report_path in reports
        for row in _constructed_failure_rows(
            model_id=model_id,
            report=report,
            pair_metadata=pair_metadata,
        )
    ]
    cluster_rows = _cluster_rows(failure_rows)
    summary = _summary(
        model_rows=model_rows,
        failure_rows=failure_rows,
        cluster_rows=cluster_rows,
    )
    return {
        "experiment": "bridge_stability_summary",
        "description": (
            "Summarizes constructed-bridge failures by model, bridge family, "
            "evaluation slice, and perturbation id for perturbation diagnostics."
        ),
        "inputs": {
            "bridge_reports": [
                {"model_id": item.model_id, "path": str(item.path)}
                for item in bridge_reports
            ],
            "fresh_source_pairs_path": (
                str(fresh_source_pairs_path) if fresh_source_pairs_path else None
            ),
        },
        "summary": summary,
        "model_rows": model_rows,
        "failure_rows": failure_rows,
        "failure_clusters": cluster_rows,
        "interpretation_guardrail": (
            "Bridge stability summaries are post-hoc text-benchmark activation "
            "diagnostics. They do not establish causal steering, human "
            "behavioral, neural, clinical, deployment, or real-world "
            "social-effect claims."
        ),
    }


def save_bridge_stability_summary(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write a bridge-stability report as JSON and Markdown."""

    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_bridge_stability_summary_markdown(report),
        encoding="utf-8",
    )


def render_bridge_stability_summary_markdown(report: Mapping[str, Any]) -> str:
    """Render a bridge-stability summary as Markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Bridge Stability Summary",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Models: {int(summary.get('models', 0))}",
        f"- Constructed failure rows: {int(summary.get('constructed_failure_rows', 0))}",
        f"- Failing models: {int(summary.get('failing_models', 0))}",
        f"- Worst constructed margin: {float(summary.get('worst_margin', 0.0)):+.3f}",
        f"- Most failed perturbation: `{summary.get('most_failed_perturbation_id', '')}`",
        f"- Most failed bridge family: `{summary.get('most_failed_bridge_family', '')}`",
        "",
        "## Model Rows",
        "",
        "| Model | Readiness | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("model_rows")):
        item = _mapping(row)
        lines.append(
            "| "
            f"`{item.get('model_id', '')}` | "
            f"`{item.get('readiness', '')}` | "
            f"{float(item.get('constructed_fresh_source_min_margin', 0.0)):+.3f} | "
            f"{int(item.get('constructed_fresh_source_failed_pairs', 0))} | "
            f"{float(item.get('constructed_fresh_target_min_margin', 0.0)):+.3f} | "
            f"{int(item.get('constructed_fresh_target_failed_pairs', 0))} |"
        )

    lines.extend(
        [
            "",
            "## Failure Clusters",
            "",
            "| Model | Evaluation | Bridge family | Perturbation | Failures | Directions | Min margin |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("failure_clusters"))[:40]:
        item = _mapping(row)
        lines.append(
            "| "
            f"`{item.get('model_id', '')}` | "
            f"`{item.get('evaluation', '')}` | "
            f"`{item.get('bridge_family', '')}` | "
            f"`{item.get('perturbation_id', '')}` | "
            f"{int(item.get('failed_pairs', 0))} | "
            f"{int(item.get('directions', 0))} | "
            f"{float(item.get('min_margin', 0.0)):+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top Failed Pairs",
            "",
            "| Model | Evaluation | Direction | Pair | Perturbation | Margin |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    sorted_failures = sorted(
        (_mapping(row) for row in _sequence(report.get("failure_rows"))),
        key=lambda row: float(row.get("margin", 0.0)),
    )
    for row in sorted_failures[:30]:
        lines.append(
            "| "
            f"`{row.get('model_id', '')}` | "
            f"`{row.get('evaluation', '')}` | "
            f"`{row.get('direction_id', '')}` | "
            f"`{row.get('pair_id', '')}` | "
            f"`{row.get('perturbation_id', '')}` | "
            f"{float(row.get('margin', 0.0)):+.3f} |"
        )

    lines.extend(["", "## Interpretation Guardrail", "", str(report.get("interpretation_guardrail", ""))])
    return "\n".join(lines) + "\n"


def _model_summary(
    *,
    model_id: str,
    report: Mapping[str, Any],
    report_path: str,
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    return {
        "model_id": model_id,
        "report_path": report_path,
        "readiness": str(summary.get("readiness", "")),
        "constructed_direction_count": int(summary.get("constructed_direction_count", 0)),
        "constructed_fresh_source_min_accuracy": float(
            summary.get("constructed_fresh_source_min_accuracy", 0.0)
        ),
        "constructed_fresh_source_min_margin": float(
            summary.get("constructed_fresh_source_min_margin", 0.0)
        ),
        "constructed_fresh_source_failed_pairs": int(
            summary.get("constructed_fresh_source_failed_pairs", 0)
        ),
        "constructed_fresh_target_min_accuracy": float(
            summary.get("constructed_fresh_target_min_accuracy", 0.0)
        ),
        "constructed_fresh_target_min_margin": float(
            summary.get("constructed_fresh_target_min_margin", 0.0)
        ),
        "constructed_fresh_target_failed_pairs": int(
            summary.get("constructed_fresh_target_failed_pairs", 0)
        ),
    }


def _constructed_failure_rows(
    *,
    model_id: str,
    report: Mapping[str, Any],
    pair_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in _sequence(report.get("direction_evaluations")):
        direction_row = _mapping(direction)
        if direction_row.get("direction_family") != "constructed_bridge":
            continue
        direction_id = str(direction_row.get("direction_id", ""))
        bridge_family = direction_id.split(":", 1)[0] if direction_id else ""
        for eval_key in ("on_fresh_source", "on_fresh_target"):
            evaluation = eval_key.removeprefix("on_")
            eval_row = _mapping(direction_row.get(eval_key))
            for failure in _sequence(eval_row.get("failed_pairs")):
                failure_row = _mapping(failure)
                pair_id = str(failure_row.get("pair_id", ""))
                metadata = _mapping(pair_metadata.get(pair_id))
                rows.append(
                    {
                        "model_id": model_id,
                        "direction_id": direction_id,
                        "bridge_family": bridge_family,
                        "evaluation": evaluation,
                        "pair_id": pair_id,
                        "base_contrast_id": str(
                            metadata.get("base_contrast_id")
                            or _base_from_pair_id(pair_id)
                        ),
                        "perturbation_id": str(
                            metadata.get("perturbation_id")
                            or _perturbation_from_pair_id(pair_id)
                        ),
                        "margin": float(failure_row.get("margin", 0.0)),
                    }
                )
    return rows


def _cluster_rows(failure_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    directions: defaultdict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in failure_rows:
        key = (
            str(row.get("model_id", "")),
            str(row.get("evaluation", "")),
            str(row.get("bridge_family", "")),
            str(row.get("perturbation_id", "")),
        )
        cluster = clusters.setdefault(
            key,
            {
                "model_id": key[0],
                "evaluation": key[1],
                "bridge_family": key[2],
                "perturbation_id": key[3],
                "failed_pairs": 0,
                "min_margin": float(row.get("margin", 0.0)),
            },
        )
        cluster["failed_pairs"] = int(cluster["failed_pairs"]) + 1
        cluster["min_margin"] = min(
            float(cluster["min_margin"]),
            float(row.get("margin", 0.0)),
        )
        directions[key].add(str(row.get("direction_id", "")))
    for key, cluster in clusters.items():
        cluster["directions"] = len(directions[key])
    return sorted(
        clusters.values(),
        key=lambda row: (
            float(row["min_margin"]),
            -int(row["failed_pairs"]),
            str(row["model_id"]),
        ),
    )


def _summary(
    *,
    model_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fresh_source_failure_rows = [
        row for row in failure_rows if row.get("evaluation") == "fresh_source"
    ]
    perturbation_counts = _counts(
        row.get("perturbation_id", "") for row in fresh_source_failure_rows
    )
    bridge_counts = _counts(row.get("bridge_family", "") for row in failure_rows)
    failing_models = {
        str(row.get("model_id", ""))
        for row in model_rows
        if int(row.get("constructed_fresh_source_failed_pairs", 0)) > 0
        or int(row.get("constructed_fresh_target_failed_pairs", 0)) > 0
    }
    margins = [float(row.get("margin", 0.0)) for row in failure_rows]
    return {
        "models": len(model_rows),
        "failing_models": len(failing_models),
        "constructed_failure_rows": len(failure_rows),
        "failure_clusters": len(cluster_rows),
        "worst_margin": min(margins) if margins else 0.0,
        "most_failed_perturbation_id": _most_common_key(perturbation_counts),
        "most_failed_bridge_family": _most_common_key(bridge_counts),
        "failed_by_perturbation": perturbation_counts,
        "failed_by_bridge_family": bridge_counts,
    }


def _load_pair_metadata(path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    return {
        pair.pair_id: dict(pair.metadata)
        for pair in load_pairwise_examples_jsonl(path)
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")))


def _counts(values: Sequence[object] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _most_common_key(counts: Mapping[str, int]) -> str:
    return next(iter(counts), "")


def _base_from_pair_id(pair_id: str) -> str:
    if "::" in pair_id:
        pair_id = pair_id.split("::", 1)[1]
    return pair_id.split("__", 1)[0]


def _perturbation_from_pair_id(pair_id: str) -> str:
    if "::" in pair_id:
        return pair_id.split("::", 1)[1]
    return _base_from_pair_id(pair_id)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []
