"""Multi-axis guardrail monitoring over trait-axis activations."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.experiments.transfer import load_activation_payload


def axis_id_from_pair_id(pair_id: str) -> str:
    """Parse a trait-axis id from a trait-axis pair id."""

    parts = pair_id.split("::")
    if len(parts) >= 3 and parts[0] == "trait-axis":
        return parts[1]
    return "unknown"


def run_guardrail_monitoring_from_npz(path: str | Path) -> dict[str, Any]:
    """Load trait-axis activations and run multi-axis monitoring."""

    payload = load_activation_payload(path)
    return run_guardrail_monitoring(
        activations=payload.activations,
        pair_ids=[str(pair_id) for pair_id in payload.pair_ids],
        labels=[str(label) for label in payload.labels],
        activation_path=str(path),
    )


def run_guardrail_monitoring(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    activation_path: str | None = None,
) -> dict[str, Any]:
    """Train one direction per axis and report weak/inverted pair margins."""

    axis_groups = _axis_pair_groups(pair_ids)
    axis_reports: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    for axis_id, selected_pairs in sorted(axis_groups.items()):
        mask = np.asarray([pair_id in selected_pairs for pair_id in pair_ids], dtype=bool)
        direction = train_direction_from_arrays(
            activations[mask],
            labels=np.asarray(labels, dtype=str)[mask],
        ).direction
        axis_pairs = _pair_margins(
            activations=activations,
            pair_ids=pair_ids,
            labels=labels,
            direction=direction,
            selected_pairs=selected_pairs,
            axis_id=axis_id,
        )
        pair_reports.extend(axis_pairs)
        margins = [float(row["margin"]) for row in axis_pairs]
        axis_reports.append(
            {
                "axis_id": axis_id,
                "pairs": len(axis_pairs),
                "alerts": sum(1 for row in axis_pairs if row["alert"]),
                "mean_margin": _mean(margins),
                "min_margin": round(min(margins), 6) if margins else 0.0,
            }
        )

    alerts = [row for row in pair_reports if row["alert"]]
    return {
        "experiment": "guardrail_monitoring",
        "description": (
            "Trains one activation direction per trait axis and flags weak or "
            "inverted positive-minus-negative margins."
        ),
        "inputs": {
            "activation_npz": activation_path,
            "prompts": int(activations.shape[0]),
            "activation_dim": int(activations.shape[1]) if activations.ndim == 2 else 0,
            "axes": len(axis_reports),
        },
        "summary": {
            "axes": len(axis_reports),
            "pairs": len(pair_reports),
            "alerts": len(alerts),
            "mean_margin": _mean(float(row["margin"]) for row in pair_reports),
        },
        "axes": axis_reports,
        "pairs": pair_reports,
    }


def save_guardrail_monitoring_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown guardrail monitoring reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_guardrail_monitoring_markdown(report), encoding="utf-8")


def render_guardrail_monitoring_markdown(report: Mapping[str, Any]) -> str:
    """Render guardrail monitoring output as markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Guardrail Monitoring",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Axes: {int(summary.get('axes', 0))}",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Alerts: {int(summary.get('alerts', 0))}",
        f"- Mean margin: {float(summary.get('mean_margin', 0.0)):.3f}",
        "",
        "## Axes",
        "",
        "| Axis | Pairs | Alerts | Mean margin | Min margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("axes")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('axis_id', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{int(row_map.get('alerts', 0))} | "
            f"{float(row_map.get('mean_margin', 0.0)):.3f} | "
            f"{float(row_map.get('min_margin', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Margins",
            "",
            "| Pair | Axis | Margin | Alert |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in _sequence(report.get("pairs")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('pair_id', '')} | "
            f"{row_map.get('axis_id', '')} | "
            f"{float(row_map.get('margin', 0.0)):.3f} | "
            f"{'yes' if row_map.get('alert') else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _axis_pair_groups(pair_ids: Sequence[str]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair_id in pair_ids:
        grouped[axis_id_from_pair_id(pair_id)].add(pair_id)
    return grouped


def _pair_margins(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    direction: np.ndarray,
    selected_pairs: set[str],
    axis_id: str,
) -> list[dict[str, Any]]:
    projections = activations @ direction
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(pair_ids, labels, projections, strict=True):
        if pair_id in selected_pairs:
            grouped[pair_id][label].append(float(projection))
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        label_scores = grouped[pair_id]
        if "positive" not in label_scores or "negative" not in label_scores:
            continue
        margin = float(np.mean(label_scores["positive"])) - float(
            np.mean(label_scores["negative"])
        )
        rows.append(
            {
                "pair_id": pair_id,
                "axis_id": axis_id,
                "margin": round(margin, 6),
                "alert": margin <= 0.0,
            }
        )
    return rows


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
