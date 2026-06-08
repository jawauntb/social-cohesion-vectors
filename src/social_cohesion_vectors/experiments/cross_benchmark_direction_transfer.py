"""Direction-transfer audit across benchmark datasets in one activation space."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.experiments.transfer import load_activation_payload

_EPSILON = 1e-12


def run_cross_benchmark_direction_transfer_from_files(
    *,
    source_vector_npz: str | Path,
    source_activation_npz: str | Path,
    target_vector_npz: str | Path,
    target_activation_npz: str | Path,
    source_name: str = "source",
    target_name: str = "target",
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Load directions and activations, then audit cross-benchmark transfer."""

    source_direction = _load_direction(source_vector_npz)
    target_direction = _load_direction(target_vector_npz)
    source_payload = load_activation_payload(source_activation_npz)
    target_payload = load_activation_payload(target_activation_npz)
    _validate_dimensions(
        source_direction=source_direction,
        target_direction=target_direction,
        source_dim=source_payload.activations.shape[1],
        target_dim=target_payload.activations.shape[1],
    )
    source_self = _pairwise_projection_eval(source_payload, source_direction)
    target_self = _pairwise_projection_eval(target_payload, target_direction)
    source_to_target = _pairwise_projection_eval(target_payload, source_direction)
    target_to_source = _pairwise_projection_eval(source_payload, target_direction)
    cosine = _cosine(source_direction, target_direction)
    readiness = _readiness(
        source_to_target=source_to_target,
        target_to_source=target_to_source,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "cross_benchmark_direction_transfer",
        "description": (
            "Audits whether contrastive directions learned on two benchmark "
            "datasets rank each other's positive and negative examples in the "
            "same activation coordinate space."
        ),
        "inputs": {
            "source_name": source_name,
            "target_name": target_name,
            "source_vector_npz": str(source_vector_npz),
            "source_activation_npz": str(source_activation_npz),
            "target_vector_npz": str(target_vector_npz),
            "target_activation_npz": str(target_activation_npz),
            "activation_dim": int(source_direction.shape[0]),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": {
            "direction_cosine": cosine,
            "source_self_accuracy": source_self["pairwise_accuracy"],
            "source_self_mean_margin": source_self["mean_margin"],
            "source_self_min_margin": source_self["min_margin"],
            "source_self_failed_pairs": source_self["failed_pair_count"],
            "target_self_accuracy": target_self["pairwise_accuracy"],
            "target_self_mean_margin": target_self["mean_margin"],
            "target_self_min_margin": target_self["min_margin"],
            "target_self_failed_pairs": target_self["failed_pair_count"],
            "source_to_target_accuracy": source_to_target["pairwise_accuracy"],
            "source_to_target_mean_margin": source_to_target["mean_margin"],
            "source_to_target_min_margin": source_to_target["min_margin"],
            "source_to_target_failed_pairs": source_to_target["failed_pair_count"],
            "target_to_source_accuracy": target_to_source["pairwise_accuracy"],
            "target_to_source_mean_margin": target_to_source["mean_margin"],
            "target_to_source_min_margin": target_to_source["min_margin"],
            "target_to_source_failed_pairs": target_to_source["failed_pair_count"],
            "transfer_readiness": readiness["status"],
            "ready_for_direction_transfer_claims": readiness["ready"],
        },
        "readiness": readiness,
        "source_self": source_self,
        "target_self": target_self,
        "source_to_target": source_to_target,
        "target_to_source": target_to_source,
        "interpretation_guardrail": (
            "Cross-benchmark direction transfer supports a shared text-benchmark "
            "axis in this activation space. It does not establish a human, "
            "neural, clinical, or deployment claim."
        ),
    }


def save_cross_benchmark_direction_transfer_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown direction-transfer reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_cross_benchmark_direction_transfer_markdown(report),
        encoding="utf-8",
    )


def render_cross_benchmark_direction_transfer_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a cross-benchmark direction-transfer report."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Cross-Benchmark Direction Transfer",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Source benchmark: `{inputs.get('source_name', '')}`",
        f"- Target benchmark: `{inputs.get('target_name', '')}`",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        "",
        "## Summary",
        "",
        f"- Direction cosine: {float(summary.get('direction_cosine', 0.0)):+.3f}",
        f"- Source self accuracy: "
        f"{float(summary.get('source_self_accuracy', 0.0)):.3f}",
        f"- Target self accuracy: "
        f"{float(summary.get('target_self_accuracy', 0.0)):.3f}",
        f"- Source-to-target accuracy: "
        f"{float(summary.get('source_to_target_accuracy', 0.0)):.3f}",
        f"- Source-to-target mean margin: "
        f"{float(summary.get('source_to_target_mean_margin', 0.0)):+.3f}",
        f"- Source-to-target min margin: "
        f"{float(summary.get('source_to_target_min_margin', 0.0)):+.3f}",
        f"- Target-to-source accuracy: "
        f"{float(summary.get('target_to_source_accuracy', 0.0)):.3f}",
        f"- Target-to-source mean margin: "
        f"{float(summary.get('target_to_source_mean_margin', 0.0)):+.3f}",
        f"- Target-to-source min margin: "
        f"{float(summary.get('target_to_source_min_margin', 0.0)):+.3f}",
        f"- Transfer readiness: "
        f"`{summary.get('transfer_readiness', 'not_ready')}`",
        "",
        "## Direction Evaluations",
        "",
        "| Direction | Evaluation set | Pairs | Accuracy | Mean margin | Min margin |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        _evaluation_row(inputs.get("source_name", "source"), "source", report.get("source_self")),
        _evaluation_row(inputs.get("target_name", "target"), "target", report.get("target_self")),
        _evaluation_row(inputs.get("source_name", "source"), "target", report.get("source_to_target")),
        _evaluation_row(inputs.get("target_name", "target"), "source", report.get("target_to_source")),
        "",
        "## Failed Pairs",
        "",
        *_failure_rows(
            source_name=inputs.get("source_name", "source"),
            target_name=inputs.get("target_name", "target"),
            report=report,
        ),
        "",
        "## Interpretation Guardrail",
        "",
        str(report.get("interpretation_guardrail", "")),
        "",
    ]
    return "\n".join(lines)


def _load_direction(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        direction = np.asarray(data["direction"], dtype=np.float64)
    if direction.ndim != 1:
        raise ValueError("direction must be a one-dimensional vector.")
    return direction


def _validate_dimensions(
    *,
    source_direction: np.ndarray,
    target_direction: np.ndarray,
    source_dim: int,
    target_dim: int,
) -> None:
    dims = {
        int(source_direction.shape[0]),
        int(target_direction.shape[0]),
        int(source_dim),
        int(target_dim),
    }
    if len(dims) != 1:
        raise ValueError("All directions and activation payloads must share a dimension.")


def _pairwise_projection_eval(payload: Any, direction: np.ndarray) -> dict[str, Any]:
    projections = np.asarray(payload.activations @ direction, dtype=np.float64)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(
        payload.pair_ids,
        payload.labels,
        projections,
        strict=True,
    ):
        grouped[str(pair_id)][str(label)].append(float(projection))

    pair_rows: list[dict[str, Any]] = []
    margins: list[float] = []
    skipped = 0
    ties = 0
    for pair_id, label_scores in grouped.items():
        if "positive" not in label_scores or "negative" not in label_scores:
            skipped += 1
            continue
        positive_projection = float(np.mean(label_scores["positive"]))
        negative_projection = float(np.mean(label_scores["negative"]))
        margin = positive_projection - negative_projection
        margins.append(margin)
        ties += 1 if margin == 0.0 else 0
        pair_rows.append(
            {
                "pair_id": pair_id,
                "positive_projection": round(positive_projection, 6),
                "negative_projection": round(negative_projection, 6),
                "margin": round(margin, 6),
                "passed": margin > 0.0,
            }
        )
    positive_margins = sum(margin > 0.0 for margin in margins)
    failed_pairs = [row for row in pair_rows if not bool(row["passed"])]
    return {
        "pairs": len(margins),
        "pairwise_accuracy": round(positive_margins / len(margins), 6)
        if margins
        else 0.0,
        "mean_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_margin": round(float(np.max(margins)), 6) if margins else 0.0,
        "skipped_pairs": skipped,
        "ties": ties,
        "failed_pair_count": len(failed_pairs),
        "failed_pairs": failed_pairs,
        "pair_margins": pair_rows,
    }


def _readiness(
    *,
    source_to_target: Mapping[str, Any],
    target_to_source: Mapping[str, Any],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    gates = [
        _gate(
            "source_to_target_pairwise_accuracy",
            float(source_to_target.get("pairwise_accuracy", 0.0)),
            min_pairwise_accuracy,
        ),
        _gate(
            "target_to_source_pairwise_accuracy",
            float(target_to_source.get("pairwise_accuracy", 0.0)),
            min_pairwise_accuracy,
        ),
        _gate(
            "source_to_target_min_margin",
            float(source_to_target.get("min_margin", 0.0)),
            min_margin,
        ),
        _gate(
            "target_to_source_min_margin",
            float(target_to_source.get("min_margin", 0.0)),
            min_margin,
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "direction_transfer_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _gate(gate_id: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "value": round(value, 6),
        "threshold": round(threshold, 6),
        "passed": value >= threshold,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPSILON:
        return 0.0
    return round(float(left @ right / denominator), 6)


def _evaluation_row(
    direction_name: object,
    evaluation_name: str,
    raw_eval: object,
) -> str:
    evaluation = _mapping(raw_eval)
    return (
        "| "
        f"`{direction_name}` | "
        f"`{evaluation_name}` | "
        f"{int(evaluation.get('pairs', 0))} | "
        f"{float(evaluation.get('pairwise_accuracy', 0.0)):.3f} | "
        f"{float(evaluation.get('mean_margin', 0.0)):+.3f} | "
        f"{float(evaluation.get('min_margin', 0.0)):+.3f} |"
    )


def _failure_rows(
    *,
    source_name: object,
    target_name: object,
    report: Mapping[str, Any],
) -> list[str]:
    rows: list[str] = [
        "| Direction | Evaluation set | Pair | Margin | Positive projection | Negative projection |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    specs = (
        (source_name, "source", report.get("source_self")),
        (target_name, "target", report.get("target_self")),
        (source_name, "target", report.get("source_to_target")),
        (target_name, "source", report.get("target_to_source")),
    )
    for direction_name, evaluation_name, raw_eval in specs:
        failed_pairs = _sequence(_mapping(raw_eval).get("failed_pairs"))
        for raw_pair in failed_pairs:
            pair = _mapping(raw_pair)
            rows.append(
                "| "
                f"`{direction_name}` | "
                f"`{evaluation_name}` | "
                f"`{pair.get('pair_id', '')}` | "
                f"{float(pair.get('margin', 0.0)):+.3f} | "
                f"{float(pair.get('positive_projection', 0.0)):+.3f} | "
                f"{float(pair.get('negative_projection', 0.0)):+.3f} |"
            )
    if len(rows) == 2:
        return ["No failed pairs."]
    return rows


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []
