"""Audit directions augmented with fresh generated examples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.transfer import load_activation_payload
from social_cohesion_vectors.schemas import PairwiseExample


@dataclass(frozen=True)
class _Dataset:
    name: str
    activations: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    pairs: tuple[PairwiseExample, ...]


def run_fresh_augmented_direction_audit_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    fresh_source_activation_npz: str | Path,
    fresh_source_pairs_path: str | Path,
    fresh_target_activation_npz: str | Path,
    fresh_target_pairs_path: str | Path,
    model_name: str = "model",
    source_name: str = "source",
    target_name: str = "target",
    fresh_source_name: str = "fresh_source",
    fresh_target_name: str = "fresh_target",
    augmentation_pair_ids: Sequence[str] | None = None,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Load activation payloads and run the fresh-augmented direction audit."""

    source = _load_dataset(
        name=source_name,
        activation_npz=source_activation_npz,
        pairs_path=source_pairs_path,
    )
    target = _load_dataset(
        name=target_name,
        activation_npz=target_activation_npz,
        pairs_path=target_pairs_path,
    )
    fresh_source = _load_dataset(
        name=fresh_source_name,
        activation_npz=fresh_source_activation_npz,
        pairs_path=fresh_source_pairs_path,
    )
    fresh_target = _load_dataset(
        name=fresh_target_name,
        activation_npz=fresh_target_activation_npz,
        pairs_path=fresh_target_pairs_path,
    )
    return run_fresh_augmented_direction_audit(
        source=source,
        target=target,
        fresh_source=fresh_source,
        fresh_target=fresh_target,
        model_name=model_name,
        augmentation_pair_ids=augmentation_pair_ids,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
        input_paths={
            "source_activation_npz": str(source_activation_npz),
            "source_pairs": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs": str(target_pairs_path),
            "fresh_source_activation_npz": str(fresh_source_activation_npz),
            "fresh_source_pairs": str(fresh_source_pairs_path),
            "fresh_target_activation_npz": str(fresh_target_activation_npz),
            "fresh_target_pairs": str(fresh_target_pairs_path),
        },
    )


def run_fresh_augmented_direction_audit(
    *,
    source: _Dataset,
    target: _Dataset,
    fresh_source: _Dataset,
    fresh_target: _Dataset,
    model_name: str = "model",
    augmentation_pair_ids: Sequence[str] | None = None,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
    input_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Train original and fresh-augmented directions, including fresh LOO."""

    _validate_shared_dim(source, target, fresh_source, fresh_target)
    source_pair_ids = _unique_pair_ids(source)
    target_pair_ids = _unique_pair_ids(target)
    fresh_pair_ids = _unique_pair_ids(fresh_source)
    requested_augmentation = set(augmentation_pair_ids or fresh_pair_ids)
    missing = sorted(requested_augmentation - fresh_pair_ids)
    if missing:
        msg = f"augmentation_pair_ids missing from fresh source activations: {missing[:5]}"
        raise ValueError(msg)
    augmentation_pair_set = requested_augmentation

    original_joint_direction = _direction_from_datasets(
        (source, source_pair_ids),
        (target, target_pair_ids),
    )
    full_augmented_direction = _direction_from_datasets(
        (source, source_pair_ids),
        (target, target_pair_ids),
        (fresh_source, augmentation_pair_set),
    )
    baseline_rows = [
        _direction_row(
            direction_id="original_source_target_joint",
            direction_family="baseline",
            direction=original_joint_direction,
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
        ),
        _direction_row(
            direction_id="full_fresh_augmented",
            direction_family="fresh_augmented",
            direction=full_augmented_direction,
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
        ),
    ]
    loo_rows = [
        _fresh_leave_one_out_row(
            held_out_pair_id=held_out_pair_id,
            training_pair_ids=augmentation_pair_set - {held_out_pair_id},
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
        )
        for held_out_pair_id in sorted(augmentation_pair_set)
    ]
    readiness = _readiness(
        baseline_rows=baseline_rows,
        loo_rows=loo_rows,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "fresh_augmented_direction_audit",
        "description": (
            "Tests whether adding fresh generated examples to the source+target "
            "training direction repairs fresh generated residuals while "
            "preserving original generated and hand-authored control margins."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "model_name": model_name,
            "source_pairs": len(source_pair_ids),
            "target_pairs": len(target_pair_ids),
            "fresh_source_pairs": len(fresh_pair_ids),
            "fresh_target_pairs": len(_unique_pair_ids(fresh_target)),
            "augmentation_pairs": len(augmentation_pair_set),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": _summary(baseline_rows=baseline_rows, loo_rows=loo_rows, readiness=readiness),
        "readiness": readiness,
        "direction_evaluations": baseline_rows,
        "fresh_leave_one_out": loo_rows,
        "interpretation_guardrail": (
            "Fresh-augmented direction audits are text-benchmark activation "
            "diagnostics. They do not establish causal steering, human "
            "behavioral, neural, clinical, deployment, or real-world "
            "social-effect claims."
        ),
    }


def save_fresh_augmented_direction_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fresh_augmented_direction_audit_markdown(report),
        encoding="utf-8",
    )


def render_fresh_augmented_direction_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render fresh-augmented audit results as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Fresh-Augmented Direction Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Model: `{inputs.get('model_name', '')}`",
        f"- Source pairs: {int(inputs.get('source_pairs', 0))}",
        f"- Target pairs: {int(inputs.get('target_pairs', 0))}",
        f"- Fresh source pairs: {int(inputs.get('fresh_source_pairs', 0))}",
        f"- Augmentation pairs: {int(inputs.get('augmentation_pairs', 0))}",
        "",
        "## Summary",
        "",
        f"- Readiness: `{summary.get('readiness', 'unknown')}`",
        f"- Full augmented source min margin: "
        f"{float(summary.get('full_augmented_source_min_margin', 0.0)):+.3f}",
        f"- Full augmented target min margin: "
        f"{float(summary.get('full_augmented_target_min_margin', 0.0)):+.3f}",
        f"- Full augmented fresh source min margin: "
        f"{float(summary.get('full_augmented_fresh_source_min_margin', 0.0)):+.3f}",
        f"- Full augmented fresh target min margin: "
        f"{float(summary.get('full_augmented_fresh_target_min_margin', 0.0)):+.3f}",
        f"- Fresh LOO held-out min margin: "
        f"{float(summary.get('fresh_loo_heldout_min_margin', 0.0)):+.3f}",
        f"- Fresh LOO failed held-out pairs: "
        f"{int(summary.get('fresh_loo_failed_heldout_pairs', 0))}",
        "",
        "## Direction Evaluations",
        "",
        *_direction_table(report.get("direction_evaluations")),
        "",
        "## Fresh Leave-One-Out",
        "",
        *_loo_table(report.get("fresh_leave_one_out")),
        "",
        "## Interpretation Guardrail",
        "",
        str(report.get("interpretation_guardrail", "")),
        "",
    ]
    return "\n".join(lines)


def _load_dataset(
    *,
    name: str,
    activation_npz: str | Path,
    pairs_path: str | Path,
) -> _Dataset:
    payload = load_activation_payload(activation_npz)
    pairs = tuple(load_pairwise_examples_jsonl(pairs_path))
    pair_ids = {pair.pair_id for pair in pairs}
    activation_pair_ids = {str(pair_id) for pair_id in payload.pair_ids}
    missing_metadata = sorted(activation_pair_ids - pair_ids)
    missing_activations = sorted(pair_ids - activation_pair_ids)
    if missing_metadata or missing_activations:
        msg = (
            f"Activation/pair metadata mismatch for {name}: "
            f"missing_metadata={missing_metadata[:5]} "
            f"missing_activations={missing_activations[:5]}"
        )
        raise ValueError(msg)
    return _Dataset(
        name=name,
        activations=np.asarray(payload.activations, dtype=np.float64),
        labels=np.asarray(payload.labels, dtype=str),
        pair_ids=np.asarray(payload.pair_ids, dtype=str),
        pairs=pairs,
    )


def _validate_shared_dim(*datasets: _Dataset) -> None:
    dims = {int(dataset.activations.shape[1]) for dataset in datasets}
    if len(dims) != 1:
        msg = f"Activation dimensions must match; got {sorted(dims)}."
        raise ValueError(msg)


def _direction_from_datasets(*parts: tuple[_Dataset, set[str]]) -> np.ndarray:
    activation_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for dataset, pair_ids in parts:
        mask = _mask(dataset, pair_ids)
        activation_parts.append(dataset.activations[mask])
        label_parts.append(dataset.labels[mask])
    activations = np.concatenate(activation_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    return train_direction_from_arrays(activations, labels=labels).direction


def _fresh_leave_one_out_row(
    *,
    held_out_pair_id: str,
    training_pair_ids: set[str],
    source: _Dataset,
    target: _Dataset,
    fresh_source: _Dataset,
    fresh_target: _Dataset,
) -> dict[str, Any]:
    direction = _direction_from_datasets(
        (source, _unique_pair_ids(source)),
        (target, _unique_pair_ids(target)),
        (fresh_source, training_pair_ids),
    )
    return {
        "fold_id": f"fresh_loo:{held_out_pair_id}",
        "held_out_pair_id": held_out_pair_id,
        "training_fresh_pairs": len(training_pair_ids),
        "held_out_evaluation": _evaluate(
            fresh_source,
            direction=direction,
            pair_ids={held_out_pair_id},
        ),
        "on_source": _evaluate(source, direction=direction, pair_ids=_unique_pair_ids(source)),
        "on_target": _evaluate(target, direction=direction, pair_ids=_unique_pair_ids(target)),
        "on_fresh_source": _evaluate(
            fresh_source,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_source),
        ),
        "on_fresh_target": _evaluate(
            fresh_target,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_target),
        ),
    }


def _direction_row(
    *,
    direction_id: str,
    direction_family: str,
    direction: np.ndarray,
    source: _Dataset,
    target: _Dataset,
    fresh_source: _Dataset,
    fresh_target: _Dataset,
) -> dict[str, Any]:
    return {
        "direction_id": direction_id,
        "direction_family": direction_family,
        "on_source": _evaluate(source, direction=direction, pair_ids=_unique_pair_ids(source)),
        "on_target": _evaluate(target, direction=direction, pair_ids=_unique_pair_ids(target)),
        "on_fresh_source": _evaluate(
            fresh_source,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_source),
        ),
        "on_fresh_target": _evaluate(
            fresh_target,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_target),
        ),
    }


def _evaluate(
    dataset: _Dataset,
    *,
    direction: np.ndarray,
    pair_ids: set[str],
) -> dict[str, Any]:
    mask = _mask(dataset, pair_ids)
    projections = np.asarray(dataset.activations[mask] @ direction, dtype=np.float64)
    grouped: dict[str, dict[str, list[float]]] = {}
    for pair_id, label, projection in zip(
        dataset.pair_ids[mask],
        dataset.labels[mask],
        projections,
        strict=True,
    ):
        labels = grouped.setdefault(str(pair_id), {})
        labels.setdefault(str(label), []).append(float(projection))

    pair_margins: list[dict[str, Any]] = []
    for pair_id, label_scores in sorted(grouped.items()):
        positive = _mean(label_scores.get("positive", ()))
        negative = _mean(label_scores.get("negative", ()))
        margin = round(positive - negative, 6)
        pair_margins.append(
            {
                "pair_id": pair_id,
                "positive_projection": round(positive, 6),
                "negative_projection": round(negative, 6),
                "margin": margin,
                "passed": margin > 0.0,
            }
        )
    margins = [float(row["margin"]) for row in pair_margins]
    failed = [row for row in pair_margins if not bool(row["passed"])]
    return {
        "test_pairs": len(pair_margins),
        "pairwise_accuracy": _ratio(len(pair_margins) - len(failed), len(pair_margins)),
        "mean_margin": round(sum(margins) / len(margins), 6) if margins else 0.0,
        "min_margin": round(min(margins), 6) if margins else 0.0,
        "failed_pair_count": len(failed),
        "failed_pairs": failed,
        "pair_margins": pair_margins,
    }


def _readiness(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    loo_rows: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    full_augmented = next(
        row for row in baseline_rows if row.get("direction_id") == "full_fresh_augmented"
    )
    gates = [
        _gate(
            "full_augmented_source_accuracy",
            _metric(full_augmented, "on_source", "pairwise_accuracy"),
            min_pairwise_accuracy,
        ),
        _gate(
            "full_augmented_source_margin",
            _metric(full_augmented, "on_source", "min_margin"),
            min_margin,
        ),
        _gate(
            "full_augmented_target_accuracy",
            _metric(full_augmented, "on_target", "pairwise_accuracy"),
            min_pairwise_accuracy,
        ),
        _gate(
            "full_augmented_target_margin",
            _metric(full_augmented, "on_target", "min_margin"),
            min_margin,
        ),
        _gate(
            "full_augmented_fresh_source_accuracy",
            _metric(full_augmented, "on_fresh_source", "pairwise_accuracy"),
            min_pairwise_accuracy,
        ),
        _gate(
            "full_augmented_fresh_source_margin",
            _metric(full_augmented, "on_fresh_source", "min_margin"),
            min_margin,
        ),
        _gate(
            "full_augmented_fresh_target_accuracy",
            _metric(full_augmented, "on_fresh_target", "pairwise_accuracy"),
            min_pairwise_accuracy,
        ),
        _gate(
            "full_augmented_fresh_target_margin",
            _metric(full_augmented, "on_fresh_target", "min_margin"),
            min_margin,
        ),
        _gate(
            "fresh_loo_heldout_accuracy",
            min(
                (
                    _metric(row, "held_out_evaluation", "pairwise_accuracy")
                    for row in loo_rows
                ),
                default=0.0,
            ),
            min_pairwise_accuracy,
        ),
        _gate(
            "fresh_loo_heldout_margin",
            min(
                (_metric(row, "held_out_evaluation", "min_margin") for row in loo_rows),
                default=0.0,
            ),
            min_margin,
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "fresh_augmented_direction_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _summary(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    loo_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    full_augmented = next(
        row for row in baseline_rows if row.get("direction_id") == "full_fresh_augmented"
    )
    held_out_margins = [
        _metric(row, "held_out_evaluation", "min_margin") for row in loo_rows
    ]
    failed_held_out = [
        row
        for row in loo_rows
        if _metric(row, "held_out_evaluation", "pairwise_accuracy") < 1.0
    ]
    return {
        "readiness": str(readiness.get("status", "")),
        "ready_for_fresh_augmented_direction_claims": bool(
            readiness.get("ready", False)
        ),
        "full_augmented_source_min_margin": _metric(
            full_augmented,
            "on_source",
            "min_margin",
        ),
        "full_augmented_target_min_margin": _metric(
            full_augmented,
            "on_target",
            "min_margin",
        ),
        "full_augmented_fresh_source_min_margin": _metric(
            full_augmented,
            "on_fresh_source",
            "min_margin",
        ),
        "full_augmented_fresh_target_min_margin": _metric(
            full_augmented,
            "on_fresh_target",
            "min_margin",
        ),
        "fresh_loo_heldout_min_margin": (
            round(min(held_out_margins), 6) if held_out_margins else 0.0
        ),
        "fresh_loo_failed_heldout_pairs": len(failed_held_out),
    }


def _gate(gate_id: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "value": round(float(value), 6),
        "threshold": round(float(threshold), 6),
        "passed": float(value) >= float(threshold),
    }


def _metric(row: Mapping[str, Any], evaluation_key: str, metric_key: str) -> float:
    return float(_mapping(row.get(evaluation_key)).get(metric_key, 0.0))


def _mask(dataset: _Dataset, pair_ids: set[str]) -> np.ndarray:
    return np.asarray([str(pair_id) in pair_ids for pair_id in dataset.pair_ids], dtype=bool)


def _unique_pair_ids(dataset: _Dataset) -> set[str]:
    return {str(pair_id) for pair_id in dataset.pair_ids}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _direction_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No direction rows."]
    lines = [
        "| Direction | Family | Source | Target | Fresh source | Fresh target |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('direction_id', '')}` | "
            f"`{row.get('direction_family', '')}` | "
            f"{_eval_cell(row, 'on_source')} | "
            f"{_eval_cell(row, 'on_target')} | "
            f"{_eval_cell(row, 'on_fresh_source')} | "
            f"{_eval_cell(row, 'on_fresh_target')} |"
        )
    return lines


def _loo_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No fresh leave-one-out rows."]
    lines = [
        "| Held-out pair | Held-out | Source | Target | Fresh source | Fresh target |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('held_out_pair_id', '')}` | "
            f"{_eval_cell(row, 'held_out_evaluation')} | "
            f"{_eval_cell(row, 'on_source')} | "
            f"{_eval_cell(row, 'on_target')} | "
            f"{_eval_cell(row, 'on_fresh_source')} | "
            f"{_eval_cell(row, 'on_fresh_target')} |"
        )
    return lines


def _eval_cell(row: Mapping[str, Any], key: str) -> str:
    evaluation = _mapping(row.get(key))
    return (
        f"{float(evaluation.get('pairwise_accuracy', 0.0)):.3f}/"
        f"{float(evaluation.get('min_margin', 0.0)):+.3f}/"
        f"{int(evaluation.get('failed_pair_count', 0))}"
    )


def _mapping(raw_value: object) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _sequence(raw_value: object) -> list[Any]:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str):
        return list(raw_value)
    return []
