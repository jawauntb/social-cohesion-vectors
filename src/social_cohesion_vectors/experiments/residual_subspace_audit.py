"""Residual and subspace audits for contrastive activation claims."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.transfer import load_activation_payload
from social_cohesion_vectors.schemas import PairwiseExample

_EPSILON = 1e-12


def run_residual_subspace_audit_from_files(
    *,
    activation_npz: str | Path,
    pairs_path: str | Path,
    metadata_key: str = "primary_fault_class",
) -> dict[str, Any]:
    """Load activations and pairs, then run a residual subspace audit."""

    return run_residual_subspace_audit(
        activation_npz=activation_npz,
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        pairs_path=str(pairs_path),
    )


def run_residual_subspace_audit(
    *,
    activation_npz: str | Path,
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    pairs_path: str | None = None,
) -> dict[str, Any]:
    """Audit what remains after projecting out the global contrastive direction."""

    payload = load_activation_payload(activation_npz)
    pair_ids = [str(pair_id) for pair_id in payload.pair_ids]
    labels = [str(label) for label in payload.labels]
    global_direction = _safe_train_direction(payload.activations, labels)
    global_report = _evaluate_optional_direction(
        activations=payload.activations,
        pair_ids=pair_ids,
        labels=labels,
        direction=global_direction,
    )
    residual_activations = (
        _project_out(payload.activations, global_direction)
        if global_direction is not None
        else payload.activations.copy()
    )
    residual_global_direction = _safe_train_direction(residual_activations, labels)
    residual_global_report = _evaluate_optional_direction(
        activations=residual_activations,
        pair_ids=pair_ids,
        labels=labels,
        direction=residual_global_direction,
    )
    group_reports = _residual_group_reports(
        residual_activations=residual_activations,
        pair_ids=pair_ids,
        labels=labels,
        pairs=pairs,
        metadata_key=metadata_key,
    )
    difference_report = _pair_difference_subspace_report(
        activations=payload.activations,
        pair_ids=pair_ids,
        labels=labels,
        global_direction=global_direction,
    )
    return {
        "experiment": "residual_subspace_audit",
        "description": (
            "Projects out the global contrastive direction, then asks whether "
            "global or metadata-group residual directions still separate pairs."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": pairs_path,
            "metadata_key": metadata_key,
            "pairs": len(pairs),
            "prompts": int(payload.activations.shape[0]),
            "activation_dim": int(payload.activations.shape[1]),
        },
        "summary": _residual_summary(global_report, residual_global_report, group_reports),
        "global_direction": global_report,
        "residual_global_direction": residual_global_report,
        "residual_group_directions": group_reports,
        "pair_difference_subspace": difference_report,
    }


def save_residual_subspace_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown residual subspace reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_residual_subspace_audit_markdown(report),
        encoding="utf-8",
    )


def render_residual_subspace_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render residual audit output as markdown."""

    summary = _mapping(report.get("summary"))
    inputs = _mapping(report.get("inputs"))
    subspace = _mapping(report.get("pair_difference_subspace"))
    lines = [
        "# Residual Subspace Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Metadata key: `{inputs.get('metadata_key', '')}`",
        f"- Pairs: {int(inputs.get('pairs', 0))}",
        f"- Prompts: {int(inputs.get('prompts', 0))}",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        "",
        "## Summary",
        "",
        f"- Global accuracy before projection: "
        f"{float(summary.get('global_accuracy', 0.0)):.3f}",
        f"- Residual global accuracy: "
        f"{float(summary.get('residual_global_accuracy', 0.0)):.3f}",
        f"- Residual group mean accuracy: "
        f"{float(summary.get('residual_group_mean_accuracy', 0.0)):.3f}",
        f"- Residual groups with positive signal: "
        f"{int(summary.get('residual_groups_with_positive_signal', 0))}",
        f"- Pair-difference energy captured by global direction: "
        f"{float(subspace.get('global_direction_energy_fraction', 0.0)):.3f}",
        f"- Residual pair-difference energy after global projection: "
        f"{float(subspace.get('residual_energy_fraction', 0.0)):.3f}",
        "",
        "## Interpretation Guardrail",
        "",
        (
            "If residual group directions still separate after the global "
            "direction is removed, a single-direction ablation does not show "
            "that the representational signal is exhausted."
        ),
        "",
        "## Residual Group Directions",
        "",
        "| Group | Pairs | Accuracy | Mean margin | Min margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("residual_group_directions")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('min_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _residual_group_reports(
    *,
    residual_activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> list[dict[str, Any]]:
    group_pairs = _group_pair_ids(pairs, metadata_key)
    reports: list[dict[str, Any]] = []
    label_array = np.asarray(labels, dtype=str)
    for group, selected_pairs in sorted(group_pairs.items()):
        mask = np.asarray([pair_id in selected_pairs for pair_id in pair_ids], dtype=bool)
        direction = _safe_train_direction(
            residual_activations[mask],
            [str(label) for label in label_array[mask]],
        )
        report = _evaluate_optional_direction(
            activations=residual_activations,
            pair_ids=pair_ids,
            labels=labels,
            direction=direction,
            selected_pairs=selected_pairs,
        )
        report.update({"group": group, "pairs": len(selected_pairs)})
        reports.append(report)
    reports.sort(key=lambda row: float(row["mean_margin"]), reverse=True)
    return reports


def _pair_difference_subspace_report(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    global_direction: np.ndarray | None,
) -> dict[str, Any]:
    differences = _pair_difference_matrix(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
    )
    total_energy = float(np.sum(differences**2))
    if differences.size == 0 or total_energy <= _EPSILON:
        return {
            "pairs": 0,
            "total_pair_difference_energy": 0.0,
            "global_direction_energy_fraction": 0.0,
            "residual_energy_fraction": 0.0,
            "svd_cumulative_energy": [],
            "residual_svd_cumulative_energy": [],
        }
    global_fraction = (
        _direction_energy_fraction(differences, global_direction)
        if global_direction is not None
        else 0.0
    )
    residual_differences = (
        _project_out(differences, global_direction)
        if global_direction is not None
        else differences.copy()
    )
    return {
        "pairs": int(differences.shape[0]),
        "total_pair_difference_energy": round(total_energy, 6),
        "global_direction_energy_fraction": global_fraction,
        "residual_energy_fraction": round(float(np.sum(residual_differences**2)) / total_energy, 6),
        "svd_cumulative_energy": _svd_cumulative_energy(differences),
        "residual_svd_cumulative_energy": _svd_cumulative_energy(residual_differences),
    }


def _evaluate_optional_direction(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    direction: np.ndarray | None,
    selected_pairs: set[str] | None = None,
) -> dict[str, Any]:
    if direction is None:
        return {
            "available": False,
            "pairs": 0,
            "accuracy": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
        }
    margins = _pair_margins(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        direction=direction,
        selected_pairs=selected_pairs,
    )
    outcomes = [1.0 if margin > 0 else 0.5 if margin == 0 else 0.0 for margin in margins]
    return {
        "available": True,
        "pairs": len(margins),
        "accuracy": round(sum(outcomes) / len(outcomes), 6) if outcomes else 0.0,
        "mean_margin": _mean(margins),
        "min_margin": round(min(margins), 6) if margins else 0.0,
    }


def _pair_margins(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    direction: np.ndarray,
    selected_pairs: set[str] | None = None,
) -> list[float]:
    projections = activations @ direction
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(pair_ids, labels, projections, strict=True):
        if selected_pairs is None or pair_id in selected_pairs:
            grouped[pair_id][label].append(float(projection))
    margins: list[float] = []
    for label_scores in grouped.values():
        if "positive" in label_scores and "negative" in label_scores:
            margins.append(
                float(np.mean(label_scores["positive"]))
                - float(np.mean(label_scores["negative"]))
            )
    return margins


def _pair_difference_matrix(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, activation in zip(pair_ids, labels, activations, strict=True):
        grouped[pair_id][label].append(np.asarray(activation, dtype=np.float64))
    rows: list[np.ndarray] = []
    for label_vectors in grouped.values():
        if "positive" in label_vectors and "negative" in label_vectors:
            positive = np.vstack(label_vectors["positive"]).mean(axis=0)
            negative = np.vstack(label_vectors["negative"]).mean(axis=0)
            rows.append(positive - negative)
    if not rows:
        return np.empty((0, activations.shape[1]), dtype=np.float64)
    return np.vstack(rows)


def _safe_train_direction(
    activations: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray | None:
    if activations.shape[0] < 2:
        return None
    try:
        return train_direction_from_arrays(activations, labels=labels).direction
    except ValueError:
        return None


def _project_out(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return matrix - np.outer(matrix @ direction, direction)


def _direction_energy_fraction(
    differences: np.ndarray,
    direction: np.ndarray,
) -> float:
    denominator = float(np.sum(differences**2))
    if denominator <= _EPSILON:
        return 0.0
    numerator = float(np.sum((differences @ direction) ** 2))
    return round(numerator / denominator, 6)


def _svd_cumulative_energy(differences: np.ndarray, max_components: int = 8) -> list[dict[str, Any]]:
    if differences.size == 0:
        return []
    singular_values = np.linalg.svd(differences, full_matrices=False, compute_uv=False)
    energies = singular_values**2
    total = float(np.sum(energies))
    if total <= _EPSILON:
        return []
    rows: list[dict[str, Any]] = []
    running = 0.0
    for index, energy in enumerate(energies[:max_components], start=1):
        running += float(energy)
        rows.append(
            {
                "components": index,
                "singular_value": round(float(singular_values[index - 1]), 6),
                "cumulative_energy_fraction": round(running / total, 6),
            }
        )
    return rows


def _residual_summary(
    global_report: Mapping[str, Any],
    residual_global_report: Mapping[str, Any],
    group_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    group_accuracies = [float(row.get("accuracy", 0.0)) for row in group_reports]
    positive_signal_groups = [
        row
        for row in group_reports
        if float(row.get("accuracy", 0.0)) >= 0.75
        and float(row.get("mean_margin", 0.0)) > 0.0
    ]
    return {
        "global_accuracy": float(global_report.get("accuracy", 0.0)),
        "global_mean_margin": float(global_report.get("mean_margin", 0.0)),
        "residual_global_accuracy": float(residual_global_report.get("accuracy", 0.0)),
        "residual_global_mean_margin": float(
            residual_global_report.get("mean_margin", 0.0)
        ),
        "residual_group_mean_accuracy": _mean(group_accuracies),
        "residual_groups_with_positive_signal": len(positive_signal_groups),
        "residual_groups": len(group_reports),
    }


def _group_pair_ids(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        raw = pair.metadata.get(metadata_key)
        values = tuple(part.strip() for part in str(raw or "").split(",") if part.strip())
        for value in values:
            grouped[value].add(pair.pair_id)
    return dict(grouped)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
