"""Cross-model representation alignment audits for activation directions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays

_EPSILON = 1e-12


@dataclass(frozen=True)
class CrossModelPayload:
    """Activation payload aligned by prompt/sample identity."""

    path: str
    activations: np.ndarray
    pair_ids: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


def run_cross_model_alignment_audit_from_files(
    *,
    source_activation_npz: str | Path,
    target_activation_npz: str | Path,
    knn_k: int = 10,
    ridge_alpha: float = 1e-3,
    max_leave_one_pair_out_pairs: int | None = None,
) -> dict[str, Any]:
    """Load two activation payloads and audit representation transfer."""

    source, target = _load_aligned_payloads(
        source_activation_npz,
        target_activation_npz,
    )
    return run_cross_model_alignment_audit(
        source=source,
        target=target,
        knn_k=knn_k,
        ridge_alpha=ridge_alpha,
        max_leave_one_pair_out_pairs=max_leave_one_pair_out_pairs,
    )


def run_cross_model_alignment_audit(
    *,
    source: CrossModelPayload,
    target: CrossModelPayload,
    knn_k: int = 10,
    ridge_alpha: float = 1e-3,
    max_leave_one_pair_out_pairs: int | None = None,
) -> dict[str, Any]:
    """Measure whether two model spaces share prompt geometry and directions.

    This is a Platonic-Representation-inspired diagnostic, not an ontology
    claim: it asks whether a simple paired linear map transfers one discovered
    contrastive direction into another model's hidden-state coordinate system.
    """

    _validate_aligned_payloads(source, target)
    alignment = _alignment_metrics(
        source.activations,
        target.activations,
        knn_k=knn_k,
    )
    source_to_target = _direction_transfer(
        source=source,
        target=target,
        ridge_alpha=ridge_alpha,
        max_leave_one_pair_out_pairs=max_leave_one_pair_out_pairs,
    )
    target_to_source = _direction_transfer(
        source=target,
        target=source,
        ridge_alpha=ridge_alpha,
        max_leave_one_pair_out_pairs=max_leave_one_pair_out_pairs,
    )
    return {
        "experiment": "cross_model_alignment_audit",
        "description": (
            "Audits prompt-neighborhood alignment and linear direction transfer "
            "between two activation spaces for the same paired prompts."
        ),
        "inputs": {
            "source_activation_npz": source.path,
            "target_activation_npz": target.path,
            "shared_samples": int(source.activations.shape[0]),
            "shared_pairs": int(len(set(source.pair_ids.tolist()))),
            "source_dim": int(source.activations.shape[1]),
            "target_dim": int(target.activations.shape[1]),
            "knn_k": int(alignment["effective_knn_k"]),
            "ridge_alpha": float(ridge_alpha),
            "max_leave_one_pair_out_pairs": max_leave_one_pair_out_pairs,
        },
        "summary": _summary(alignment, source_to_target, target_to_source),
        "alignment": alignment,
        "source_to_target_direction_transfer": source_to_target,
        "target_to_source_direction_transfer": target_to_source,
        "interpretation_guardrail": (
            "Treat high alignment as evidence of cross-model representational "
            "compatibility for this synthetic contrast, not as evidence that an "
            "ideal social-cohesion representation has been extracted."
        ),
    }


def save_cross_model_alignment_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown cross-model alignment reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_cross_model_alignment_audit_markdown(report),
        encoding="utf-8",
    )


def render_cross_model_alignment_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact reviewer-facing alignment report."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    source_to_target = _mapping(report.get("source_to_target_direction_transfer"))
    target_to_source = _mapping(report.get("target_to_source_direction_transfer"))
    source_to_target_loo = _mapping(
        source_to_target.get("leave_one_pair_out_mapped_source_direction")
    )
    target_to_source_loo = _mapping(
        target_to_source.get("leave_one_pair_out_mapped_source_direction")
    )
    lines = [
        "# Cross-Model Alignment Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Source: `{Path(str(inputs.get('source_activation_npz', ''))).name}`",
        f"- Target: `{Path(str(inputs.get('target_activation_npz', ''))).name}`",
        f"- Shared samples: {int(inputs.get('shared_samples', 0))}",
        f"- Shared pairs: {int(inputs.get('shared_pairs', 0))}",
        f"- Source dim: {int(inputs.get('source_dim', 0))}",
        f"- Target dim: {int(inputs.get('target_dim', 0))}",
        f"- Pair-LOO fold cap: {inputs.get('max_leave_one_pair_out_pairs')}",
        "",
        "## Summary",
        "",
        f"- Linear CKA: {float(summary.get('linear_cka', 0.0)):.3f}",
        f"- Mutual kNN overlap: {float(summary.get('mutual_knn_overlap', 0.0)):.3f}",
        f"- Source-to-target mapped accuracy: "
        f"{float(summary.get('source_to_target_mapped_accuracy', 0.0)):.3f}",
        f"- Source-to-target pair-LOO mapped accuracy: "
        f"{float(summary.get('source_to_target_loo_mapped_accuracy', 0.0)):.3f}",
        f"- Source-to-target pair-LOO folds: "
        f"{int(source_to_target_loo.get('pairs', 0))}/"
        f"{int(source_to_target_loo.get('available_pairs', 0))}",
        f"- Source-to-target pair-LOO mapped mean margin: "
        f"{float(summary.get('source_to_target_loo_mapped_mean_margin', 0.0)):+.3f}",
        f"- Source-to-target mapped mean margin: "
        f"{float(summary.get('source_to_target_mapped_mean_margin', 0.0)):+.3f}",
        f"- Source-to-target cosine with target direction: "
        f"{float(summary.get('source_to_target_direction_cosine', 0.0)):+.3f}",
        f"- Target-to-source mapped accuracy: "
        f"{float(summary.get('target_to_source_mapped_accuracy', 0.0)):.3f}",
        f"- Target-to-source pair-LOO mapped accuracy: "
        f"{float(summary.get('target_to_source_loo_mapped_accuracy', 0.0)):.3f}",
        f"- Target-to-source pair-LOO folds: "
        f"{int(target_to_source_loo.get('pairs', 0))}/"
        f"{int(target_to_source_loo.get('available_pairs', 0))}",
        f"- Target-to-source pair-LOO mapped mean margin: "
        f"{float(summary.get('target_to_source_loo_mapped_mean_margin', 0.0)):+.3f}",
        f"- Target-to-source mapped mean margin: "
        f"{float(summary.get('target_to_source_mapped_mean_margin', 0.0)):+.3f}",
        f"- Target-to-source cosine with source direction: "
        f"{float(summary.get('target_to_source_direction_cosine', 0.0)):+.3f}",
        "",
        "## Direction Transfer",
        "",
        "| Direction | Target self accuracy | In-sample mapped accuracy | Pair-LOO mapped accuracy | Pair-LOO mean margin | Cosine to target direction | Oracle-sign accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _transfer_row("source -> target", source_to_target),
        _transfer_row("target -> source", target_to_source),
        "",
        "## Interpretation Guardrail",
        "",
        str(report.get("interpretation_guardrail", "")),
        "",
    ]
    return "\n".join(lines)


def _load_aligned_payloads(
    source_path: str | Path,
    target_path: str | Path,
) -> tuple[CrossModelPayload, CrossModelPayload]:
    source = _load_payload(source_path)
    target = _load_payload(target_path)
    target_by_key = {
        key: index for index, key in enumerate(target.sample_ids.astype(str).tolist())
    }
    source_rows: list[int] = []
    target_rows: list[int] = []
    for index, key in enumerate(source.sample_ids.astype(str).tolist()):
        target_index = target_by_key.get(key)
        if target_index is not None:
            source_rows.append(index)
            target_rows.append(target_index)
    if not source_rows:
        raise ValueError("Activation payloads have no shared sample ids.")
    return (
        _subset_payload(source, source_rows),
        _subset_payload(target, target_rows),
    )


def _load_payload(path: str | Path) -> CrossModelPayload:
    activation_path = Path(path)
    with np.load(activation_path, allow_pickle=False) as data:
        activations = np.asarray(data["activations"], dtype=np.float64)
        pair_ids = np.asarray(data["pair_ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=str)
        sample_ids = _optional_array(data, ("sample_ids", "ids"))
    if activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional matrix.")
    if sample_ids is None or len(sample_ids) != activations.shape[0]:
        sample_ids = _fallback_sample_ids(pair_ids, labels)
    payload = CrossModelPayload(
        path=str(activation_path),
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        sample_ids=np.asarray(sample_ids, dtype=str),
    )
    _validate_payload(payload)
    return payload


def _optional_array(data: Any, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=str)
    return None


def _fallback_sample_ids(pair_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    keys: list[str] = []
    for pair_id, label in zip(pair_ids.astype(str), labels.astype(str), strict=True):
        count_key = (pair_id, label)
        index = counts[count_key]
        counts[count_key] += 1
        keys.append(f"{pair_id}:{label}:{index}")
    return np.asarray(keys, dtype=str)


def _subset_payload(payload: CrossModelPayload, rows: Sequence[int]) -> CrossModelPayload:
    row_array = np.asarray(rows, dtype=np.int64)
    return CrossModelPayload(
        path=payload.path,
        activations=payload.activations[row_array],
        pair_ids=payload.pair_ids[row_array],
        labels=payload.labels[row_array],
        sample_ids=payload.sample_ids[row_array],
    )


def _validate_payload(payload: CrossModelPayload) -> None:
    row_count = payload.activations.shape[0]
    for name, value in (
        ("pair_ids", payload.pair_ids),
        ("labels", payload.labels),
        ("sample_ids", payload.sample_ids),
    ):
        if len(value) != row_count:
            raise ValueError(f"{name} length does not match activations.")


def _validate_aligned_payloads(
    source: CrossModelPayload,
    target: CrossModelPayload,
) -> None:
    _validate_payload(source)
    _validate_payload(target)
    if source.activations.shape[0] != target.activations.shape[0]:
        raise ValueError("Aligned payloads must have the same row count.")
    if not np.array_equal(source.sample_ids, target.sample_ids):
        raise ValueError("Aligned payloads must use identical sample ids.")
    if not np.array_equal(source.pair_ids, target.pair_ids):
        raise ValueError("Aligned payloads must use identical pair ids.")
    if not np.array_equal(source.labels, target.labels):
        raise ValueError("Aligned payloads must use identical labels.")


def _alignment_metrics(
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    *,
    knn_k: int,
) -> dict[str, Any]:
    return {
        "linear_cka": _linear_cka(source_activations, target_activations),
        "mutual_knn_overlap": _mutual_knn_overlap(
            source_activations,
            target_activations,
            knn_k=knn_k,
        ),
        "effective_knn_k": max(0, min(int(knn_k), source_activations.shape[0] - 1)),
    }


def _linear_cka(source: np.ndarray, target: np.ndarray) -> float:
    source_centered = _center_columns(source)
    target_centered = _center_columns(target)
    source_kernel_energy = _kernel_energy(source_centered)
    target_kernel_energy = _kernel_energy(target_centered)
    denominator = np.sqrt(source_kernel_energy * target_kernel_energy)
    if denominator <= _EPSILON:
        return 0.0
    cross = source_centered.T @ target_centered
    return round(float(np.sum(cross**2) / denominator), 6)


def _kernel_energy(matrix: np.ndarray) -> float:
    kernel = matrix.T @ matrix
    return float(np.sum(kernel**2))


def _mutual_knn_overlap(source: np.ndarray, target: np.ndarray, *, knn_k: int) -> float:
    effective_k = max(0, min(int(knn_k), source.shape[0] - 1))
    if effective_k == 0:
        return 0.0
    source_neighbors = _knn_indices(source, effective_k)
    target_neighbors = _knn_indices(target, effective_k)
    overlaps = [
        len(set(source_row.tolist()) & set(target_row.tolist())) / effective_k
        for source_row, target_row in zip(source_neighbors, target_neighbors, strict=True)
    ]
    return round(float(np.mean(overlaps)), 6) if overlaps else 0.0


def _knn_indices(matrix: np.ndarray, knn_k: int) -> np.ndarray:
    normalized = _row_normalize(_center_columns(matrix))
    similarity = np.asarray(normalized @ normalized.T, dtype=np.float64)
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1)[:, :knn_k]


def _direction_transfer(
    *,
    source: CrossModelPayload,
    target: CrossModelPayload,
    ridge_alpha: float,
    max_leave_one_pair_out_pairs: int | None,
) -> dict[str, Any]:
    source_direction = train_direction_from_arrays(
        source.activations,
        labels=source.labels,
    ).direction
    target_direction = train_direction_from_arrays(
        target.activations,
        labels=target.labels,
    ).direction
    linear_map = _ridge_linear_map(
        source.activations,
        target.activations,
        ridge_alpha=ridge_alpha,
    )
    mapped_direction = _normalize_vector(source_direction @ linear_map)
    mapped_metrics = _pairwise_direction_metrics(target, mapped_direction)
    target_self_metrics = _pairwise_direction_metrics(target, target_direction)
    oracle_direction = (
        -mapped_direction
        if float(mapped_metrics["mean_margin"]) < 0.0
        else mapped_direction
    )
    oracle_metrics = _pairwise_direction_metrics(target, oracle_direction)
    return {
        "source_activation_npz": source.path,
        "target_activation_npz": target.path,
        "source_dim": int(source.activations.shape[1]),
        "target_dim": int(target.activations.shape[1]),
        "mapped_direction_cosine_to_target_direction": round(
            float(mapped_direction @ target_direction),
            6,
        ),
        "target_self_direction": target_self_metrics,
        "mapped_source_direction": mapped_metrics,
        "oracle_sign_mapped_source_direction": oracle_metrics,
        "leave_one_pair_out_mapped_source_direction": (
            _leave_one_pair_out_direction_transfer(
                source=source,
                target=target,
                ridge_alpha=ridge_alpha,
                max_pairs=max_leave_one_pair_out_pairs,
            )
        ),
        "mapped_margin_retention": _margin_retention(
            float(mapped_metrics["mean_margin"]),
            float(target_self_metrics["mean_margin"]),
        ),
        "linear_map_frobenius_norm": round(float(np.linalg.norm(linear_map)), 6),
    }


def _ridge_linear_map(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
) -> np.ndarray:
    source_centered = _center_columns(source)
    target_centered = _center_columns(target)
    xtx = source_centered.T @ source_centered
    scale = max(float(np.trace(xtx)) / max(1, xtx.shape[0]), _EPSILON)
    penalty = float(ridge_alpha) * scale
    system = xtx + penalty * np.eye(xtx.shape[0], dtype=np.float64)
    rhs = source_centered.T @ target_centered
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ rhs


def _leave_one_pair_out_direction_transfer(
    *,
    source: CrossModelPayload,
    target: CrossModelPayload,
    ridge_alpha: float,
    max_pairs: int | None,
) -> dict[str, Any]:
    outcomes: list[float] = []
    margins: list[float] = []
    all_pair_ids = sorted(set(source.pair_ids.astype(str).tolist()))
    selected_pair_ids = _selected_pair_ids(all_pair_ids, max_pairs=max_pairs)
    for pair_id in selected_pair_ids:
        test_mask = source.pair_ids.astype(str) == pair_id
        train_mask = np.logical_not(test_mask)
        if int(test_mask.sum()) < 2 or int(train_mask.sum()) < 2:
            continue
        if not _has_both_labels(source.labels[test_mask]):
            continue
        try:
            source_direction = train_direction_from_arrays(
                source.activations[train_mask],
                labels=source.labels[train_mask],
            ).direction
        except ValueError:
            continue
        mapped_direction = _ridge_mapped_direction(
            source_activations=source.activations[train_mask],
            target_activations=target.activations[train_mask],
            source_direction=source_direction,
            ridge_alpha=ridge_alpha,
        )
        test_payload = _mask_payload(target, test_mask)
        metrics = _pairwise_direction_metrics(test_payload, mapped_direction)
        if int(metrics["pairs"]) != 1:
            continue
        margin = float(metrics["mean_margin"])
        margins.append(margin)
        outcomes.append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)
    return {
        "available_pairs": len(all_pair_ids),
        "evaluated_pairs": len(selected_pair_ids),
        "pairs": len(margins),
        "pairwise_accuracy": round(float(np.mean(outcomes)), 6) if outcomes else 0.0,
        "mean_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_margin": round(float(np.max(margins)), 6) if margins else 0.0,
    }


def _selected_pair_ids(pair_ids: Sequence[str], *, max_pairs: int | None) -> list[str]:
    if max_pairs is None or max_pairs <= 0 or len(pair_ids) <= max_pairs:
        return list(pair_ids)
    if max_pairs == 1:
        return [pair_ids[0]]
    indices = np.linspace(0, len(pair_ids) - 1, num=max_pairs, dtype=int)
    return [pair_ids[int(index)] for index in sorted(set(indices.tolist()))]


def _ridge_mapped_direction(
    *,
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    source_direction: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    source_centered = _center_columns(source_activations)
    target_centered = _center_columns(target_activations)
    kernel = source_centered @ source_centered.T
    scale = max(
        float(np.trace(kernel)) / max(1, source_centered.shape[1]),
        _EPSILON,
    )
    penalty = float(ridge_alpha) * scale
    system = kernel + penalty * np.eye(kernel.shape[0], dtype=np.float64)
    rhs = source_centered @ source_direction
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(system) @ rhs
    return _normalize_vector(coefficients @ target_centered)


def _pairwise_direction_metrics(
    payload: CrossModelPayload,
    direction: np.ndarray,
) -> dict[str, Any]:
    projections = payload.activations @ direction
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(
        payload.pair_ids.astype(str),
        payload.labels.astype(str),
        projections,
        strict=True,
    ):
        grouped[pair_id][label].append(float(projection))
    margins = [
        float(np.mean(scores["positive"])) - float(np.mean(scores["negative"]))
        for scores in grouped.values()
        if "positive" in scores and "negative" in scores
    ]
    outcomes = [1.0 if margin > 0 else 0.5 if margin == 0 else 0.0 for margin in margins]
    return {
        "pairs": len(margins),
        "pairwise_accuracy": round(float(np.mean(outcomes)), 6) if outcomes else 0.0,
        "mean_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_margin": round(float(np.max(margins)), 6) if margins else 0.0,
    }


def _mask_payload(payload: CrossModelPayload, mask: np.ndarray) -> CrossModelPayload:
    return CrossModelPayload(
        path=payload.path,
        activations=payload.activations[mask],
        pair_ids=payload.pair_ids[mask],
        labels=payload.labels[mask],
        sample_ids=payload.sample_ids[mask],
    )


def _has_both_labels(labels: np.ndarray) -> bool:
    values = set(labels.astype(str).tolist())
    return "positive" in values and "negative" in values


def _summary(
    alignment: Mapping[str, Any],
    source_to_target: Mapping[str, Any],
    target_to_source: Mapping[str, Any],
) -> dict[str, Any]:
    source_mapped = _mapping(source_to_target.get("mapped_source_direction"))
    target_mapped = _mapping(target_to_source.get("mapped_source_direction"))
    source_loo = _mapping(
        source_to_target.get("leave_one_pair_out_mapped_source_direction")
    )
    target_loo = _mapping(
        target_to_source.get("leave_one_pair_out_mapped_source_direction")
    )
    return {
        "linear_cka": float(alignment.get("linear_cka", 0.0)),
        "mutual_knn_overlap": float(alignment.get("mutual_knn_overlap", 0.0)),
        "source_to_target_mapped_accuracy": float(
            source_mapped.get("pairwise_accuracy", 0.0)
        ),
        "source_to_target_mapped_mean_margin": float(
            source_mapped.get("mean_margin", 0.0)
        ),
        "source_to_target_direction_cosine": float(
            source_to_target.get("mapped_direction_cosine_to_target_direction", 0.0)
        ),
        "target_to_source_mapped_accuracy": float(
            target_mapped.get("pairwise_accuracy", 0.0)
        ),
        "target_to_source_mapped_mean_margin": float(
            target_mapped.get("mean_margin", 0.0)
        ),
        "target_to_source_direction_cosine": float(
            target_to_source.get("mapped_direction_cosine_to_target_direction", 0.0)
        ),
        "source_to_target_loo_mapped_accuracy": float(
            source_loo.get("pairwise_accuracy", 0.0)
        ),
        "source_to_target_loo_mapped_mean_margin": float(
            source_loo.get("mean_margin", 0.0)
        ),
        "target_to_source_loo_mapped_accuracy": float(
            target_loo.get("pairwise_accuracy", 0.0)
        ),
        "target_to_source_loo_mapped_mean_margin": float(
            target_loo.get("mean_margin", 0.0)
        ),
    }


def _transfer_row(label: str, transfer: Mapping[str, Any]) -> str:
    target_self = _mapping(transfer.get("target_self_direction"))
    mapped = _mapping(transfer.get("mapped_source_direction"))
    loo = _mapping(transfer.get("leave_one_pair_out_mapped_source_direction"))
    oracle = _mapping(transfer.get("oracle_sign_mapped_source_direction"))
    return (
        "| "
        f"{label} | "
        f"{float(target_self.get('pairwise_accuracy', 0.0)):.3f} | "
        f"{float(mapped.get('pairwise_accuracy', 0.0)):.3f} | "
        f"{float(loo.get('pairwise_accuracy', 0.0)):.3f} | "
        f"{float(loo.get('mean_margin', 0.0)):+.3f} | "
        f"{float(transfer.get('mapped_direction_cosine_to_target_direction', 0.0)):+.3f} | "
        f"{float(oracle.get('pairwise_accuracy', 0.0)):.3f} |"
    )


def _margin_retention(mapped_margin: float, target_margin: float) -> float:
    if abs(target_margin) <= _EPSILON:
        return 0.0
    return round(mapped_margin / target_margin, 6)


def _center_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _EPSILON)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
