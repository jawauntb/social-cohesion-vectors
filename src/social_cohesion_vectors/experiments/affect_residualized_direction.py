"""Train steering-ready cohesion directions with coarse affect removed."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from social_cohesion_vectors.activations.contrastive import (
    ContrastiveDirection,
    save_direction,
    train_direction_from_arrays,
)
from social_cohesion_vectors.experiments.affect_controls import AFFECT_LABELS

_EPSILON = 1e-12


def train_affect_residualized_direction(
    *,
    activations: NDArray[np.float64],
    pair_ids: Sequence[str],
    labels: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    activation_npz: str | Path,
    pairs_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[ContrastiveDirection, dict[str, Any]]:
    """Train a contrastive direction after removing an affect-label subspace.

    This is a training artifact, not a held-out causal evaluation. The saved
    vector remains in the original activation coordinate system, but it is
    explicitly projected into the orthogonal complement of the learned affect
    basis so it can be used by the existing Modal steering hook.
    """

    matrix = np.asarray(activations, dtype=np.float64)
    pair_array = np.asarray(pair_ids, dtype=str)
    label_array = np.asarray(labels, dtype=str)
    metadata_by_pair = _metadata_by_pair(pairs)
    affect_labels = np.asarray(
        [
            str(metadata_by_pair[str(pair_id)].get("affect_label", "unknown"))
            for pair_id in pair_array
        ],
        dtype=str,
    )

    basis = affect_basis(matrix, affect_labels)
    residualized = residualize(matrix, basis)
    raw_direction = train_direction_from_arrays(matrix, labels=label_array)
    residualized_direction = train_direction_from_arrays(
        residualized,
        labels=label_array,
    )
    orthogonal_direction = _normalize_vector(
        residualize(residualized_direction.direction[None, :], basis)[0]
    )
    direction = ContrastiveDirection(
        direction=orthogonal_direction,
        top_mean=residualized_direction.top_mean,
        bottom_mean=residualized_direction.bottom_mean,
        top_count=residualized_direction.top_count,
        bottom_count=residualized_direction.bottom_count,
        activation_count=residualized_direction.activation_count,
        source_paths=(str(activation_npz),),
        metadata={
            "split_method": "label",
            "residualization": "affect_label_subspace",
            "affect_subspace_rank": int(basis.shape[1]),
            "retained_norm_fraction": _retained_norm_fraction(matrix, residualized),
            "max_abs_affect_basis_dot": _max_abs_basis_dot(
                orthogonal_direction,
                basis,
            ),
        },
    )
    saved_path = str(save_direction(output_path, direction)) if output_path else None
    report = _report(
        activations=matrix,
        residualized=residualized,
        pair_ids=pair_array,
        labels=label_array,
        affect_labels=affect_labels,
        pairs=metadata_by_pair,
        basis=basis,
        raw_direction=raw_direction,
        residualized_direction=direction,
        activation_npz=activation_npz,
        pairs_path=pairs_path,
        output_path=saved_path,
    )
    return direction, report


def affect_basis(
    activations: NDArray[np.float64],
    affect_labels: NDArray[np.str_],
) -> NDArray[np.float64]:
    """Return an orthonormal basis for one-vs-rest affect-label directions."""

    centered = activations - activations.mean(axis=0, keepdims=True)
    directions: list[NDArray[np.float64]] = []
    for affect_label in AFFECT_LABELS:
        positive_mask = affect_labels == affect_label
        negative_mask = ~positive_mask
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            continue
        direction = centered[positive_mask].mean(axis=0) - centered[negative_mask].mean(
            axis=0
        )
        norm = float(np.linalg.norm(direction))
        if norm > _EPSILON:
            directions.append(direction / norm)
    if not directions:
        return np.zeros((activations.shape[1], 0), dtype=np.float64)
    stacked = np.vstack(directions).T
    q, r = np.linalg.qr(stacked)
    keep = np.abs(np.diag(r)) > 1e-10
    return q[:, keep]


def residualize(
    activations: NDArray[np.float64],
    basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project activation rows into the orthogonal complement of ``basis``."""

    if basis.shape[1] == 0:
        return activations.copy()
    return activations - (activations @ basis) @ basis.T


def write_affect_residualized_direction_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown reports for the residualized direction."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_affect_residualized_direction_markdown(report),
        encoding="utf-8",
    )


def render_affect_residualized_direction_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a concise report for the residualized steering direction."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Affect-Residualized Direction",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompts: {int(summary.get('prompts', 0))}",
        f"- Activation dim: {int(summary.get('activation_dim', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Affect classes: {int(summary.get('affect_classes', 0))}",
        f"- Affect subspace rank: {int(summary.get('affect_subspace_rank', 0))}",
        (
            "- Retained norm fraction: "
            f"{float(summary.get('retained_norm_fraction', 0.0)):.3f}"
        ),
        (
            "- Max absolute dot with affect basis: "
            f"{float(summary.get('max_abs_affect_basis_dot', 0.0)):.6f}"
        ),
        (
            "- Raw direction in-sample accuracy: "
            f"{float(summary.get('raw_pairwise_accuracy', 0.0)):.3f}"
        ),
        (
            "- Raw direction mean margin: "
            f"{float(summary.get('raw_mean_margin', 0.0)):+.3f}"
        ),
        (
            "- Residualized direction in-sample accuracy: "
            f"{float(summary.get('residualized_pairwise_accuracy', 0.0)):.3f}"
        ),
        (
            "- Residualized direction mean margin: "
            f"{float(summary.get('residualized_mean_margin', 0.0)):+.3f}"
        ),
        (
            "- Residualized direction min margin: "
            f"{float(summary.get('residualized_min_margin', 0.0)):+.3f}"
        ),
        "",
        "## Lowest Residualized Margins",
        "",
        "| Pair | Affect | Mechanism | Pole | Residualized margin |",
        "| --- | --- | --- | --- | ---: |",
    ]
    rows = sorted(
        (_mapping(row) for row in _sequence(report.get("pairs"))),
        key=lambda row: float(row.get("residualized_margin", 0.0)),
    )
    for row in rows[:20]:
        lines.append(
            "| "
            f"{row.get('pair_id', '')} | "
            f"{row.get('affect_label', '')} | "
            f"{row.get('mechanism', '')} | "
            f"{row.get('negative_pole', '')} | "
            f"{float(row.get('residualized_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _report(
    *,
    activations: NDArray[np.float64],
    residualized: NDArray[np.float64],
    pair_ids: NDArray[np.str_],
    labels: NDArray[np.str_],
    affect_labels: NDArray[np.str_],
    pairs: Mapping[str, Mapping[str, Any]],
    basis: NDArray[np.float64],
    raw_direction: ContrastiveDirection,
    residualized_direction: ContrastiveDirection,
    activation_npz: str | Path,
    pairs_path: str | Path,
    output_path: str | None,
) -> dict[str, Any]:
    raw_rows = _pair_rows(
        pair_ids=pair_ids,
        labels=labels,
        projections=raw_direction.project(activations),
        metadata_by_pair=pairs,
        margin_key="raw_margin",
    )
    residual_rows = _pair_rows(
        pair_ids=pair_ids,
        labels=labels,
        projections=residualized_direction.project(residualized),
        metadata_by_pair=pairs,
        margin_key="residualized_margin",
    )
    rows = [
        {
            **raw_row,
            "residualized_margin": residual_row["residualized_margin"],
            "residualized_positive_projection": residual_row[
                "residualized_positive_projection"
            ],
            "residualized_negative_projection": residual_row[
                "residualized_negative_projection"
            ],
        }
        for raw_row, residual_row in zip(raw_rows, residual_rows, strict=True)
    ]
    return {
        "experiment": "affect_residualized_direction",
        "description": (
            "Trains a social-cohesion activation direction after projecting out "
            "a coarse affect-label subspace. The vector is intended for causal "
            "steering controls, while held-out causal claims still require the "
            "separate steering benchmark."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": str(pairs_path),
            "output_path": output_path,
        },
        "summary": {
            "prompts": int(activations.shape[0]),
            "activation_dim": int(activations.shape[1]),
            "pairwise_examples": len(set(str(pair_id) for pair_id in pair_ids)),
            "affect_classes": len(set(str(label) for label in affect_labels)),
            "affect_subspace_rank": int(basis.shape[1]),
            "retained_norm_fraction": _retained_norm_fraction(
                activations,
                residualized,
            ),
            "max_abs_affect_basis_dot": _max_abs_basis_dot(
                residualized_direction.direction,
                basis,
            ),
            "raw_pairwise_accuracy": _accuracy(raw_rows, "raw_margin"),
            "raw_mean_margin": _mean(float(row["raw_margin"]) for row in raw_rows),
            "raw_min_margin": _min(float(row["raw_margin"]) for row in raw_rows),
            "residualized_pairwise_accuracy": _accuracy(
                residual_rows,
                "residualized_margin",
            ),
            "residualized_mean_margin": _mean(
                float(row["residualized_margin"]) for row in residual_rows
            ),
            "residualized_min_margin": _min(
                float(row["residualized_margin"]) for row in residual_rows
            ),
        },
        "pairs": rows,
    }


def _pair_rows(
    *,
    pair_ids: NDArray[np.str_],
    labels: NDArray[np.str_],
    projections: NDArray[np.float64],
    metadata_by_pair: Mapping[str, Mapping[str, Any]],
    margin_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    projection_key = margin_key.removesuffix("_margin")
    for pair_id in sorted(set(str(pair_id) for pair_id in pair_ids)):
        mask = pair_ids == pair_id
        if int(mask.sum()) != 2:
            continue
        scores = {
            str(label): float(projection)
            for label, projection in zip(labels[mask], projections[mask], strict=True)
        }
        if "positive" not in scores or "negative" not in scores:
            continue
        metadata = metadata_by_pair[pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "affect_label": str(metadata.get("affect_label", "")),
                "mechanism": str(metadata.get("mechanism", "")),
                "negative_pole": str(metadata.get("negative_pole", "")),
                f"{projection_key}_positive_projection": round(
                    scores["positive"],
                    6,
                ),
                f"{projection_key}_negative_projection": round(
                    scores["negative"],
                    6,
                ),
                margin_key: round(scores["positive"] - scores["negative"], 6),
            }
        )
    return rows


def _metadata_by_pair(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    metadata: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        pair_metadata = _mapping(pair.get("metadata"))
        metadata[pair_id] = pair_metadata
    return metadata


def _normalize_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        raise ValueError("Cannot normalize a zero-length direction.")
    return vector / norm


def _retained_norm_fraction(
    original: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> float:
    original_norm = float(np.linalg.norm(original))
    if original_norm <= _EPSILON:
        return 0.0
    return round(float(np.linalg.norm(residual) / original_norm), 6)


def _max_abs_basis_dot(
    vector: NDArray[np.float64],
    basis: NDArray[np.float64],
) -> float:
    if basis.shape[1] == 0:
        return 0.0
    return round(float(np.max(np.abs(vector @ basis))), 12)


def _accuracy(rows: Sequence[Mapping[str, Any]], margin_key: str) -> float:
    if not rows:
        return 0.0
    wins = sum(1.0 for row in rows if float(row[margin_key]) > 0.0)
    ties = sum(0.5 for row in rows if float(row[margin_key]) == 0.0)
    return round((wins + ties) / len(rows), 6)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(float(np.mean(materialized)), 6) if materialized else 0.0


def _min(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(float(np.min(materialized)), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
