"""Reviewer-oriented geometry audits for contrastive activation directions."""

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


def run_direction_geometry_audit_from_files(
    *,
    activation_npz: str | Path,
    pairs_path: str | Path,
    metadata_key: str = "primary_fault_class",
    high_abs_threshold: float = 0.8,
) -> dict[str, Any]:
    """Load activations and pairs, then audit direction cosine geometry."""

    return run_direction_geometry_audit(
        activation_npz=activation_npz,
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        pairs_path=str(pairs_path),
        high_abs_threshold=high_abs_threshold,
    )


def run_direction_geometry_audit(
    *,
    activation_npz: str | Path,
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    pairs_path: str | None = None,
    high_abs_threshold: float = 0.8,
) -> dict[str, Any]:
    """Train one direction per metadata group and inspect signed cosines.

    The key reviewer guardrail is that a near-zero mean signed cosine can hide
    strongly aligned and anti-aligned directions. Anti-alignment means the same
    unsigned axis with reversed polarity, not independence.
    """

    payload = load_activation_payload(activation_npz)
    group_pairs = _group_pair_ids(pairs, metadata_key)
    rows = _train_group_directions(
        activations=payload.activations,
        pair_ids=[str(pair_id) for pair_id in payload.pair_ids],
        labels=[str(label) for label in payload.labels],
        group_pairs=group_pairs,
    )
    comparisons, cosine_matrix = _direction_comparisons(
        rows,
        high_abs_threshold=high_abs_threshold,
    )
    return {
        "experiment": "direction_geometry_audit",
        "description": (
            "Trains one contrastive activation direction per metadata group and "
            "reports signed and absolute off-diagonal cosine structure."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": pairs_path,
            "metadata_key": metadata_key,
            "pairs": len(pairs),
            "prompts": int(payload.activations.shape[0]),
            "activation_dim": int(payload.activations.shape[1]),
            "high_abs_threshold": float(high_abs_threshold),
        },
        "summary": _geometry_summary(
            comparisons,
            high_abs_threshold,
            group_count=len(rows),
        ),
        "groups": [_public_group_row(row) for row in rows],
        "cosine_matrix": cosine_matrix.round(6).tolist(),
        "comparisons": comparisons,
    }


def save_direction_geometry_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown direction-geometry reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_direction_geometry_audit_markdown(report),
        encoding="utf-8",
    )


def render_direction_geometry_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render direction geometry as a compact reviewer-facing report."""

    summary = _mapping(report.get("summary"))
    inputs = _mapping(report.get("inputs"))
    lines = [
        "# Direction Geometry Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Metadata key: `{inputs.get('metadata_key', '')}`",
        f"- Pairs: {int(inputs.get('pairs', 0))}",
        f"- Prompts: {int(inputs.get('prompts', 0))}",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        f"- Groups: {int(summary.get('groups', 0))}",
        "",
        "## Summary",
        "",
        f"- Comparisons: {int(summary.get('comparisons', 0))}",
        f"- Mean signed off-diagonal cosine: "
        f"{float(summary.get('mean_signed_off_diagonal_cosine', 0.0)):+.3f}",
        f"- Mean absolute off-diagonal cosine: "
        f"{float(summary.get('mean_absolute_off_diagonal_cosine', 0.0)):.3f}",
        f"- Cancellation gap: {float(summary.get('cancellation_gap', 0.0)):.3f}",
        f"- Strongly aligned pairs: {int(summary.get('strongly_aligned_pairs', 0))}",
        f"- Strongly anti-aligned pairs: "
        f"{int(summary.get('strongly_anti_aligned_pairs', 0))}",
        "",
        "## Interpretation Guardrail",
        "",
        (
            "A low mean signed cosine is not enough to claim independent axes. "
            "Large absolute cosines or anti-aligned pairs indicate shared axes "
            "with different polarity."
        ),
        "",
        "## Top Absolute-Cosine Comparisons",
        "",
        "| Group A | Group B | Signed cosine | Absolute cosine | Relation |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in _sequence(report.get("comparisons"))[:20]:
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group_a', '')} | "
            f"{row_map.get('group_b', '')} | "
            f"{float(row_map.get('signed_cosine', 0.0)):+.3f} | "
            f"{float(row_map.get('absolute_cosine', 0.0)):.3f} | "
            f"{row_map.get('relation', '')} |"
        )
    return "\n".join(lines) + "\n"


def _train_group_directions(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    group_pairs: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_array = np.asarray(labels, dtype=str)
    for group, selected_pairs in sorted(group_pairs.items()):
        mask = np.asarray([pair_id in selected_pairs for pair_id in pair_ids], dtype=bool)
        if int(mask.sum()) < 2:
            continue
        try:
            direction = train_direction_from_arrays(
                activations[mask],
                labels=label_array[mask],
            )
        except ValueError:
            continue
        rows.append(
            {
                "group": group,
                "pair_count": len(selected_pairs),
                "prompt_count": int(mask.sum()),
                "positive_count": direction.top_count,
                "negative_count": direction.bottom_count,
                "direction": direction.direction,
            }
        )
    return rows


def _direction_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    high_abs_threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not rows:
        return [], np.empty((0, 0), dtype=np.float64)
    direction_matrix = np.vstack([np.asarray(row["direction"]) for row in rows])
    cosine_matrix = direction_matrix @ direction_matrix.T
    comparisons: list[dict[str, Any]] = []
    for i, row_a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            cosine = float(cosine_matrix[i, j])
            comparisons.append(
                {
                    "group_a": str(row_a["group"]),
                    "group_b": str(rows[j]["group"]),
                    "signed_cosine": round(cosine, 6),
                    "absolute_cosine": round(abs(cosine), 6),
                    "relation": _relation(cosine, high_abs_threshold),
                }
            )
    comparisons.sort(key=lambda row: float(row["absolute_cosine"]), reverse=True)
    return comparisons, cosine_matrix


def _geometry_summary(
    comparisons: Sequence[Mapping[str, Any]],
    high_abs_threshold: float,
    *,
    group_count: int,
) -> dict[str, Any]:
    signed = [float(row["signed_cosine"]) for row in comparisons]
    absolute = [abs(value) for value in signed]
    mean_signed = _mean(signed)
    return {
        "groups": group_count,
        "comparisons": len(comparisons),
        "mean_signed_off_diagonal_cosine": mean_signed,
        "mean_absolute_off_diagonal_cosine": _mean(absolute),
        "median_signed_off_diagonal_cosine": _quantile(signed, 0.5),
        "median_absolute_off_diagonal_cosine": _quantile(absolute, 0.5),
        "p90_absolute_off_diagonal_cosine": _quantile(absolute, 0.9),
        "cancellation_gap": round(_mean(absolute) - abs(mean_signed), 6),
        "strongly_aligned_pairs": sum(
            1 for value in signed if value >= high_abs_threshold
        ),
        "strongly_anti_aligned_pairs": sum(
            1 for value in signed if value <= -high_abs_threshold
        ),
        "high_absolute_pairs": sum(
            1 for value in absolute if value >= high_abs_threshold
        ),
    }


def _public_group_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group": str(row["group"]),
        "pair_count": int(row["pair_count"]),
        "prompt_count": int(row["prompt_count"]),
        "positive_count": int(row["positive_count"]),
        "negative_count": int(row["negative_count"]),
    }


def _group_pair_ids(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        for value in _metadata_values(pair, metadata_key):
            grouped[value].add(pair.pair_id)
    return dict(grouped)


def _metadata_values(pair: PairwiseExample, metadata_key: str) -> tuple[str, ...]:
    raw = pair.metadata.get(metadata_key)
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _relation(cosine: float, high_abs_threshold: float) -> str:
    if cosine >= high_abs_threshold:
        return "strongly_aligned"
    if cosine <= -high_abs_threshold:
        return "strongly_anti_aligned"
    if abs(cosine) <= 0.1:
        return "near_zero"
    return "mixed"
def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.quantile(np.asarray(values, dtype=np.float64), q)), 6)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
