"""Evaluate cohesion activation directions after affect-subspace residualization."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from social_cohesion_vectors.activations.contrastive import (
    train_direction_from_arrays,
)
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.affect_controls import AFFECT_LABELS


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _load_activation_payload(args.activation_npz)
    pairs = load_pairwise_examples_jsonl(args.pairs)
    report = run_affect_activation_residualization(
        activations=payload["activations"],
        pair_ids=payload["pair_ids"],
        labels=payload["labels"],
        pairs=[pair.model_dump(mode="json") for pair in pairs],
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
    )
    _write_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = _mapping(report["summary"])
    print(
        "affect activation residualization: "
        f"original_loo={float(summary['original_pair_loo_accuracy']):.3f} "
        f"residualized_loo={float(summary['residualized_pair_loo_accuracy']):.3f} "
        f"pairs={int(summary['pairwise_examples'])} "
        f"rank={float(summary['mean_affect_subspace_rank']):.1f}"
    )
    return 0


def run_affect_activation_residualization(
    *,
    activations: NDArray[np.float64],
    pair_ids: NDArray[np.str_],
    labels: NDArray[np.str_],
    pairs: Sequence[Mapping[str, Any]],
    activation_npz: str | Path,
    pairs_path: str | Path,
) -> dict[str, Any]:
    """Return original and affect-residualized leave-one-pair-out metrics."""

    metadata_by_pair = _metadata_by_pair(pairs)
    sample_affects = np.asarray(
        [
            str(metadata_by_pair[str(pair_id)].get("affect_label", "unknown"))
            for pair_id in pair_ids
        ],
        dtype=str,
    )
    original_rows = _leave_one_pair_out_rows(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        sample_affects=sample_affects,
        metadata_by_pair=metadata_by_pair,
        residualize=False,
    )
    residual_rows = _leave_one_pair_out_rows(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        sample_affects=sample_affects,
        metadata_by_pair=metadata_by_pair,
        residualize=True,
    )
    basis_rows = [
        row for row in residual_rows if isinstance(row.get("affect_subspace_rank"), int)
    ]
    return {
        "experiment": "affect_activation_residualization",
        "description": (
            "Compares cohesion directions trained on raw activations with "
            "directions trained after projecting out an affect-label subspace "
            "learned from anger/sadness/fear/disgust/happy/neutral labels."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": str(pairs_path),
        },
        "summary": {
            "prompts": int(activations.shape[0]),
            "activation_dim": int(activations.shape[1]),
            "pairwise_examples": len(set(str(pair_id) for pair_id in pair_ids)),
            "affect_classes": len(set(str(label) for label in sample_affects)),
            "original_pair_loo_accuracy": _accuracy(original_rows, "margin"),
            "original_mean_margin": _mean(float(row["margin"]) for row in original_rows),
            "original_min_margin": _min(float(row["margin"]) for row in original_rows),
            "residualized_pair_loo_accuracy": _accuracy(
                residual_rows,
                "residualized_margin",
            ),
            "residualized_mean_margin": _mean(
                float(row["residualized_margin"]) for row in residual_rows
            ),
            "residualized_min_margin": _min(
                float(row["residualized_margin"]) for row in residual_rows
            ),
            "mean_retained_norm_fraction": _mean(
                float(row["retained_norm_fraction"]) for row in residual_rows
            ),
            "mean_affect_subspace_rank": _mean(
                float(row["affect_subspace_rank"]) for row in basis_rows
            ),
        },
        "affect_groups": _group_rows(residual_rows, "affect_label"),
        "mechanism_groups": _group_rows(residual_rows, "mechanism"),
        "negative_pole_groups": _group_rows(residual_rows, "negative_pole"),
        "original_pairs": original_rows,
        "residualized_pairs": residual_rows,
    }


def _leave_one_pair_out_rows(
    *,
    activations: NDArray[np.float64],
    pair_ids: NDArray[np.str_],
    labels: NDArray[np.str_],
    sample_affects: NDArray[np.str_],
    metadata_by_pair: Mapping[str, Mapping[str, Any]],
    residualize: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for heldout_pair in sorted(set(str(pair_id) for pair_id in pair_ids)):
        test_mask = pair_ids == heldout_pair
        train_mask = ~test_mask
        if int(test_mask.sum()) != 2:
            continue
        train_activations = activations[train_mask]
        test_activations = activations[test_mask]
        rank = 0
        retained_fraction = 1.0
        if residualize:
            basis = _affect_basis(
                train_activations,
                sample_affects[train_mask],
            )
            rank = int(basis.shape[1])
            train_activations = _residualize(train_activations, basis)
            test_residual = _residualize(test_activations, basis)
            retained_fraction = _retained_norm_fraction(test_activations, test_residual)
            test_activations = test_residual
        direction = train_direction_from_arrays(
            train_activations,
            labels=labels[train_mask],
        )
        projections = direction.project(test_activations)
        scores = {
            str(label): float(projection)
            for label, projection in zip(labels[test_mask], projections, strict=True)
        }
        if "positive" not in scores or "negative" not in scores:
            continue
        metadata = metadata_by_pair[heldout_pair]
        margin = scores["positive"] - scores["negative"]
        row: dict[str, Any] = {
            "pair_id": heldout_pair,
            "affect_label": str(metadata.get("affect_label", "")),
            "mechanism": str(metadata.get("mechanism", "")),
            "negative_pole": str(metadata.get("negative_pole", "")),
            "positive_projection": round(scores["positive"], 6),
            "negative_projection": round(scores["negative"], 6),
        }
        if residualize:
            row.update(
                {
                    "residualized_margin": round(margin, 6),
                    "affect_subspace_rank": rank,
                    "retained_norm_fraction": round(retained_fraction, 6),
                }
            )
        else:
            row["margin"] = round(margin, 6)
        rows.append(row)
    return rows


def _affect_basis(
    activations: NDArray[np.float64],
    affect_labels: NDArray[np.str_],
) -> NDArray[np.float64]:
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
        if norm > 1e-12:
            directions.append(direction / norm)
    if not directions:
        return np.zeros((activations.shape[1], 0), dtype=np.float64)
    stacked = np.vstack(directions).T
    q, r = np.linalg.qr(stacked)
    keep = np.abs(np.diag(r)) > 1e-10
    return q[:, keep]


def _residualize(
    activations: NDArray[np.float64],
    basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    if basis.shape[1] == 0:
        return activations.copy()
    return activations - (activations @ basis) @ basis.T


def _retained_norm_fraction(
    original: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> float:
    original_norm = float(np.linalg.norm(original))
    if original_norm <= 1e-12:
        return 0.0
    return float(np.linalg.norm(residual) / original_norm)


def _metadata_by_pair(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    metadata: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        pair_metadata = _mapping(pair.get("metadata"))
        metadata[pair_id] = pair_metadata
    return metadata


def _load_activation_payload(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "activations": np.asarray(data["activations"], dtype=np.float64),
            "pair_ids": np.asarray(data["pair_ids"], dtype=str),
            "labels": np.asarray(data["labels"], dtype=str),
        }


def _write_reports(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render an activation residualization report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Affect Activation Residualization",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompts: {int(summary.get('prompts', 0))}",
        f"- Activation dim: {int(summary.get('activation_dim', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Affect classes: {int(summary.get('affect_classes', 0))}",
        (
            "- Original pair LOO accuracy: "
            f"{float(summary.get('original_pair_loo_accuracy', 0.0)):.3f}"
        ),
        (
            "- Original mean margin: "
            f"{float(summary.get('original_mean_margin', 0.0)):+.3f}"
        ),
        (
            "- Residualized pair LOO accuracy: "
            f"{float(summary.get('residualized_pair_loo_accuracy', 0.0)):.3f}"
        ),
        (
            "- Residualized mean margin: "
            f"{float(summary.get('residualized_mean_margin', 0.0)):+.3f}"
        ),
        (
            "- Residualized min margin: "
            f"{float(summary.get('residualized_min_margin', 0.0)):+.3f}"
        ),
        (
            "- Mean retained norm fraction: "
            f"{float(summary.get('mean_retained_norm_fraction', 0.0)):.3f}"
        ),
        (
            "- Mean affect subspace rank: "
            f"{float(summary.get('mean_affect_subspace_rank', 0.0)):.1f}"
        ),
        "",
        "## Affect Groups",
        "",
        "| Affect | Pairs | Residualized accuracy | Mean residualized margin |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("affect_groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest Residualized Margins",
            "",
            "| Pair | Affect | Mechanism | Pole | Residualized margin | Retained norm |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    rows = sorted(
        (_mapping(row) for row in _sequence(report.get("residualized_pairs"))),
        key=lambda row: float(row.get("residualized_margin", 0.0)),
    )
    for row in rows[:20]:
        lines.append(
            "| "
            f"{row.get('pair_id', '')} | "
            f"{row.get('affect_label', '')} | "
            f"{row.get('mechanism', '')} | "
            f"{row.get('negative_pole', '')} | "
            f"{float(row.get('residualized_margin', 0.0)):+.3f} | "
            f"{float(row.get('retained_norm_fraction', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _group_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    output: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        output.append(
            {
                "group": group,
                "pairs": len(group_rows),
                "accuracy": _accuracy(group_rows, "residualized_margin"),
                "mean_margin": _mean(
                    float(row["residualized_margin"]) for row in group_rows
                ),
                "min_margin": _min(
                    float(row["residualized_margin"]) for row in group_rows
                ),
            }
        )
    return output


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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/training/affect_control_pairwise_probe_dataset.jsonl"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/reports/affect_activation_residualization.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/reports/affect_activation_residualization.md"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
