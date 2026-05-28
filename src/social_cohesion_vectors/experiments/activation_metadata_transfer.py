"""Activation-vector transfer over held-out pair metadata groups."""

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


def run_activation_metadata_transfer_from_files(
    *,
    activation_npz: str | Path,
    pairs_path: str | Path,
    metadata_key: str = "primary_fault_class",
) -> dict[str, Any]:
    """Load activations and pairs, then evaluate held-out metadata transfer."""

    return run_activation_metadata_transfer(
        activation_npz=activation_npz,
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        pairs_path=str(pairs_path),
    )


def run_activation_metadata_transfer(
    *,
    activation_npz: str | Path,
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    pairs_path: str | None = None,
) -> dict[str, Any]:
    """Train on all but one metadata value and evaluate activation margins."""

    payload = load_activation_payload(activation_npz)
    pair_values = _pair_metadata_values(pairs, metadata_key)
    folds: list[dict[str, Any]] = []
    for held_out in sorted({value for values in pair_values.values() for value in values}):
        test_pair_ids = {
            pair_id for pair_id, values in pair_values.items() if held_out in values
        }
        fold = _evaluate_activation_fold(
            activations=payload.activations,
            pair_ids=[str(pair_id) for pair_id in payload.pair_ids],
            labels=[str(label) for label in payload.labels],
            held_out=held_out,
            test_pair_ids=test_pair_ids,
        )
        if fold is not None:
            folds.append(fold)
    return {
        "experiment": "activation_metadata_transfer",
        "description": (
            "Trains a contrastive activation direction with one pair metadata "
            "group held out, then evaluates margins on that unseen group."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": pairs_path,
            "metadata_key": metadata_key,
            "pairs": len(pairs),
            "prompts": int(payload.activations.shape[0]),
            "activation_dim": int(payload.activations.shape[1]),
            "groups": len(folds),
        },
        "summary": _summary(folds),
        "folds": folds,
    }


def save_activation_metadata_transfer_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown activation metadata transfer reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_activation_metadata_transfer_markdown(report),
        encoding="utf-8",
    )


def render_activation_metadata_transfer_markdown(report: Mapping[str, Any]) -> str:
    """Render held-out activation transfer as markdown."""

    summary = _mapping(report.get("summary"))
    inputs = _mapping(report.get("inputs"))
    lines = [
        "# Activation Metadata Transfer",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Metadata key: `{inputs.get('metadata_key', '')}`",
        f"- Pairs: {int(inputs.get('pairs', 0))}",
        f"- Prompts: {int(inputs.get('prompts', 0))}",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        f"- Groups: {int(inputs.get('groups', 0))}",
        "",
        "## Summary",
        "",
        f"- Test pairs: {int(summary.get('test_pairs', 0))}",
        f"- Mean test accuracy: {float(summary.get('mean_test_accuracy', 0.0)):.3f}",
        f"- Mean test margin: {float(summary.get('mean_test_margin', 0.0)):+.3f}",
        "",
        "## Held-Out Groups",
        "",
        "| Held-out | Train pairs | Test pairs | Test accuracy | Test margin | Min margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in _sequence(report.get("folds")):
        fold_map = _mapping(fold)
        lines.append(
            "| "
            f"{fold_map.get('held_out', '')} | "
            f"{int(fold_map.get('train_pairs', 0))} | "
            f"{int(fold_map.get('test_pairs', 0))} | "
            f"{float(fold_map.get('test_accuracy', 0.0)):.3f} | "
            f"{float(fold_map.get('mean_margin', 0.0)):+.3f} | "
            f"{float(fold_map.get('min_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _evaluate_activation_fold(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    held_out: str,
    test_pair_ids: set[str],
) -> dict[str, Any] | None:
    train_mask = np.asarray([pair_id not in test_pair_ids for pair_id in pair_ids])
    test_mask = np.asarray([pair_id in test_pair_ids for pair_id in pair_ids])
    if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 2:
        return None
    direction = train_direction_from_arrays(
        activations[train_mask],
        labels=np.asarray(labels, dtype=str)[train_mask],
    ).direction
    projections = activations[test_mask] @ direction
    test_pairs = np.asarray(pair_ids, dtype=str)[test_mask]
    test_labels = np.asarray(labels, dtype=str)[test_mask]
    margins = _pair_margins(
        pair_ids=[str(pair_id) for pair_id in test_pairs],
        labels=[str(label) for label in test_labels],
        projections=[float(projection) for projection in projections],
    )
    outcomes = [1.0 if margin > 0 else 0.5 if margin == 0 else 0.0 for margin in margins]
    return {
        "held_out": held_out,
        "train_pairs": len({pair_id for pair_id in pair_ids if pair_id not in test_pair_ids}),
        "test_pairs": len(margins),
        "test_accuracy": round(sum(outcomes) / len(outcomes), 6) if outcomes else 0.0,
        "mean_margin": _mean(margins),
        "min_margin": round(min(margins), 6) if margins else 0.0,
    }


def _pair_margins(
    *,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    projections: Sequence[float],
) -> list[float]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for pair_id, label, projection in zip(pair_ids, labels, projections, strict=True):
        grouped[pair_id][label] = float(projection)
    margins: list[float] = []
    for scores in grouped.values():
        if "positive" in scores and "negative" in scores:
            margins.append(scores["positive"] - scores["negative"])
    return margins


def _pair_metadata_values(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for pair in pairs:
        raw = pair.metadata.get(metadata_key)
        values[pair.pair_id] = tuple(
            part.strip() for part in str(raw or "").split(",") if part.strip()
        )
    return values


def _summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_pairs = sum(int(fold.get("test_pairs", 0)) for fold in folds)
    weighted_accuracy = sum(
        float(fold.get("test_accuracy", 0.0)) * int(fold.get("test_pairs", 0))
        for fold in folds
    )
    weighted_margin = sum(
        float(fold.get("mean_margin", 0.0)) * int(fold.get("test_pairs", 0))
        for fold in folds
    )
    return {
        "folds": len(folds),
        "test_pairs": total_pairs,
        "mean_test_accuracy": round(weighted_accuracy / total_pairs, 6)
        if total_pairs
        else 0.0,
        "mean_test_margin": round(weighted_margin / total_pairs, 6)
        if total_pairs
        else 0.0,
    }


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
