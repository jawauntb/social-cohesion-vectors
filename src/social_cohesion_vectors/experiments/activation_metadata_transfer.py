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
    coverage_metadata_keys: Sequence[str] | None = None,
    required_coverage_metadata_keys: Sequence[str] | None = None,
    min_coverage_groups_per_key: int = 1,
) -> dict[str, Any]:
    """Load activations and pairs, then evaluate held-out metadata transfer."""

    return run_activation_metadata_transfer(
        activation_npz=activation_npz,
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        coverage_metadata_keys=coverage_metadata_keys,
        required_coverage_metadata_keys=required_coverage_metadata_keys,
        min_coverage_groups_per_key=min_coverage_groups_per_key,
        pairs_path=str(pairs_path),
    )


def run_activation_metadata_transfer(
    *,
    activation_npz: str | Path,
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    coverage_metadata_keys: Sequence[str] | None = None,
    required_coverage_metadata_keys: Sequence[str] | None = None,
    min_coverage_groups_per_key: int = 1,
    pairs_path: str | None = None,
) -> dict[str, Any]:
    """Train on all but one metadata value and evaluate activation margins."""

    payload = load_activation_payload(activation_npz)
    pair_values = _pair_metadata_values(pairs, metadata_key)
    coverage_keys = _coverage_metadata_keys(metadata_key, coverage_metadata_keys)
    metadata_coverage = _metadata_coverage(pairs, coverage_keys)
    required_coverage_keys = _required_coverage_metadata_keys(
        coverage_keys,
        required_coverage_metadata_keys,
    )
    coverage_readiness = _metadata_coverage_readiness(
        pairs=pairs,
        metadata_coverage=metadata_coverage,
        required_metadata_keys=required_coverage_keys,
        min_groups_per_key=min_coverage_groups_per_key,
    )
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
            "coverage_metadata_keys": coverage_keys,
            "required_coverage_metadata_keys": required_coverage_keys,
            "min_coverage_groups_per_key": min_coverage_groups_per_key,
            "metadata_coverage": metadata_coverage,
        },
        "summary": {
            **_summary(folds),
            "metadata_coverage_readiness": coverage_readiness["status"],
            "ready_for_metadata_coverage_claims": coverage_readiness["ready"],
        },
        "readiness": coverage_readiness,
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
    readiness = _mapping(report.get("readiness"))
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
        f"- Metadata coverage readiness: "
        f"`{readiness.get('status', 'not_ready')}`",
        f"- Ready for metadata coverage claims: "
        f"{bool(readiness.get('ready', False))}",
        "",
        "## Metadata Coverage Readiness",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for gate in _sequence(readiness.get("gates")):
        gate_map = _mapping(gate)
        lines.append(
            "| "
            f"{gate_map.get('gate_id', '')} | "
            f"{float(gate_map.get('value', 0.0)):.3f} | "
            f"{float(gate_map.get('threshold', 0.0)):.3f} | "
            f"{bool(gate_map.get('passed', False))} |"
        )
    failed_metadata_keys = _sequence(readiness.get("failed_metadata_keys"))
    if failed_metadata_keys:
        lines.extend(
            [
                "",
                "Not ready for metadata coverage claims: incomplete metadata "
                f"for {', '.join(str(key) for key in failed_metadata_keys)}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Metadata Coverage",
            "",
            "| Metadata key | Pairs with values | Missing pairs | Groups | Values |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _sequence(inputs.get("metadata_coverage")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('metadata_key', '')}` | "
            f"{int(row_map.get('pairs_with_values', 0))} | "
            f"{int(row_map.get('missing_pairs', 0))} | "
            f"{int(row_map.get('groups', 0))} | "
            f"{', '.join(f'`{value}`' for value in _sequence(row_map.get('values')))} |"
        )
    lines.extend(
        [
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
    )
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


def _coverage_metadata_keys(
    metadata_key: str,
    requested_keys: Sequence[str] | None,
) -> list[str]:
    keys = [
        metadata_key,
        "fault_classes",
        "source",
        "provider",
        "generated_style",
    ]
    if requested_keys is not None:
        keys.extend(requested_keys)
    return list(dict.fromkeys(key for key in keys if key))


def _required_coverage_metadata_keys(
    coverage_keys: Sequence[str],
    requested_keys: Sequence[str] | None,
) -> list[str]:
    keys = coverage_keys if requested_keys is None else requested_keys
    return list(dict.fromkeys(key for key in keys if key))


def _metadata_coverage(
    pairs: Sequence[PairwiseExample],
    metadata_keys: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_key in metadata_keys:
        pair_values = _pair_metadata_values(pairs, metadata_key)
        values = sorted({value for item in pair_values.values() for value in item})
        pairs_with_values = sum(1 for item in pair_values.values() if item)
        rows.append(
            {
                "metadata_key": metadata_key,
                "pairs_with_values": pairs_with_values,
                "missing_pairs": len(pairs) - pairs_with_values,
                "groups": len(values),
                "values": values,
            }
        )
    return rows


def _metadata_coverage_readiness(
    *,
    pairs: Sequence[PairwiseExample],
    metadata_coverage: Sequence[Mapping[str, Any]],
    required_metadata_keys: Sequence[str],
    min_groups_per_key: int,
) -> dict[str, Any]:
    row_by_key = {
        str(row.get("metadata_key", "")): row
        for row in metadata_coverage
        if str(row.get("metadata_key", ""))
    }
    missing_required_keys = sorted(
        key for key in required_metadata_keys if key not in row_by_key
    )
    incomplete_metadata_keys = sorted(
        key
        for key in required_metadata_keys
        if int(_mapping(row_by_key.get(key)).get("missing_pairs", len(pairs))) > 0
    )
    thin_group_keys = sorted(
        key
        for key in required_metadata_keys
        if int(_mapping(row_by_key.get(key)).get("groups", 0))
        < min_groups_per_key
    )
    failed_metadata_keys = sorted(
        set(missing_required_keys) | set(incomplete_metadata_keys) | set(thin_group_keys)
    )
    min_pairs_with_values = min(
        (
            int(_mapping(row_by_key.get(key)).get("pairs_with_values", 0))
            for key in required_metadata_keys
        ),
        default=0,
    )
    min_groups = min(
        (
            int(_mapping(row_by_key.get(key)).get("groups", 0))
            for key in required_metadata_keys
        ),
        default=0,
    )
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(len(pairs)),
            "threshold": 1.0,
            "passed": len(pairs) > 0,
        },
        {
            "gate_id": "required_metadata_keys_present",
            "value": float(len(required_metadata_keys) - len(missing_required_keys)),
            "threshold": float(len(required_metadata_keys)),
            "passed": not missing_required_keys,
        },
        {
            "gate_id": "complete_pair_coverage",
            "value": float(min_pairs_with_values),
            "threshold": float(len(pairs)),
            "passed": not incomplete_metadata_keys,
        },
        {
            "gate_id": "min_groups_per_key",
            "value": float(min_groups),
            "threshold": float(min_groups_per_key),
            "passed": not thin_group_keys,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": (
            "metadata_coverage_ready"
            if ready
            else "not_ready_for_metadata_coverage_claims"
        ),
        "ready": ready,
        "required_metadata_keys": list(required_metadata_keys),
        "min_groups_per_key": min_groups_per_key,
        "missing_required_keys": missing_required_keys,
        "incomplete_metadata_keys": incomplete_metadata_keys,
        "thin_group_keys": thin_group_keys,
        "failed_metadata_keys": failed_metadata_keys,
        "gates": gates,
    }


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
