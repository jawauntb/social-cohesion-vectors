"""Run a tiny raw EEG bridge encoder on local stub arrays only."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from social_cohesion_vectors.datasets import read_jsonl

CLAIM_BOUNDARY = (
    "Representation-learning encoder scaffold only; no neural synchrony, "
    "prosociality, social-cohesion, empathy, bonding, cooperation, subjective-state, "
    "or human-intervention claim."
)
MANIFEST_VERSION = "raw-eeg-bridge-manifest-v0"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_raw_eeg_encoder_stub(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_path=args.output,
        feature_field=args.feature_field,
        eval_split=args.eval_split,
        ridge_alpha=args.ridge_alpha,
        shuffle_seed=args.shuffle_seed,
    )
    print(
        "ran raw EEG encoder stub: "
        f"dataset={report['dataset']} "
        f"train_rows={report['train_rows']} "
        f"eval_rows={report['eval_rows']} "
        f"output={args.output}"
    )
    return 0


def run_raw_eeg_encoder_stub(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    output_path: str | Path | None = None,
    feature_field: str = "visual_embedding_path",
    eval_split: str = "test",
    ridge_alpha: float = 1.0,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    """Fit a ridge map from stimulus features to response embeddings."""

    records = read_jsonl(manifest_path)
    train_rows = _eligible_rows(records, split_name="train")
    eval_rows = _eligible_rows(records, split_name=eval_split)
    if not train_rows:
        raise ValueError("manifest has no included train rows")
    if not eval_rows:
        raise ValueError(f"manifest has no included {eval_split!r} rows")

    train_features, train_responses = _load_arrays(
        train_rows,
        data_root=Path(data_root),
        feature_field=feature_field,
    )
    eval_features, eval_responses = _load_arrays(
        eval_rows,
        data_root=Path(data_root),
        feature_field=feature_field,
    )
    _validate_feature_response_dims(
        train_features,
        train_responses,
        eval_features,
        eval_responses,
    )

    weights = _fit_ridge(train_features, train_responses, alpha=ridge_alpha)
    predictions = eval_features @ weights
    mse = _mean_squared_error(eval_responses, predictions)

    shuffled_train_responses = _shuffle_rows(train_responses, seed=shuffle_seed)
    shuffled_weights = _fit_ridge(
        train_features,
        shuffled_train_responses,
        alpha=ridge_alpha,
    )
    shuffled_predictions = eval_features @ shuffled_weights
    shuffle_mse = _mean_squared_error(eval_responses, shuffled_predictions)

    first_record = records[0]
    report = {
        "dataset": first_record.get("dataset"),
        "manifest_version": first_record.get("manifest_version"),
        "feature_field": feature_field,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "feature_dim": int(train_features.shape[1]),
        "response_dim": int(train_responses.shape[1]),
        "ridge_alpha": ridge_alpha,
        "split_policy": _split_policy(eval_rows),
        "eval_split": eval_split,
        "mse": mse,
        "shuffle_mse": shuffle_mse,
        "mse_delta_vs_shuffle": mse - shuffle_mse,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _validate_report(report)
    if output_path is not None:
        _write_report(report, output_path)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a tiny THINGS-EEG2-style feature-to-response encoder on local "
            "stub .npy arrays. This does not read restricted raw EEG or stimuli."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-field", default="visual_embedding_path")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--ridge-alpha", default=1.0, type=float)
    parser.add_argument("--shuffle-seed", default=0, type=int)
    return parser.parse_args(argv)


def _eligible_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if _section(record, "quality").get("include") is True
        and _section(record, "split").get("split_name") == split_name
    ]


def _load_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    feature_field: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features = [
        _load_vector(
            data_root=data_root,
            relative_path=_required_path(_section(row, "features"), feature_field),
        )
        for row in rows
    ]
    responses = [
        _load_vector(
            data_root=data_root,
            relative_path=_required_path(_section(row, "brain_response"), "response_path"),
        )
        for row in rows
    ]
    return np.vstack(features), np.vstack(responses)


def _load_vector(*, data_root: Path, relative_path: str) -> NDArray[np.float64]:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"array path must be relative: {relative_path}")
    full_path = data_root / path
    if not full_path.exists():
        raise ValueError(f"array path does not exist: {relative_path}")
    array = np.load(full_path)
    if array.ndim != 1:
        raise ValueError(f"array must be 1D vector: {relative_path}")
    return np.asarray(array, dtype=np.float64)


def _fit_ridge(
    features: NDArray[np.float64],
    responses: NDArray[np.float64],
    *,
    alpha: float,
) -> NDArray[np.float64]:
    if alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    regularizer = alpha * np.eye(features.shape[1], dtype=np.float64)
    return np.linalg.solve(features.T @ features + regularizer, features.T @ responses)


def _shuffle_rows(
    rows: NDArray[np.float64],
    *,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(rows.shape[0])
    return rows[indices]


def _mean_squared_error(
    expected: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> float:
    return float(np.mean((expected - predicted) ** 2))


def _validate_feature_response_dims(
    train_features: NDArray[np.float64],
    train_responses: NDArray[np.float64],
    eval_features: NDArray[np.float64],
    eval_responses: NDArray[np.float64],
) -> None:
    if train_features.shape[1] != eval_features.shape[1]:
        raise ValueError("train and eval feature dimensions differ")
    if train_responses.shape[1] != eval_responses.shape[1]:
        raise ValueError("train and eval response dimensions differ")


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("manifest_version must be raw-eeg-bridge-manifest-v0")
    if report.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("claim_boundary is missing or changed")


def _section(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest row missing section {name!r}")
    return value


def _required_path(section: Mapping[str, Any], field: str) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row missing path field {field!r}")
    return value


def _split_policy(rows: Sequence[Mapping[str, Any]]) -> str | None:
    policies = {
        policy
        for row in rows
        if isinstance(policy := _section(row, "split").get("split_policy"), str)
        and policy
    }
    if len(policies) > 1:
        raise ValueError(f"eval rows contain multiple split policies: {sorted(policies)}")
    return next(iter(policies), None)


def _write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
