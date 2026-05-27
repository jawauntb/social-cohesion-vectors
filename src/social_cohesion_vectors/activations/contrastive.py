"""Numpy-only utilities for simple contrastive activation directions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_ACTIVATION_KEYS = (
    "activations",
    "pooled_activations",
    "vectors",
    "activation_vectors",
)
_SCORE_KEYS = ("target_scores", "scores", "score")
_LABEL_KEYS = ("labels", "label")
_POSITIVE_LABELS = ("positive", "top", "cooperative", "high")
_NEGATIVE_LABELS = ("negative", "bottom", "adversarial", "low")
_EPSILON = 1e-12

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class ActivationBatch:
    """Activation vectors and optional supervision loaded from one ``.npz`` file."""

    activations: np.ndarray
    scores: np.ndarray | None = None
    labels: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    source_path: str | None = None

    @property
    def count(self) -> int:
        return int(self.activations.shape[0])

    @property
    def dim(self) -> int:
        return int(self.activations.shape[1])


@dataclass(frozen=True)
class ContrastiveDirection:
    """A normalized direction pointing from bottom examples toward top examples."""

    direction: np.ndarray
    top_mean: np.ndarray
    bottom_mean: np.ndarray
    top_count: int
    bottom_count: int
    activation_count: int
    source_paths: tuple[str, ...] = ()
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.direction.shape[0])

    def project(self, activations: np.ndarray | Sequence[float]) -> np.ndarray:
        return project_activations(activations, self)


def load_activation_file(
    path: str | Path,
    *,
    activation_key: str | None = None,
    score_key: str | None = None,
    label_key: str | None = None,
) -> ActivationBatch:
    """Load pooled activation vectors and supervision metadata from a ``.npz``."""

    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        activations = _array_from_npz(data, activation_key, _ACTIVATION_KEYS)
        scores = _optional_array_from_npz(data, score_key, _SCORE_KEYS)
        labels = _optional_array_from_npz(data, label_key, _LABEL_KEYS)
        sample_ids = _optional_array_from_npz(data, None, ("sample_ids", "ids"))

    batch = ActivationBatch(
        activations=_ensure_activation_matrix(activations),
        scores=_ensure_optional_vector(scores, "scores"),
        labels=_ensure_optional_vector(labels, "labels"),
        sample_ids=_ensure_optional_vector(sample_ids, "sample_ids"),
        source_path=str(npz_path),
    )
    _validate_batch_lengths(batch)
    return batch


def train_contrastive_direction(
    paths: str | Path | Iterable[str | Path],
    *,
    activation_key: str | None = None,
    score_key: str | None = None,
    label_key: str | None = None,
    top_fraction: float = 0.25,
    positive_labels: Sequence[str] = _POSITIVE_LABELS,
    negative_labels: Sequence[str] = _NEGATIVE_LABELS,
) -> ContrastiveDirection:
    """Train a normalized top-vs-bottom direction from one or more ``.npz`` files."""

    path_list = _coerce_paths(paths)
    batches = [
        load_activation_file(
            path,
            activation_key=activation_key,
            score_key=score_key,
            label_key=label_key,
        )
        for path in path_list
    ]
    if not batches:
        raise ValueError("At least one activation file is required.")

    activations = np.concatenate([batch.activations for batch in batches], axis=0)
    scores = _concatenate_optional([batch.scores for batch in batches])
    labels = _concatenate_optional([batch.labels for batch in batches])
    return train_direction_from_arrays(
        activations,
        scores=scores,
        labels=labels,
        top_fraction=top_fraction,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        source_paths=tuple(str(path) for path in path_list),
    )


def train_direction_from_arrays(
    activations: np.ndarray | Sequence[Sequence[float]],
    *,
    scores: np.ndarray | Sequence[float] | None = None,
    labels: np.ndarray | Sequence[str] | None = None,
    top_fraction: float = 0.25,
    positive_labels: Sequence[str] = _POSITIVE_LABELS,
    negative_labels: Sequence[str] = _NEGATIVE_LABELS,
    source_paths: tuple[str, ...] = (),
) -> ContrastiveDirection:
    """Train a direction from activation arrays plus scores or labels."""

    matrix = _ensure_activation_matrix(np.asarray(activations, dtype=np.float64))
    if scores is not None and _finite_scores(scores):
        _validate_supervision_length("scores", scores, matrix.shape[0])
        top_indices, bottom_indices = _top_bottom_indices(scores, top_fraction)
        split_method = "score"
    elif labels is not None:
        _validate_supervision_length("labels", labels, matrix.shape[0])
        top_indices, bottom_indices = _label_indices(
            labels,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )
        split_method = "label"
    elif scores is not None:
        _top_bottom_indices(scores, top_fraction)
        raise AssertionError("unreachable")
    else:
        raise ValueError("Provide either scores or labels to define top/bottom groups.")

    _validate_indices(top_indices, bottom_indices, matrix.shape[0])
    top_mean = matrix[top_indices].mean(axis=0)
    bottom_mean = matrix[bottom_indices].mean(axis=0)
    direction = _normalize_vector(top_mean - bottom_mean)
    return ContrastiveDirection(
        direction=direction,
        top_mean=top_mean,
        bottom_mean=bottom_mean,
        top_count=int(top_indices.size),
        bottom_count=int(bottom_indices.size),
        activation_count=int(matrix.shape[0]),
        source_paths=source_paths,
        metadata={
            "split_method": split_method,
            "top_fraction": float(top_fraction),
        },
    )


def project_activations(
    activations: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    direction: ContrastiveDirection | np.ndarray | Sequence[float],
) -> np.ndarray:
    """Project activation vectors onto a trained direction."""

    matrix = _ensure_activation_matrix(np.asarray(activations, dtype=np.float64))
    direction_array = (
        direction.direction
        if isinstance(direction, ContrastiveDirection)
        else np.asarray(direction, dtype=np.float64)
    )
    if direction_array.ndim != 1:
        raise ValueError("Direction must be a one-dimensional vector.")
    if matrix.shape[1] != direction_array.shape[0]:
        raise ValueError(
            f"Activation dimension {matrix.shape[1]} does not match direction "
            f"dimension {direction_array.shape[0]}."
        )
    return matrix @ direction_array


def save_direction(path: str | Path, direction: ContrastiveDirection) -> Path:
    """Persist a contrastive direction as a compressed ``.npz`` file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        direction=direction.direction,
        top_mean=direction.top_mean,
        bottom_mean=direction.bottom_mean,
        top_count=np.array(direction.top_count, dtype=np.int64),
        bottom_count=np.array(direction.bottom_count, dtype=np.int64),
        activation_count=np.array(direction.activation_count, dtype=np.int64),
        source_paths=np.array(direction.source_paths, dtype=str),
        metadata_json=np.array(json.dumps(dict(direction.metadata), sort_keys=True)),
    )
    return output_path


def load_direction(path: str | Path) -> ContrastiveDirection:
    """Load a direction written by :func:`save_direction`."""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata_json = str(data["metadata_json"].item())
        metadata = json.loads(metadata_json) if metadata_json else {}
        return ContrastiveDirection(
            direction=_normalize_vector(
                np.asarray(data["direction"], dtype=np.float64)
            ),
            top_mean=np.asarray(data["top_mean"], dtype=np.float64),
            bottom_mean=np.asarray(data["bottom_mean"], dtype=np.float64),
            top_count=int(data["top_count"].item()),
            bottom_count=int(data["bottom_count"].item()),
            activation_count=int(data["activation_count"].item()),
            source_paths=tuple(str(path) for path in data["source_paths"]),
            metadata=metadata,
        )


def _coerce_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _array_from_npz(
    data: Any,
    preferred_key: str | None,
    fallback_keys: Sequence[str],
) -> np.ndarray:
    if preferred_key is not None:
        if preferred_key not in data:
            raise KeyError(f"Activation file is missing key {preferred_key!r}.")
        return np.asarray(data[preferred_key])

    for key in fallback_keys:
        if key in data:
            return np.asarray(data[key])
    joined = ", ".join(repr(key) for key in fallback_keys)
    raise KeyError(f"Activation file is missing one of: {joined}.")


def _optional_array_from_npz(
    data: Any,
    preferred_key: str | None,
    fallback_keys: Sequence[str],
) -> np.ndarray | None:
    if preferred_key is not None:
        return np.asarray(data[preferred_key]) if preferred_key in data else None
    for key in fallback_keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _ensure_activation_matrix(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Activations must be a one- or two-dimensional array.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Activations must be non-empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Activations contain non-finite values.")
    return matrix


def _ensure_optional_vector(array: np.ndarray | None, name: str) -> np.ndarray | None:
    if array is None:
        return None
    vector = np.asarray(array)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    elif vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return vector


def _validate_batch_lengths(batch: ActivationBatch) -> None:
    for name, vector in [
        ("scores", batch.scores),
        ("labels", batch.labels),
        ("sample_ids", batch.sample_ids),
    ]:
        if vector is not None and vector.shape[0] != batch.count:
            raise ValueError(
                f"{name} length {vector.shape[0]} does not match activation "
                f"count {batch.count} in {batch.source_path}."
            )


def _concatenate_optional(parts: Sequence[np.ndarray | None]) -> np.ndarray | None:
    present_parts = [part for part in parts if part is not None]
    if len(present_parts) != len(parts):
        return None
    return np.concatenate(present_parts, axis=0)


def _top_bottom_indices(
    scores: np.ndarray | Sequence[float],
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be in the interval (0, 0.5].")
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size < 2:
        raise ValueError("At least two scored activations are required.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("Scores contain non-finite values.")

    group_count = max(1, int(np.ceil(score_array.size * top_fraction)))
    group_count = min(group_count, score_array.size // 2)
    order = np.argsort(score_array, kind="stable")
    return order[-group_count:], order[:group_count]


def _finite_scores(scores: np.ndarray | Sequence[float]) -> bool:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    return bool(score_array.size > 0 and np.all(np.isfinite(score_array)))


def _validate_supervision_length(
    name: str,
    values: np.ndarray | Sequence[float] | Sequence[str],
    activation_count: int,
) -> None:
    count = np.asarray(values).reshape(-1).shape[0]
    if count != activation_count:
        raise ValueError(
            f"{name} length {count} does not match activation count {activation_count}."
        )


def _label_indices(
    labels: np.ndarray | Sequence[str],
    *,
    positive_labels: Sequence[str],
    negative_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.array([str(label).strip().lower() for label in labels])
    positives = {label.strip().lower() for label in positive_labels}
    negatives = {label.strip().lower() for label in negative_labels}
    top_indices = np.flatnonzero(np.isin(normalized, list(positives)))
    bottom_indices = np.flatnonzero(np.isin(normalized, list(negatives)))
    return top_indices, bottom_indices


def _validate_indices(
    top_indices: np.ndarray,
    bottom_indices: np.ndarray,
    activation_count: int,
) -> None:
    if top_indices.size == 0:
        raise ValueError("No top/positive activations were selected.")
    if bottom_indices.size == 0:
        raise ValueError("No bottom/negative activations were selected.")
    if (
        np.max(top_indices) >= activation_count
        or np.max(bottom_indices) >= activation_count
    ):
        raise ValueError("Selected group indices exceed activation count.")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        raise ValueError(
            "Cannot train a direction from identical top and bottom means."
        )
    return vector / norm
