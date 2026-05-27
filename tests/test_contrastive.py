from __future__ import annotations

import numpy as np

from social_cohesion_vectors.activations.contrastive import (
    load_direction,
    project_activations,
    save_direction,
    train_contrastive_direction,
    train_direction_from_arrays,
)


def test_train_direction_from_scores_projects_top_above_bottom() -> None:
    activations = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    scores = np.array([0.0, 0.1, 0.9, 1.0])

    direction = train_direction_from_arrays(
        activations,
        scores=scores,
        top_fraction=0.5,
    )
    projections = direction.project(activations)

    assert np.isclose(np.linalg.norm(direction.direction), 1.0)
    assert projections[-1] > projections[0]
    assert np.allclose(direction.direction, np.array([1.0, 0.0]))


def test_train_direction_from_npz_label_fallback(tmp_path) -> None:
    path = tmp_path / "activations.npz"
    np.savez_compressed(
        path,
        activations=np.array(
            [
                [0.0, -2.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ]
        ),
        labels=np.array(["negative", "negative", "positive", "positive"]),
    )

    direction = train_contrastive_direction(path)
    projections = project_activations(np.array([[0.0, -3.0], [0.0, 3.0]]), direction)

    assert direction.metadata["split_method"] == "label"
    assert np.allclose(direction.direction, np.array([0.0, 1.0]))
    assert projections[1] > projections[0]


def test_nan_scores_fall_back_to_labels(tmp_path) -> None:
    path = tmp_path / "activations.npz"
    np.savez_compressed(
        path,
        activations=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        target_scores=np.array([np.nan, np.nan]),
        labels=np.array(["negative", "positive"]),
    )

    direction = train_contrastive_direction(path)

    assert direction.metadata["split_method"] == "label"
    assert np.allclose(direction.direction, np.array([1.0, 0.0]))


def test_save_and_load_direction_round_trip(tmp_path) -> None:
    direction = train_direction_from_arrays(
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        labels=np.array(["negative", "positive"]),
    )
    path = save_direction(tmp_path / "direction.npz", direction)

    loaded = load_direction(path)

    assert loaded.top_count == direction.top_count
    assert loaded.bottom_count == direction.bottom_count
    assert np.allclose(loaded.direction, direction.direction)
