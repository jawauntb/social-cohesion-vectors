from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.cross_model_alignment import (
    render_cross_model_alignment_audit_markdown,
    run_cross_model_alignment_audit_from_files,
)


def test_cross_model_alignment_maps_direction_between_spaces(tmp_path) -> None:
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "target.npz"
    source = np.asarray(
        [
            [3.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [4.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [2.5, 0.0, -1.0],
            [0.5, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    transform = np.asarray(
        [
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=np.float64,
    )
    target = source @ transform
    sample_ids = np.asarray(
        [
            "pair-a:positive",
            "pair-a:negative",
            "pair-b:positive",
            "pair-b:negative",
            "pair-c:positive",
            "pair-c:negative",
        ],
        dtype=str,
    )
    pair_ids = np.asarray(["pair-a", "pair-a", "pair-b", "pair-b", "pair-c", "pair-c"])
    labels = np.asarray(["positive", "negative"] * 3, dtype=str)
    np.savez(
        source_path,
        activations=source,
        sample_ids=sample_ids,
        pair_ids=pair_ids,
        labels=labels,
    )
    shuffled = np.asarray([2, 3, 0, 1, 4, 5])
    np.savez(
        target_path,
        activations=target[shuffled],
        sample_ids=sample_ids[shuffled],
        pair_ids=pair_ids[shuffled],
        labels=labels[shuffled],
    )

    report = run_cross_model_alignment_audit_from_files(
        source_activation_npz=source_path,
        target_activation_npz=target_path,
        knn_k=2,
        ridge_alpha=1e-6,
    )
    markdown = render_cross_model_alignment_audit_markdown(report)

    assert report["inputs"]["shared_samples"] == 6
    assert report["alignment"]["linear_cka"] > 0.8
    assert (
        report["source_to_target_direction_transfer"]["mapped_source_direction"][
            "pairwise_accuracy"
        ]
        == 1.0
    )
    assert (
        report["source_to_target_direction_transfer"][
            "leave_one_pair_out_mapped_source_direction"
        ]["pairwise_accuracy"]
        == 1.0
    )
    assert report["summary"]["source_to_target_direction_cosine"] > 0.8
    assert "ideal social-cohesion representation" in markdown


def test_cross_model_alignment_falls_back_to_pair_label_keys(tmp_path) -> None:
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "target.npz"
    pair_ids = np.asarray(["p1", "p1", "p2", "p2"], dtype=str)
    labels = np.asarray(["positive", "negative", "positive", "negative"], dtype=str)
    source = np.asarray([[2.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    target = np.asarray([[0.0, 2.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    np.savez(source_path, activations=source, pair_ids=pair_ids, labels=labels)
    np.savez(target_path, activations=target, pair_ids=pair_ids, labels=labels)

    report = run_cross_model_alignment_audit_from_files(
        source_activation_npz=source_path,
        target_activation_npz=target_path,
        knn_k=1,
    )

    assert report["inputs"]["shared_samples"] == 4
    assert report["source_to_target_direction_transfer"]["mapped_source_direction"][
        "pairs"
    ] == 2
