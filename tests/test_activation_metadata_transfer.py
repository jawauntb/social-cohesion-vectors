from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.activation_metadata_transfer import (
    render_activation_metadata_transfer_markdown,
    run_activation_metadata_transfer,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_activation_metadata_transfer_holds_out_pair_groups(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez(
        activation_path,
        activations=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.1],
                [0.0, 0.1],
                [2.0, 0.2],
                [0.0, 0.2],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(["p1", "p1", "p2", "p2", "p3", "p3"], dtype=str),
        labels=np.asarray(
            ["positive", "negative", "positive", "negative", "positive", "negative"],
            dtype=str,
        ),
    )
    pairs = [
        _pair("p1", "a"),
        _pair("p2", "b"),
        _pair("p3", "b"),
    ]

    report = run_activation_metadata_transfer(
        activation_npz=activation_path,
        pairs=pairs,
        metadata_key="primary_fault_class",
    )
    markdown = render_activation_metadata_transfer_markdown(report)

    assert report["summary"]["folds"] == 2
    assert report["summary"]["mean_test_accuracy"] == 1.0
    assert "Activation Metadata Transfer" in markdown


def _pair(pair_id: str, fault_class: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata={"primary_fault_class": fault_class},
    )
