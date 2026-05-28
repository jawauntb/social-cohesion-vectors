from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.residual_subspace_audit import (
    render_residual_subspace_audit_markdown,
    run_residual_subspace_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_residual_group_directions_find_signal_after_global_projection(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez(
        activation_path,
        activations=np.asarray(
            [
                [3.0, 0.0],
                [0.0, 0.0],
                [3.1, 0.0],
                [0.1, 0.0],
                [0.0, 3.0],
                [0.0, 0.0],
                [0.0, 3.1],
                [0.0, 0.1],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(
            ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
            dtype=str,
        ),
        labels=np.asarray(
            [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
            dtype=str,
        ),
    )
    report = run_residual_subspace_audit(
        activation_npz=activation_path,
        pairs=[
            _pair("a1", "x_fault"),
            _pair("a2", "x_fault"),
            _pair("b1", "y_fault"),
            _pair("b2", "y_fault"),
        ],
    )
    markdown = render_residual_subspace_audit_markdown(report)

    assert report["summary"]["global_accuracy"] == 1.0
    assert report["summary"]["residual_global_accuracy"] < 1.0
    assert report["summary"]["residual_group_mean_accuracy"] == 1.0
    assert report["summary"]["residual_groups_with_positive_signal"] == 2
    assert (
        report["pair_difference_subspace"]["residual_energy_fraction"] > 0.0
    )
    assert "single-direction ablation does not show" in markdown


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
