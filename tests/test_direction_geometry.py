from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.direction_geometry import (
    render_direction_geometry_audit_markdown,
    run_direction_geometry_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_direction_geometry_reports_anti_aligned_same_axis(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez(
        activation_path,
        activations=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [-2.0, 0.0],
                [0.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(["a", "a", "b", "b", "c", "c"], dtype=str),
        labels=np.asarray(
            ["positive", "negative", "positive", "negative", "positive", "negative"],
            dtype=str,
        ),
    )
    report = run_direction_geometry_audit(
        activation_npz=activation_path,
        pairs=[
            _pair("a", "repair"),
            _pair("b", "anti_repair"),
            _pair("c", "truth"),
        ],
    )
    markdown = render_direction_geometry_audit_markdown(report)

    assert report["summary"]["strongly_anti_aligned_pairs"] == 1
    assert report["summary"]["high_absolute_pairs"] == 1
    assert report["comparisons"][0]["relation"] == "strongly_anti_aligned"
    assert "A low mean signed cosine is not enough" in markdown


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
