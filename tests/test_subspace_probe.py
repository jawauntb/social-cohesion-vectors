from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.subspace_probe import (
    render_activation_subspace_probe_markdown,
    run_activation_subspace_probe,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_subspace_probe_reports_signed_and_squared_metrics(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez(
        activation_path,
        activations=np.asarray(
            [
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.1, 0.0],
                [0.0, 0.1, 0.0],
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
    report = run_activation_subspace_probe(
        activation_npz=activation_path,
        pairs=[
            _pair("a1", "axis_a"),
            _pair("a2", "axis_a"),
            _pair("b1", "axis_b"),
            _pair("b2", "axis_b"),
        ],
        metadata_key="mechanism",
        max_components=2,
    )
    markdown = render_activation_subspace_probe_markdown(report)

    assert report["inputs"]["pairs"] == 4
    assert report["summary"]["best_pair_loo_signed_vote_accuracy"] == 1.0
    assert report["summary"]["best_pair_loo_squared_energy_accuracy"] == 1.0
    assert len(report["component_sweeps"]) == 8
    assert (
        report["pair_difference_subspace"]["svd_cumulative_energy"][0][
            "cumulative_energy_fraction"
        ]
        == 0.5
    )
    assert "Squared-energy accuracy is useful for localization" in markdown


def _pair(pair_id: str, mechanism: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata={"mechanism": mechanism},
    )
