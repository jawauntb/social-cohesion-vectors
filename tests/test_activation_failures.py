from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.activation_failures import (
    analyze_leave_one_pair_out_failures,
    render_markdown,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_analyze_leave_one_pair_out_failures_groups_failure_styles() -> None:
    pairs = [
        _pair(f"s{index}::good__truth_first_repair__offline::over::bad__adversarial_escalation__offline")
        for index in range(1, 6)
    ]
    pairs.append(
        _pair(
            "s6::good__truth_first_repair__offline::over::bad__pseudo_cohesion_compliance__offline"
        )
    )
    pair_ids = np.asarray(
        [pair_id for pair in pairs for pair_id in (pair.pair_id, pair.pair_id)],
        dtype=str,
    )
    labels = np.asarray(["positive", "negative"] * len(pairs), dtype=str)
    activations = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.1],
            [0.0, 0.1],
            [1.0, 0.2],
            [0.0, 0.2],
            [1.0, 0.3],
            [0.0, 0.3],
            [1.0, 0.4],
            [0.0, 0.4],
            [-1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    report = analyze_leave_one_pair_out_failures(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        pairs=pairs,
        activation_npz="fake.npz",
    )

    assert report["n_pairs"] == 6
    assert report["n_failures"] == 1
    assert report["failure_styles"] == {"pseudo_cohesion_compliance": 1}
    assert report["worst_failures"][0]["negative_style"] == (
        "pseudo_cohesion_compliance"
    )

    markdown = render_markdown(report)
    assert "# Activation Failure Analysis" in markdown
    assert "pseudo_cohesion_compliance" in markdown


def _pair(pair_id: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id.split("::", 1)[0],
        positive_run_id=pair_id.split("::")[1],
        negative_run_id=pair_id.split("over::", 1)[1],
        positive_text="positive",
        negative_text="negative",
        positive_score=0.9,
        negative_score=0.3,
    )
