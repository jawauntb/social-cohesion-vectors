from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.guardrail_monitoring import (
    axis_id_from_pair_id,
    render_guardrail_monitoring_markdown,
    run_guardrail_monitoring,
)


def test_guardrail_monitoring_trains_per_axis_and_flags_inversions() -> None:
    pair_ids = [
        "trait-axis::truth_vs_deception::a",
        "trait-axis::truth_vs_deception::a",
        "trait-axis::truth_vs_deception::b",
        "trait-axis::truth_vs_deception::b",
        "trait-axis::autonomy_vs_coercion::a",
        "trait-axis::autonomy_vs_coercion::a",
        "trait-axis::autonomy_vs_coercion::b",
        "trait-axis::autonomy_vs_coercion::b",
    ]
    labels = [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    activations = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.1],
            [0.0, 0.1],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.1, 2.0],
            [0.1, 0.0],
        ],
        dtype=np.float64,
    )

    report = run_guardrail_monitoring(
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
    )
    markdown = render_guardrail_monitoring_markdown(report)

    assert axis_id_from_pair_id("trait-axis::truth_vs_deception::a") == (
        "truth_vs_deception"
    )
    assert report["summary"]["axes"] == 2
    assert report["summary"]["pairs"] == 4
    assert report["summary"]["alerts"] == 0
    assert "Guardrail Monitoring" in markdown
