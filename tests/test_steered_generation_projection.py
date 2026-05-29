from __future__ import annotations

import json

import numpy as np

from social_cohesion_vectors.experiments.steered_generation_projection import (
    projection_prompt_records,
    render_generation_projection_markdown,
    summarize_generation_projection,
)


def test_projection_prompt_records_encode_strength_metadata() -> None:
    records = projection_prompt_records(
        [
            {
                "prompt_id": "case",
                "prompt": "Choose a policy.",
                "generated_text": "People can opt out without punishment.",
                "strength": 2.0,
                "cohesion_score": 0.8,
            }
        ],
        report_name="steering_run",
    )

    assert records == [
        {
            "sample_id": "steering_run::prompt=case::strength=2",
            "pair_id": "steering_run::case",
            "label": "positive",
            "target_score": 0.8,
            "text": "People can opt out without punishment.",
        }
    ]


def test_summarize_generation_projection_reports_direction_margins(tmp_path) -> None:
    activation_path = tmp_path / "generated.npz"
    direction_path = tmp_path / "direction.npz"
    np.savez_compressed(
        activation_path,
        activations=np.asarray(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [-2.0, 0.0],
                [0.0, 1.0],
                [-3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        sample_ids=np.asarray(
            [
                "run::prompt=a::strength=-2",
                "run::prompt=a::strength=0",
                "run::prompt=a::strength=2",
                "run::prompt=b::strength=-2",
                "run::prompt=b::strength=0",
                "run::prompt=b::strength=2",
            ],
            dtype=str,
        ),
        pair_ids=np.asarray(
            ["run::a", "run::a", "run::a", "run::b", "run::b", "run::b"],
            dtype=str,
        ),
        labels=np.asarray(
            ["negative", "baseline", "positive", "negative", "baseline", "positive"],
            dtype=str,
        ),
        target_scores=np.asarray([0.4, 0.5, 0.9, 0.8, 0.5, 0.2], dtype=np.float32),
    )
    np.savez_compressed(
        direction_path,
        direction=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    report = summarize_generation_projection(
        activation_npz=activation_path,
        direction_npz=direction_path,
    )
    markdown = render_generation_projection_markdown(report)

    assert json.loads(json.dumps(report))["samples"] == 6
    assert report["rows"][0]["projection_success_rate"] == 0.5
    assert report["rows"][0]["projection_positive_minus_negative_mean_delta"] == 1.0
    assert report["rows"][0]["score_success_rate"] == 0.5
    assert "Steered Generation Projection Check" in markdown
