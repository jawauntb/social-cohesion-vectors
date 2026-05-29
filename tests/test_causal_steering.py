from __future__ import annotations

from social_cohesion_vectors.experiments.causal_steering import (
    default_steering_prompt_records,
    render_steering_markdown,
    shape_steering_report,
)


def test_default_steering_prompt_records_are_well_formed() -> None:
    records = default_steering_prompt_records()

    assert len(records) == 6
    assert all(record["prompt_id"] for record in records)
    assert all(record["mechanism"] for record in records)
    assert all(record["text"] for record in records)


def test_shape_steering_report_scores_positive_polarity() -> None:
    records = [
        {
            "prompt_id": "case",
            "mechanism": "consent",
            "prompt": "Write a decision note.",
            "strength": -2.0,
            "generated_text": (
                "Everyone must comply now. Saying no is betrayal and there is "
                "no appeal."
            ),
        },
        {
            "prompt_id": "case",
            "mechanism": "consent",
            "prompt": "Write a decision note.",
            "strength": 0.0,
            "generated_text": "The group should coordinate on the next decision.",
        },
        {
            "prompt_id": "case",
            "mechanism": "consent",
            "prompt": "Write a decision note.",
            "strength": 2.0,
            "generated_text": (
                "The group can review evidence, appeal the plan, and decline "
                "roles without punishment."
            ),
        },
    ]

    report = shape_steering_report(records)
    markdown = render_steering_markdown(report)

    assert report["summary"]["positive_vs_negative_success_rate"] == 1.0
    assert report["summary"]["positive_minus_negative_mean_score_delta"] > 0.0
    assert "Causal Activation Steering Smoke" in markdown
