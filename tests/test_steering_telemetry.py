from __future__ import annotations

from social_cohesion_vectors.experiments.steering_telemetry import (
    render_telemetry_markdown,
    shape_steering_telemetry_report,
)


def test_shape_steering_telemetry_report_tracks_hidden_delta_and_score() -> None:
    traces = [
        _trace(strength=-2.0, before=1.0, after=-1.0, text="No one may appeal."),
        _trace(strength=0.0, before=1.0, after=1.0, text="The group coordinates."),
        _trace(
            strength=2.0,
            before=1.0,
            after=3.0,
            text="People can review evidence and decline roles.",
        ),
    ]

    report = shape_steering_telemetry_report(traces)
    markdown = render_telemetry_markdown(report)

    assert report["summary"]["mean_absolute_delta_error"] == 0.0
    assert report["summary"]["positive_minus_negative_post_projection_delta"] == 4.0
    assert report["summary"]["positive_minus_negative_score_delta"] > 0.0
    assert report["summary"]["positive_minus_negative_mean_slack_delta"] > 0.0
    assert report["promotion_gate"]["status"] == "success"
    assert report["traces"][0]["model_id"] == "test-model"
    assert report["traces"][0]["layer"] == -2
    assert "Steering Hidden Projection Telemetry" in markdown
    assert "Promotion status: success" in markdown


def test_steering_telemetry_report_marks_projection_bottleneck() -> None:
    traces = [
        _trace(strength=-2.0, before=1.0, after=-1.0, text="The group coordinates."),
        _trace(strength=0.0, before=1.0, after=1.0, text="The group coordinates."),
        _trace(strength=2.0, before=1.0, after=3.0, text="The group coordinates."),
    ]

    report = shape_steering_telemetry_report(traces)

    assert report["summary"]["positive_minus_negative_post_projection_delta"] == 4.0
    assert report["promotion_gate"]["status"] == "projection_to_output_bottleneck"


def _trace(
    *,
    strength: float,
    before: float,
    after: float,
    text: str,
) -> dict[str, object]:
    return {
        "prompt_id": "case",
        "mechanism": "consent",
        "prompt": "Write a decision note.",
        "model_id": "test-model",
        "layer": -2,
        "hook_site": "post",
        "steering_position": "last",
        "steering_timing": "generate",
        "strength": strength,
        "generated_text": text,
        "steps": [{"step": 0, "token_text": "The"}],
        "events": [
            {
                "mean_projection_before": before,
                "mean_projection_after": after,
                "mean_projection_delta": after - before,
            }
        ],
    }
