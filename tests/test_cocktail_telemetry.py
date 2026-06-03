from __future__ import annotations

from social_cohesion_vectors.modal_app.functions.activation_steering import (
    summarize_component_projection_telemetry,
)


def test_component_projection_telemetry_summarizes_actual_direction_movement() -> None:
    summaries = summarize_component_projection_telemetry(
        [
            {
                "component_id": "ck1",
                "strength": 0.5,
                "layer": -1,
                "hook_site": "post",
                "steering_position": "last",
                "steering_timing": "generate",
                "steering_schedule": "first-2",
                "telemetry": [
                    {
                        "strength": 0.5,
                        "tokens_steered": 1,
                        "mean_projection_before": 1.0,
                        "mean_projection_after": 1.5,
                        "mean_projection_delta": 0.5,
                    },
                    {
                        "strength": 0.25,
                        "tokens_steered": 1,
                        "mean_projection_before": 1.5,
                        "mean_projection_after": 1.75,
                        "mean_projection_delta": 0.25,
                    },
                ],
            },
            {
                "component_id": "guardrail",
                "strength": 0.2,
                "layer": -2,
                "hook_site": "pre",
                "steering_position": "all",
                "steering_timing": "prefill",
                "steering_schedule": "constant",
                "telemetry": [],
            },
        ]
    )

    assert summaries[0] == {
        "component_id": "ck1",
        "layer": -1,
        "hook_site": "post",
        "steering_position": "last",
        "steering_timing": "generate",
        "steering_schedule": "first-2",
        "configured_strength": 0.5,
        "event_count": 2,
        "tokens_steered": 2,
        "mean_effective_strength": 0.375,
        "mean_projection_before": 1.25,
        "mean_projection_after": 1.625,
        "mean_projection_delta": 0.375,
        "total_projection_delta": 0.75,
        "mean_projection_delta_error": 0.0,
    }
    assert summaries[1]["event_count"] == 0
    assert summaries[1]["mean_projection_delta"] == 0.0
