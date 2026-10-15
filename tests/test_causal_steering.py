from __future__ import annotations

import json
import sys
from pathlib import Path

from social_cohesion_vectors.experiments.causal_steering import (
    default_steering_prompt_records,
    render_steering_markdown,
    shape_steering_report,
    steering_promotion_gate,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_causal_steering_reports import main as summarize_main


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
    assert report["summary"]["slack_positive_vs_negative_success_rate"] == 1.0
    assert report["promotion_gate"]["status"] == "needs_projection_telemetry"
    assert "Causal Activation Steering Smoke" in markdown
    assert "Promotion status: needs_projection_telemetry" in markdown


def test_steering_promotion_gate_marks_projection_to_output_bottleneck() -> None:
    gate = steering_promotion_gate(
        {
            "positive_minus_negative_post_projection_delta": 0.4,
            "mean_absolute_delta_error": 0.02,
            "positive_minus_negative_mean_score_delta": 0.0,
            "positive_minus_negative_mean_slack_delta": 0.0,
            "positive_minus_negative_mean_autonomy_delta": 0.0,
        }
    )

    assert gate["status"] == "projection_to_output_bottleneck"
    assert gate["promoted"] is False
    assert "behavior_delta_too_small" in gate["reasons"]


def test_steering_promotion_gate_blocks_behavior_gain_with_control_regression() -> None:
    gate = steering_promotion_gate(
        {
            "positive_minus_negative_post_projection_delta": 0.4,
            "mean_absolute_delta_error": 0.02,
            "positive_minus_negative_mean_score_delta": 0.03,
            "positive_minus_negative_mean_slack_delta": -0.02,
            "positive_minus_negative_mean_autonomy_delta": 0.01,
        }
    )

    assert gate["status"] == "failed_controls"
    assert "anti_compliance_control_failed" in gate["reasons"]


def test_steering_promotion_gate_promotes_only_when_all_signals_agree() -> None:
    gate = steering_promotion_gate(
        {
            "positive_minus_negative_post_projection_delta": 0.4,
            "mean_absolute_delta_error": 0.02,
            "positive_minus_negative_mean_score_delta": 0.03,
            "positive_minus_negative_mean_slack_delta": 0.03,
            "positive_minus_negative_mean_autonomy_delta": 0.01,
        }
    )

    assert gate["status"] == "success"
    assert gate["promoted"] is True
    assert gate["reasons"] == ["projection_behavior_slack_controls_agree"]


def test_summarize_causal_steering_reports_writes_table(tmp_path) -> None:
    report_path = tmp_path / "steering.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "strengths": [-2.0, 0.0, 2.0],
                    "positive_vs_negative_success_rate": 0.75,
                    "autonomy_positive_vs_negative_success_rate": 0.5,
                    "positive_minus_baseline_mean_score_delta": 0.01,
                    "positive_minus_negative_mean_score_delta": 0.02,
                },
                "promotion_gate": {
                    "status": "projection_to_output_bottleneck",
                    "promoted": False,
                    "reasons": ["behavior_delta_too_small"],
                },
                "records": [
                    {
                        "model_id": "model",
                        "layer": -1,
                        "hook_site": "post",
                        "steering_position": "last",
                        "steering_timing": "generate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"

    summarize_main(
        [
            str(report_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert "Causal Steering Sweep Summary" in markdown_output.read_text(
        encoding="utf-8"
    )
    assert (
        json.loads(json_output.read_text(encoding="utf-8"))["rows"][0]["hook_site"]
        == "post"
    )
    assert "projection_to_output_bottleneck" in markdown_output.read_text(
        encoding="utf-8"
    )
