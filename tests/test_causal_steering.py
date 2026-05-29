from __future__ import annotations

import json
import sys
from pathlib import Path

from social_cohesion_vectors.experiments.causal_steering import (
    default_steering_prompt_records,
    render_steering_markdown,
    shape_steering_report,
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
    assert "Causal Activation Steering Smoke" in markdown


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
