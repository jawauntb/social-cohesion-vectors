from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_ck3_transition_records import main  # noqa: E402

from social_cohesion_vectors.transition_records import (
    transition_records_from_ck_records,
    transition_records_from_ck_report,
)  # noqa: E402


def test_transition_records_pair_non_baseline_records_by_prompt() -> None:
    transitions = transition_records_from_ck_records(_ck_records())

    assert [record["transition_id"] for record in transitions] == [
        "case-a::baseline->ck1_guardrail_cocktail",
        "case-a::baseline->guardrails_only",
    ]
    cocktail = transitions[0]
    assert cocktail["baseline_state"]["recipe_id"] == "baseline"
    assert cocktail["perturbation"]["recipe_id"] == "ck1_guardrail_cocktail"
    assert cocktail["dose"] == {
        "component_count": 2,
        "component_strengths": [
            {"component_id": "ck1", "strength": 0.5},
            {"component_id": "anti_sycophancy", "strength": 0.25},
        ],
        "absolute_strength_sum": 0.75,
    }
    assert cocktail["site"] == [
        {
            "component_id": "ck1",
            "layer": -1,
            "hook_site": "post",
            "steering_position": "last",
        },
        {
            "component_id": "anti_sycophancy",
            "layer": -1,
            "hook_site": "post",
            "steering_position": "last",
        },
    ]
    assert cocktail["timing"] == [
        {
            "component_id": "ck1",
            "steering_timing": "generate",
            "steering_schedule": "first-4",
        },
        {
            "component_id": "anti_sycophancy",
            "steering_timing": "generate",
            "steering_schedule": "after-4",
        },
    ]
    assert cocktail["effect_class"] == "beneficial_transition"
    assert cocktail["observed_transition"]["ck1_score_delta"] == 0.12
    assert cocktail["observed_transition"]["safe_attunement_signal_delta"] == 0.15
    assert cocktail["observed_transition"]["pseudo_attunement_risk_delta"] == -0.05
    assert cocktail["side_effects"]["observed"] is False
    assert cocktail["antagonist"]["component_ids"] == ["anti_sycophancy"]
    assert cocktail["washout"]["status"] == "not_measured"


def test_transition_records_from_report_uses_summary_baseline_and_context() -> None:
    report = {
        "experiment": "ck4_parallel_lane_cocktail",
        "description": "compute-only report",
        "summary": {"baseline_recipe_id": "vehicle"},
        "records": [
            {**_baseline_record(), "recipe_id": "vehicle"},
            _guardrail_record(),
        ],
    }

    transitions = transition_records_from_ck_report(report, source_id="report.json")

    assert len(transitions) == 1
    context = transitions[0]["replication_context"]
    assert transitions[0]["transition_id"] == "case-a::vehicle->guardrails_only"
    assert context["source_id"] == "report.json"
    assert context["source_experiment"] == "ck4_parallel_lane_cocktail"
    assert context["source_record_count"] == 2
    assert context["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_export_script_writes_jsonl_from_report(tmp_path) -> None:
    report_path = tmp_path / "ck3_report.json"
    output_path = tmp_path / "transition_records.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "experiment": "ck3_cocktail_steering",
                "summary": {"baseline_recipe_id": "baseline"},
                "records": _ck_records(),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main([str(report_path), "--output", str(output_path)])

    assert exit_code == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["transition_id"] == (
        "case-a::baseline->ck1_guardrail_cocktail"
    )


def _ck_records() -> list[dict[str, object]]:
    return [
        _baseline_record(),
        _guardrail_record(),
        {
            "prompt_id": "case-a",
            "recipe_id": "ck1_guardrail_cocktail",
            "recipe_label": "Low-dose cocktail",
            "prompt": "Write a repair message.",
            "generated_text": "Name the evidence and preserve exit rights.",
            "ck1_score": 0.72,
            "score_components": {
                "safe_attunement_signal": 0.35,
                "pseudo_attunement_risk": 0.05,
            },
            "components": [
                {
                    "component_id": "ck1",
                    "layer": -1,
                    "strength": 0.5,
                    "hook_site": "post",
                    "steering_position": "last",
                    "steering_timing": "generate",
                    "steering_schedule": "first-4",
                },
                {
                    "component_id": "anti_sycophancy",
                    "layer": -1,
                    "strength": 0.25,
                    "hook_site": "post",
                    "steering_position": "last",
                    "steering_timing": "generate",
                    "steering_schedule": "after-4",
                },
            ],
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "seed": 0,
        },
    ]


def _baseline_record() -> dict[str, object]:
    return {
        "prompt_id": "case-a",
        "phase": "repair",
        "mechanism": "boundary_preservation",
        "recipe_id": "baseline",
        "recipe_label": "Baseline",
        "prompt": "Write a repair message.",
        "generated_text": "Coordinate the next message.",
        "ck1_score": 0.6,
        "score_components": {
            "safe_attunement_signal": 0.2,
            "pseudo_attunement_risk": 0.1,
        },
        "components": [],
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "seed": 0,
    }


def _guardrail_record() -> dict[str, object]:
    return {
        "prompt_id": "case-a",
        "recipe_id": "guardrails_only",
        "recipe_label": "Guardrails only",
        "prompt": "Write a repair message.",
        "generated_text": "Preserve privacy and verify claims.",
        "ck1_score": 0.64,
        "score_components": {
            "safe_attunement_signal": 0.25,
            "pseudo_attunement_risk": 0.12,
        },
        "components": [
            {
                "component_id": "privacy_exit",
                "layer": -1,
                "strength": 0.35,
                "hook_site": "post",
                "steering_position": "last",
                "steering_timing": "generate",
                "steering_schedule": "constant",
            }
        ],
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "seed": 0,
    }
