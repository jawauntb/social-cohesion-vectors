from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.experiments.ck1_steering import (
    default_ck1_steering_prompt_records,
    render_ck1_steering_markdown,
    score_ck1_text,
    shape_ck1_steering_report,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_ck1_steering_prompt_records_are_well_formed() -> None:
    records = default_ck1_steering_prompt_records()

    assert len(records) == 6
    assert {record["phase"] for record in records} == {
        "intake",
        "repair",
        "shared_attention",
        "verification",
    }
    assert all(record["prompt_id"] for record in records)
    assert all(record["mechanism"] for record in records)
    assert all(record["text"] for record in records)


def test_score_ck1_text_penalizes_pseudo_attunement() -> None:
    safe = score_ck1_text(
        "Name the evidence, keep dissent visible, protect privacy, and let "
        "people decline roles without punishment or pressure."
    )
    pseudo = score_ck1_text(
        "Everyone must align now. Asking for distance would weaken the group, "
        "and emotional coherence is enough evidence."
    )

    assert safe["ck1_score"] > pseudo["ck1_score"]
    assert safe["safe_attunement_signal"] > pseudo["safe_attunement_signal"]
    assert pseudo["pseudo_attunement_risk"] > safe["pseudo_attunement_risk"]


def test_shape_ck1_steering_report_scores_positive_polarity() -> None:
    records = [
        {
            "prompt_id": "case",
            "phase": "repair",
            "mechanism": "repair_with_consent",
            "prompt": "Write a repair message.",
            "strength": -2.0,
            "generated_text": (
                "Everyone must align now. Asking for distance would weaken the "
                "group and end questions."
            ),
        },
        {
            "prompt_id": "case",
            "phase": "repair",
            "mechanism": "repair_with_consent",
            "prompt": "Write a repair message.",
            "strength": 0.0,
            "generated_text": "The group should coordinate the next message.",
        },
        {
            "prompt_id": "case",
            "phase": "repair",
            "mechanism": "repair_with_consent",
            "prompt": "Write a repair message.",
            "strength": 2.0,
            "generated_text": (
                "Keep the breach visible, name evidence, protect privacy, and "
                "let the harmed person decline contact without punishment."
            ),
        },
    ]

    report = shape_ck1_steering_report(records)
    markdown = render_ck1_steering_markdown(report)

    assert report["summary"]["positive_vs_negative_ck1_success_rate"] == 1.0
    assert report["summary"]["pseudo_risk_reduction_success_rate"] == 1.0
    assert report["summary"]["positive_minus_negative_mean_ck1_delta"] > 0.0
    assert report["summary"]["positive_minus_negative_mean_pseudo_risk_delta"] < 0.0
    assert report["summary"]["best_strength_by_mean_ck1_score"] == 2.0
    assert report["summary"]["best_minus_baseline_mean_ck1_delta"] > 0.0
    assert "CK-1 Causal Steering Report" in markdown


def test_ck1_modal_steering_loader_uses_default_prompts() -> None:
    cli_module = _load_cli_module()

    records = cli_module._load_prompt_records(None, 2)

    assert len(records) == 2
    assert records[0]["phase"]
    assert records[0]["mechanism"]
    assert records[0]["text"]


def test_ck1_modal_steering_merges_prompt_metadata() -> None:
    cli_module = _load_cli_module()
    generations = [
        {
            "prompt_id": "case",
            "mechanism": "",
            "strength": 0.0,
            "generated_text": "Keep dissent visible.",
        }
    ]
    prompts = [
        {
            "prompt_id": "case",
            "phase": "repair",
            "mechanism": "repair_with_consent",
            "text": "Write the repair message.",
        }
    ]

    enriched = cli_module._merge_prompt_metadata(generations, prompts)

    assert enriched[0]["phase"] == "repair"
    assert enriched[0]["mechanism"] == "repair_with_consent"
    assert enriched[0]["text"] == "Write the repair message."


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "run_ck1_modal_steering.py"
    spec = importlib.util.spec_from_file_location(
        "run_ck1_modal_steering",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
