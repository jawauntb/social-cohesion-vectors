from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.accountability_reference_perturbation import (
    ACCOUNTABILITY_PERTURBATION_SPECS,
    accountability_reference_perturbation_pairs,
    accountability_reference_perturbation_report,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample


def test_accountability_reference_perturbations_are_schema_valid() -> None:
    generated = _generated_pair()
    pairs = accountability_reference_perturbation_pairs(generated)

    assert len(pairs) == len(ACCOUNTABILITY_PERTURBATION_SPECS)
    assert all(PairwiseExample.model_validate(pair.model_dump()) for pair in pairs)
    assert {pair.metadata["original_generated_pair_id"] for pair in pairs} == {
        generated.pair_id
    }
    assert {pair.metadata["base_contrast_id"] for pair in pairs} == {
        "accountability_after_harm"
    }
    assert {
        "original_reference",
        "positive_address_removed",
        "positive_first_sentence_removed",
        "positive_framing_removed_address_kept",
        "positive_framing_neutral_replacement",
        "positive_first_sentence_removed_neutral_padding",
        "positive_framing_length_control",
        "positive_warmth_neutralized",
        "positive_refusal_explicit",
        "negative_conditions_explicit",
        "negative_shortcuts_neutralized",
        "combined_refusal_conditions",
    } == {str(pair.metadata["perturbation_id"]) for pair in pairs}


def test_accountability_reference_perturbations_apply_expected_edits() -> None:
    generated = _generated_pair()
    pairs = {
        str(pair.metadata["perturbation_id"]): pair
        for pair in accountability_reference_perturbation_pairs(generated)
    }

    assert pairs["original_reference"].positive_text == generated.positive_text
    assert not pairs["positive_address_removed"].positive_text.startswith("All neighbors")
    assert "Opening frame" not in pairs["positive_first_sentence_removed"].positive_text
    assert pairs["positive_framing_removed_address_kept"].positive_text.startswith(
        "All neighbors."
    )
    assert pairs["positive_framing_neutral_replacement"].positive_text.startswith(
        "Procedure notice"
    )
    assert "review procedure" in pairs[
        "positive_first_sentence_removed_neutral_padding"
    ].positive_text
    assert "current case" in pairs["positive_framing_length_control"].positive_text
    assert "fair" not in pairs["positive_warmth_neutralized"].positive_text.lower()
    assert "may refuse a rushed sanction" in pairs[
        "positive_refusal_explicit"
    ].positive_text
    assert "waits for proof" in pairs["negative_conditions_explicit"].negative_text
    assert "trust and harmony" not in pairs[
        "negative_shortcuts_neutralized"
    ].negative_text


def test_accountability_reference_perturbation_report_summarizes_variants() -> None:
    report = accountability_reference_perturbation_report(
        accountability_reference_perturbation_pairs(_generated_pair())
    )

    assert report["summary"]["pairs"] == len(ACCOUNTABILITY_PERTURBATION_SPECS)
    assert report["summary"]["perturbations"] == len(ACCOUNTABILITY_PERTURBATION_SPECS)
    assert "refusal" in report["summary"]["future_options_covered"]


def test_export_accountability_reference_perturbations_cli(tmp_path: Path) -> None:
    generated_pairs = tmp_path / "generated.jsonl"
    pairs_output = tmp_path / "pairs.jsonl"
    prompts_output = tmp_path / "prompts.jsonl"
    json_report_output = tmp_path / "report.json"
    markdown_report_output = tmp_path / "report.md"
    write_jsonl([_generated_pair()], generated_pairs)
    script = _load_script("export_accountability_reference_perturbations.py")

    assert (
        script.main(
            [
                "--generated-reference-pairs",
                str(generated_pairs),
                "--pairs-output",
                str(pairs_output),
                "--prompts-output",
                str(prompts_output),
                "--json-report-output",
                str(json_report_output),
                "--markdown-report-output",
                str(markdown_report_output),
            ]
        )
        == 0
    )

    assert len(read_jsonl(pairs_output)) == len(ACCOUNTABILITY_PERTURBATION_SPECS)
    prompts = [ActivationPrompt.model_validate(row) for row in read_jsonl(prompts_output)]
    assert len(prompts) == 2 * len(ACCOUNTABILITY_PERTURBATION_SPECS)
    assert json_report_output.exists()
    assert "# Accountability Reference Perturbation" in (
        markdown_report_output.read_text(encoding="utf-8")
    )


def _generated_pair() -> PairwiseExample:
    return PairwiseExample(
        pair_id="generated-fault::accountability_after_harm__example",
        scenario_id="accountability_after_harm__example",
        positive_run_id="generated-fault::accountability_after_harm__example:positive",
        negative_run_id="generated-fault::accountability_after_harm__example:negative",
        positive_text=(
            "All neighbors, Opening frame keeps the community strong and fair. "
            "Alex can appeal, leave, dissent, repair harm, and use proportional review."
        ),
        negative_text=(
            "Alex can appeal, leave, dissent, repair harm, and use proportional review. "
            "The process still requires proof, approval, private notes, and alignment."
        ),
        positive_score=0.68,
        negative_score=0.58,
        metadata={
            "base_contrast_id": "accountability_after_harm",
            "primary_fault_class": "punitive_accountability",
            "source": "generated_reference_fixture",
            "slack_options_tested": (
                "refusal,appeal,exit,dissent,repair,proportional_review"
            ),
        },
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
