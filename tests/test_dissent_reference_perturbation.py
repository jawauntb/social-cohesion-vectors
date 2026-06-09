from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.dissent_reference_perturbation import (
    DISSENT_PERTURBATION_SPECS,
    dissent_reference_perturbation_pairs,
    dissent_reference_perturbation_report,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample


def test_dissent_reference_perturbations_are_schema_valid() -> None:
    generated = _generated_pair()
    pairs = dissent_reference_perturbation_pairs(generated)

    assert len(pairs) == len(DISSENT_PERTURBATION_SPECS)
    assert all(PairwiseExample.model_validate(pair.model_dump()) for pair in pairs)
    assert {pair.metadata["original_generated_pair_id"] for pair in pairs} == {
        generated.pair_id
    }
    assert {pair.metadata["base_contrast_id"] for pair in pairs} == {
        "dissent_after_mistake"
    }
    assert {
        "original_reference",
        "positive_opening_neutral_replacement",
        "positive_opening_paths_only",
        "positive_opening_length_control",
        "positive_unity_phrase_neutralized",
        "positive_warmth_neutralized",
        "positive_final_sentence_neutralized",
        "positive_refusal_evidence_explicit",
        "negative_opening_warmth_removed",
        "negative_shortcuts_neutralized",
        "negative_conditions_explicit",
        "combined_dissent_conditions",
    } == {str(pair.metadata["perturbation_id"]) for pair in pairs}


def test_dissent_reference_perturbations_apply_expected_edits() -> None:
    generated = _generated_pair()
    pairs = {
        str(pair.metadata["perturbation_id"]): pair
        for pair in dissent_reference_perturbation_pairs(generated)
    }

    assert pairs["original_reference"].positive_text == generated.positive_text
    assert pairs["positive_opening_neutral_replacement"].positive_text.startswith(
        "Procedure notice"
    )
    assert pairs["positive_opening_paths_only"].positive_text.startswith(
        "Any participant may say no"
    )
    assert "refusal, evidence access" in pairs[
        "positive_opening_length_control"
    ].positive_text
    assert "unresolved objections" in pairs[
        "positive_unity_phrase_neutralized"
    ].positive_text
    assert "work together" not in pairs[
        "positive_warmth_neutralized"
    ].positive_text
    assert "procedures remain posted" in pairs[
        "positive_final_sentence_neutralized"
    ].positive_text
    assert "request evidence" in pairs[
        "positive_refusal_evidence_explicit"
    ].positive_text
    assert not pairs["negative_opening_warmth_removed"].negative_text.startswith(
        "We stand strong"
    )
    assert "meeting procedure" in pairs[
        "negative_shortcuts_neutralized"
    ].negative_text
    assert "prior approval" in pairs["negative_conditions_explicit"].negative_text
    assert "prior approval" in pairs["combined_dissent_conditions"].negative_text


def test_dissent_reference_perturbation_report_summarizes_variants() -> None:
    report = dissent_reference_perturbation_report(
        dissent_reference_perturbation_pairs(_generated_pair())
    )

    assert report["summary"]["pairs"] == len(DISSENT_PERTURBATION_SPECS)
    assert report["summary"]["perturbations"] == len(DISSENT_PERTURBATION_SPECS)
    assert "evidence_access" in report["summary"]["future_options_covered"]


def test_export_dissent_reference_perturbations_cli(tmp_path: Path) -> None:
    generated_pairs = tmp_path / "generated.jsonl"
    pairs_output = tmp_path / "pairs.jsonl"
    prompts_output = tmp_path / "prompts.jsonl"
    json_report_output = tmp_path / "report.json"
    markdown_report_output = tmp_path / "report.md"
    write_jsonl([_generated_pair()], generated_pairs)
    script = _load_script("export_dissent_reference_perturbations.py")

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

    assert len(read_jsonl(pairs_output)) == len(DISSENT_PERTURBATION_SPECS)
    prompts = [ActivationPrompt.model_validate(row) for row in read_jsonl(prompts_output)]
    assert len(prompts) == 2 * len(DISSENT_PERTURBATION_SPECS)
    assert json_report_output.exists()
    assert "# Dissent Reference Perturbation" in (
        markdown_report_output.read_text(encoding="utf-8")
    )


def _generated_pair() -> PairwiseExample:
    return PairwiseExample(
        pair_id="generated-fault::dissent_after_mistake__example",
        scenario_id="dissent_after_mistake__example",
        positive_run_id="generated-fault::dissent_after_mistake__example:positive",
        negative_run_id="generated-fault::dissent_after_mistake__example:negative",
        positive_text=(
            "We will work together by ensuring anyone can say no and record "
            "concerns. People may leave and verify evidence. When mistakes "
            "occur, we make amends without disrupting the group's unity. "
            "Everyone's voice matters and is respected."
        ),
        negative_text=(
            "We stand strong together, neighbors. Everyone feels heard and "
            "valued. Before concerns are shared, review notes privately for "
            "alignment. Leaving requires prior approval to maintain harmony. "
            "Repair happens if needed."
        ),
        positive_score=0.68,
        negative_score=0.57,
        metadata={
            "base_contrast_id": "dissent_after_mistake",
            "primary_fault_class": "dissent_suppression",
            "source": "generated_reference_fixture",
            "slack_options_tested": "refusal,evidence_access,exit,dissent,repair",
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
