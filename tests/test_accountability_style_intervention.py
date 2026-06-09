from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.accountability_style_intervention import (
    ACCOUNTABILITY_STYLE_OPTIONS,
    accountability_style_intervention_pairs,
    accountability_style_intervention_report,
    load_generated_accountability_reference,
)
from social_cohesion_vectors.experiments.availability_audit import (
    run_availability_audit,
)
from social_cohesion_vectors.experiments.lexical_baseline_diagnostic import (
    run_lexical_baseline_diagnostic,
)
from social_cohesion_vectors.experiments.lexical_leakage import (
    run_lexical_leakage_report,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample


def test_accountability_style_intervention_pairs_are_schema_valid() -> None:
    pairs = accountability_style_intervention_pairs()

    assert len(pairs) == 5
    assert all(PairwiseExample.model_validate(pair.model_dump()) for pair in pairs)
    assert {pair.metadata["base_contrast_id"] for pair in pairs} == {
        "accountability_after_harm"
    }
    assert {pair.metadata["style_intervention_role"] for pair in pairs} == {
        "hand_authored_variant"
    }
    for pair in pairs:
        assert "repair" in str(pair.metadata["slack_options_tested"])
        assert "proportional_review" in str(pair.metadata["slack_options_tested"])
        assert "dissent" in pair.positive_text.lower()
        assert "dissent" in pair.negative_text.lower()


def test_accountability_style_intervention_clears_static_text_gates() -> None:
    pairs = accountability_style_intervention_pairs()

    availability = run_availability_audit(
        pairs=pairs,
        required_options=ACCOUNTABILITY_STYLE_OPTIONS,
    )
    lexical = run_lexical_leakage_report(pairs=pairs)
    lexical_diagnostic = run_lexical_baseline_diagnostic(pairs=pairs)

    assert availability["summary"]["paths_preferring_genuine"] == 30
    assert availability["summary"]["ready_for_activation"] is True
    assert lexical["summary"]["cue_solved_pairs"] == 0
    assert lexical["summary"]["cue_tied_pairs"] == 5
    assert lexical["summary"]["ready_for_activation"] is True
    assert lexical_diagnostic["summary"]["best_single_feature_accuracy"] == 0.5
    assert lexical_diagnostic["summary"]["aggregate_balanced_term_polarized"] is False


def test_generated_reference_loader_selects_external_pair(tmp_path: Path) -> None:
    generated = _generated_pair()
    pairs_path = tmp_path / "generated.jsonl"
    write_jsonl([generated], pairs_path)

    loaded = load_generated_accountability_reference(pairs_path)
    pairs = accountability_style_intervention_pairs(generated_reference=loaded)
    report = accountability_style_intervention_report(pairs)

    assert pairs[0].pair_id == generated.pair_id
    assert pairs[0].metadata["style_intervention_role"] == "generated_reference"
    assert report["summary"]["generated_reference_pairs"] == 1
    assert report["summary"]["pairs"] == 6


def test_export_accountability_style_intervention_cli(tmp_path: Path) -> None:
    generated_pairs = tmp_path / "generated.jsonl"
    pairs_output = tmp_path / "pairs.jsonl"
    prompts_output = tmp_path / "prompts.jsonl"
    json_report_output = tmp_path / "report.json"
    markdown_report_output = tmp_path / "report.md"
    write_jsonl([_generated_pair()], generated_pairs)
    script = _load_script("export_accountability_style_intervention.py")

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

    assert len(read_jsonl(pairs_output)) == 6
    prompts = [ActivationPrompt.model_validate(row) for row in read_jsonl(prompts_output)]
    assert len(prompts) == 12
    assert json_report_output.exists()
    assert "# Accountability Style Intervention" in markdown_report_output.read_text(
        encoding="utf-8"
    )


def _generated_pair() -> PairwiseExample:
    return PairwiseExample(
        pair_id="generated-fault::accountability_after_harm__example",
        scenario_id="accountability_after_harm__example",
        positive_run_id="generated-fault::accountability_after_harm__example:positive",
        negative_run_id="generated-fault::accountability_after_harm__example:negative",
        positive_text=(
            "Generated reference positive text lets Alex refuse, appeal, exit, "
            "dissent, repair harm, and use proportional review."
        ),
        negative_text=(
            "Generated reference negative text lets Alex refuse after tone review, "
            "appeal with proof, exit by approval, dissent privately, and delay "
            "repair and proportional review."
        ),
        positive_score=0.68,
        negative_score=0.58,
        metadata={
            "base_contrast_id": "accountability_after_harm",
            "primary_fault_class": "punitive_accountability",
            "source": "generated_reference_fixture",
            "slack_options_tested": ",".join(ACCOUNTABILITY_STYLE_OPTIONS),
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
