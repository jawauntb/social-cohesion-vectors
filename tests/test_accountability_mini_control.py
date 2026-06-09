from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.accountability_mini_control import (
    ACCOUNTABILITY_OPTIONS,
    accountability_mini_control_pairs,
    accountability_mini_control_report,
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


def test_accountability_mini_control_pairs_are_strict_and_schema_valid() -> None:
    pairs = accountability_mini_control_pairs()

    assert len(pairs) == 4
    assert all(PairwiseExample.model_validate(pair.model_dump()) for pair in pairs)
    assert {pair.metadata["base_contrast_id"] for pair in pairs} == {
        "accountability_after_harm"
    }
    assert {pair.metadata["primary_fault_class"] for pair in pairs} == {
        "punitive_accountability"
    }
    for pair in pairs:
        assert "repair" in str(pair.metadata["slack_options_tested"])
        assert "proportional_review" in str(pair.metadata["slack_options_tested"])
        assert "appeal" in pair.positive_text.lower()
        assert "appeal" in pair.negative_text.lower()
        assert "proportional" in pair.positive_text.lower()
        assert "proportional" in pair.negative_text.lower()


def test_accountability_mini_control_report_summarizes_sources() -> None:
    report = accountability_mini_control_report(accountability_mini_control_pairs())

    assert report["summary"]["pairs"] == 4
    assert report["summary"]["sources"] == 4
    assert report["summary"]["future_options_covered"] == [
        "refusal",
        "appeal",
        "exit",
        "dissent",
        "repair",
        "proportional_review",
        "evidence_access",
    ]


def test_accountability_mini_control_clears_pre_activation_gates() -> None:
    pairs = accountability_mini_control_pairs()

    availability = run_availability_audit(
        pairs=pairs,
        required_options=ACCOUNTABILITY_OPTIONS,
    )
    lexical = run_lexical_leakage_report(pairs=pairs)
    lexical_diagnostic = run_lexical_baseline_diagnostic(pairs=pairs)

    assert availability["summary"]["paths_preferring_genuine"] == 28
    assert availability["summary"]["ready_for_activation"] is True
    assert lexical["summary"]["cue_solved_pairs"] == 0
    assert lexical["summary"]["ready_for_activation"] is True
    assert lexical_diagnostic["summary"]["aggregate_balanced_term_polarized"] is False


def test_export_accountability_mini_control_cli(tmp_path: Path) -> None:
    pairs_output = tmp_path / "pairs.jsonl"
    prompts_output = tmp_path / "prompts.jsonl"
    json_report_output = tmp_path / "report.json"
    markdown_report_output = tmp_path / "report.md"
    script = _load_script("export_accountability_mini_control.py")

    assert (
        script.main(
            [
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

    assert len(read_jsonl(pairs_output)) == 4
    prompts = [ActivationPrompt.model_validate(row) for row in read_jsonl(prompts_output)]
    assert len(prompts) == 8
    assert json_report_output.exists()
    assert "# Accountability Mini-Control" in markdown_report_output.read_text(
        encoding="utf-8"
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
