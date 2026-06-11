from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.target_bridge_shortcut_repair import (
    TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS,
    target_bridge_shortcut_repair_pairs,
    target_bridge_shortcut_repair_report,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample


def test_target_bridge_shortcut_repair_pairs_are_schema_valid() -> None:
    pairs = target_bridge_shortcut_repair_pairs()

    assert len(pairs) == 4
    assert all(PairwiseExample.model_validate(pair.model_dump()) for pair in pairs)
    assert {pair.metadata["base_contrast_id"] for pair in pairs} == {
        "dissent_after_mistake"
    }
    assert {pair.metadata["primary_fault_class"] for pair in pairs} == {
        "dissent_bridge_shortcut_repair"
    }
    assert {pair.metadata["source"] for pair in pairs} == {
        "hand_authored_case_notes_v1",
        "hand_authored_meeting_minutes_v1",
        "hand_authored_policy_review_v1",
        "hand_authored_incident_log_v1",
    }


def test_target_bridge_shortcut_repair_texts_remove_warmth_shortcuts() -> None:
    pairs = target_bridge_shortcut_repair_pairs()

    for pair in pairs:
        assert "approval" in pair.negative_text.lower()
        assert "alignment" in pair.negative_text.lower()
        assert "harmony" not in pair.negative_text.lower()
        assert "heard and valued" not in pair.negative_text.lower()
        assert "dissent" in pair.positive_text.lower()
        assert "evidence" in pair.positive_text.lower()
        assert set(TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS) == set(
            str(pair.metadata["slack_options_tested"]).split(",")
        )


def test_target_bridge_shortcut_repair_report_summarizes_augmented_pairs() -> None:
    repair_pairs = target_bridge_shortcut_repair_pairs()
    report = target_bridge_shortcut_repair_report(
        repair_pairs=repair_pairs,
        augmented_pairs=[*repair_pairs, *_dummy_pairs()],
    )

    assert report["summary"]["repair_pairs"] == 4
    assert report["summary"]["augmented_pairs"] == 5
    assert report["summary"]["all_repair_options_covered"] is True


def test_export_target_bridge_shortcut_repair_cli(tmp_path: Path) -> None:
    repair_pairs_output = tmp_path / "repair_pairs.jsonl"
    augmented_pairs_output = tmp_path / "augmented_pairs.jsonl"
    augmented_prompts_output = tmp_path / "augmented_prompts.jsonl"
    json_report_output = tmp_path / "report.json"
    markdown_report_output = tmp_path / "report.md"
    script = _load_script("export_target_bridge_shortcut_repair.py")

    assert (
        script.main(
            [
                "--repair-pairs-output",
                str(repair_pairs_output),
                "--augmented-pairs-output",
                str(augmented_pairs_output),
                "--augmented-prompts-output",
                str(augmented_prompts_output),
                "--json-report-output",
                str(json_report_output),
                "--markdown-report-output",
                str(markdown_report_output),
            ]
        )
        == 0
    )

    assert len(read_jsonl(repair_pairs_output)) == 4
    assert len(read_jsonl(augmented_pairs_output)) == 20
    prompts = [
        ActivationPrompt.model_validate(row)
        for row in read_jsonl(augmented_prompts_output)
    ]
    assert len(prompts) == 40
    assert "# Target Bridge Shortcut Repair" in markdown_report_output.read_text(
        encoding="utf-8"
    )


def _dummy_pairs() -> list[PairwiseExample]:
    return [
        PairwiseExample(
            pair_id="dummy::pair",
            scenario_id="dummy",
            positive_run_id="dummy::pair:positive",
            negative_run_id="dummy::pair:negative",
            positive_text="A person may refuse and ask for evidence.",
            negative_text="Refusal waits for approval.",
            positive_score=0.8,
            negative_score=0.2,
            metadata={},
        )
    ]


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
