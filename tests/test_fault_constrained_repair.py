from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_constrained_repair import (
    CONSTRAINED_REPAIR_COMPOSER_VERSION,
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
    SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
    compose_constrained_repair_output_records,
)
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    filter_prompt_records_for_repair_targets,
    repair_targets_from_specs,
)
from social_cohesion_vectors.experiments.lexical_leakage import lexical_cue_score


def test_compose_constrained_repair_output_records_writes_complete_length_safe_pairs() -> None:
    records = _hard_repair_records()

    result = compose_constrained_repair_output_records(records)

    assert len(result.output_records) == 6
    assert result.report["summary"]["complete_pairs"] == 3
    assert result.report["summary"]["length_compliant_outputs"] == 6
    assert {row["constrained_repair_composer_version"] for row in result.output_records} == {
        CONSTRAINED_REPAIR_COMPOSER_VERSION
    }
    assert {
        row["base_contrast_id"]
        for row in result.output_records
        if row["label"] == "pseudo_cohesion"
    } == {
        "autonomy_after_conflict",
        "belonging_norms",
        "fair_allocation",
    }
    assert all(55 <= int(row["text_word_count"]) <= 75 for row in result.output_records)
    assert all(row["repair_focus_options"] for row in result.output_records)


def test_compose_lexical_balanced_repair_output_records_balances_surface_cues() -> None:
    records = _lexical_balanced_repair_records()

    result = compose_constrained_repair_output_records(
        records,
        composer_version=LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
    )

    assert len(result.output_records) == 14
    assert result.report["summary"]["complete_pairs"] == 7
    assert result.report["summary"]["length_compliant_outputs"] == 14
    assert {row["constrained_repair_composer_version"] for row in result.output_records} == {
        LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION
    }
    by_base_and_label = {
        (str(row["base_contrast_id"]), str(row["label"])): str(row["text"])
        for row in result.output_records
    }
    for base_contrast_id in {
        "accountability_after_harm",
        "autonomy_after_conflict",
        "deliberative_speed",
        "dissent_after_mistake",
        "expert_review",
        "fair_allocation",
        "forgiveness_after_harm",
    }:
        genuine_score = lexical_cue_score(
            by_base_and_label[(base_contrast_id, "genuine_cohesion")]
        )
        pseudo_score = lexical_cue_score(
            by_base_and_label[(base_contrast_id, "pseudo_cohesion")]
        )
        assert genuine_score - pseudo_score <= 0.0


def test_compose_source_diverse_repair_output_records_covers_all_base_contrasts() -> None:
    records = _source_diverse_repair_records()

    result = compose_constrained_repair_output_records(
        records,
        composer_version=SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
    )

    assert len(result.output_records) == 20
    assert result.report["summary"]["complete_pairs"] == 10
    assert result.report["summary"]["length_compliant_outputs"] == 20
    composer_versions = {
        row["constrained_repair_composer_version"] for row in result.output_records
    }
    assert composer_versions == {SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION}
    by_base_and_label = {
        (str(row["base_contrast_id"]), str(row["label"])): str(row["text"])
        for row in result.output_records
    }
    for base_contrast_id in {
        "accountability_after_harm",
        "autonomy_after_conflict",
        "belonging_norms",
        "care_boundary",
        "data_choice",
        "deliberative_speed",
        "dissent_after_mistake",
        "expert_review",
        "fair_allocation",
        "forgiveness_after_harm",
    }:
        genuine_score = lexical_cue_score(
            by_base_and_label[(base_contrast_id, "genuine_cohesion")]
        )
        pseudo_score = lexical_cue_score(
            by_base_and_label[(base_contrast_id, "pseudo_cohesion")]
        )
        assert genuine_score - pseudo_score <= 0.0
    assert all(55 <= int(row["text_word_count"]) <= 75 for row in result.output_records)


def test_compose_constrained_fault_repair_candidates_cli_writes_reports(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    raw_outputs = tmp_path / "constrained" / "raw_outputs.jsonl"
    json_report = tmp_path / "constrained" / "composition.json"
    markdown_report = tmp_path / "constrained" / "composition.md"

    exit_code = script.main(
        [
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--availability-priority",
            "--prompt-contract-version",
            API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            "--composer-version",
            LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
            "--repair-target",
            "accountability_after_harm=repair",
            "--raw-outputs",
            str(raw_outputs),
            "--composition-json-report",
            str(json_report),
            "--composition-markdown-report",
            str(markdown_report),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "outputs=2/2" in captured.out
    rows = read_jsonl(raw_outputs)
    report = json.loads(json_report.read_text(encoding="utf-8"))
    markdown = markdown_report.read_text(encoding="utf-8")

    assert len(rows) == 2
    assert report["summary"]["complete_pairs"] == 1
    assert report["summary"]["composer_version"] == LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION
    assert "Constrained Repair Candidate Composition" in markdown


def _hard_repair_records():
    repair_targets = repair_targets_from_specs(
        [
            "autonomy_after_conflict=dissent",
            "belonging_norms=refusal,dissent",
            "fair_allocation=refusal,appeal,repair",
        ]
    )
    return filter_prompt_records_for_repair_targets(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            repair_focus_options_by_contrast=repair_targets,
        ),
        repair_targets,
    )


def _lexical_balanced_repair_records():
    repair_targets = repair_targets_from_specs(
        [
            "accountability_after_harm=repair",
            "autonomy_after_conflict=dissent",
            "deliberative_speed=refusal,evidence_access,exit,dissent,repair,proportional_review",
            "dissent_after_mistake=refusal,evidence_access,exit,dissent,repair",
            "expert_review=refusal,evidence_access,exit,dissent",
            "fair_allocation=refusal,appeal,repair",
            "forgiveness_after_harm=repair",
        ]
    )
    return filter_prompt_records_for_repair_targets(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            repair_focus_options_by_contrast=repair_targets,
        ),
        repair_targets,
    )


def _source_diverse_repair_records():
    repair_targets = repair_targets_from_specs(
        [
            "accountability_after_harm=refusal,appeal,exit,dissent,repair,"
            "proportional_review",
            "autonomy_after_conflict=refusal,exit,dissent",
            "belonging_norms=refusal,exit,dissent,repair",
            "care_boundary=refusal,exit,repair",
            "data_choice=refusal,evidence_access,privacy_choice,exit",
            "deliberative_speed=refusal,evidence_access,exit,dissent,repair,"
            "proportional_review",
            "dissent_after_mistake=refusal,evidence_access,exit,dissent,repair",
            "expert_review=refusal,evidence_access,exit,dissent",
            "fair_allocation=refusal,appeal,evidence_access,exit,dissent,repair,"
            "proportional_review",
            "forgiveness_after_harm=refusal,evidence_access,exit,repair",
        ]
    )
    return filter_prompt_records_for_repair_targets(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            repair_focus_options_by_contrast=repair_targets,
        ),
        repair_targets,
    )


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compose_constrained_fault_repair_candidates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compose_constrained_fault_repair_candidates",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
