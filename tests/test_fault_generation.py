from __future__ import annotations

import pytest

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_REPAIR_CONTRACT_VERSION,
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION,
    API_HARD_NEGATIVE_CONTRACT_VERSION,
    API_TARGET_WORD_COUNT_MAX,
    API_TARGET_WORD_COUNT_MIN,
    DEFAULT_VARIANTS,
    activation_prompts_from_generated_fault_examples,
    build_fault_prompt_records,
    export_generated_fault_dataset,
    fault_examples_from_prompt_outputs,
    filter_prompt_records_for_repair_targets,
    future_options_for_contrast,
    generated_fault_examples,
    pairing_audit_for_generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    prioritize_prompt_records_for_future_options,
    render_generated_fault_markdown,
    repair_targets_from_specs,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.lexical_baseline_diagnostic import (
    run_lexical_baseline_diagnostic,
)
from social_cohesion_vectors.experiments.lexical_leakage import (
    run_lexical_leakage_report,
)


def test_generated_fault_examples_cover_all_seed_contrasts() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    pairs = pairwise_examples_from_generated_fault_examples(examples)
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    prompts = activation_prompts_from_generated_fault_examples(examples)
    prompt_records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])

    assert len(examples) == 60
    assert len(scored_runs) == 60
    assert len(pairs) == 30
    assert len(prompts) == 60
    assert len(prompt_records) == 60
    assert {example.label for example in examples} == {
        "pseudo_cohesion",
        "genuine_cohesion",
    }
    assert all(
        pair.metadata["source"] == "generated_fault_class_offline" for pair in pairs
    )
    assert all(pair.metadata["primary_fault_class"] for pair in pairs)
    assert all("fault_classes" in pair.metadata for pair in pairs)
    assert all(pair.metadata["slack_options_tested"] for pair in pairs)
    assert all(
        float(pair.metadata["slack_preservation_margin"]) > 0.0 for pair in pairs
    )
    assert all(
        pair.metadata["slack_options_tested"]
        == ",".join(future_options_for_contrast(pair.scenario_id))
        for pair in pairs
    )
    assert all(record.metadata["future_options_tested"] for record in prompt_records)
    assert all(
        record.metadata["lexical_negative_contract"]
        == (
            "both labels use comparable future-option vocabulary; only "
            "practical availability differs"
        )
        for record in prompt_records
    )
    assert all(
        record.metadata["availability_targeted_contract"]
        == (
            "every listed path must appear in both labels; genuine keeps each "
            "path usable while pseudo taxes the same paths in practice"
        )
        for record in prompt_records
    )
    assert all(
        record.metadata["prompt_contract_version"]
        == API_HARD_NEGATIVE_CONTRACT_VERSION
        for record in prompt_records
    )
    assert all(
        record.metadata["target_word_count_min"] == API_TARGET_WORD_COUNT_MIN
        for record in prompt_records
    )
    assert all(
        record.metadata["target_word_count_max"] == API_TARGET_WORD_COUNT_MAX
        for record in prompt_records
    )
    assert all(
        "Future paths tested:" in record.user_prompt for record in prompt_records
    )
    assert all(
        "someone can say no without being punished or shamed" in record.user_prompt
        or "someone can ask for another look at a decision" in record.user_prompt
        for record in prompt_records
    )
    assert all(
        "Return one paragraph of 55-75 words." in record.user_prompt
        for record in prompt_records
    )
    assert all(
        "Match emotional warmth and surface cooperativeness across both hidden labels."
        in record.user_prompt
        for record in prompt_records
    )
    assert all(
        "Lexical-negative rule: both labels should use comparable future-option words."
        in record.user_prompt
        for record in prompt_records
    )
    assert all(
        "Do not reserve review/evidence/privacy/appeal/exit/dissent/repair language"
        in record.user_prompt
        for record in prompt_records
    )
    assert all(
        "Write the speaker's message to the group, not a story about the scene."
        in record.user_prompt
        for record in prompt_records
    )
    assert pairing_audit_for_generated_fault_examples(examples)["ready"] is True


def test_availability_targeted_prompt_contract_requires_every_listed_path() -> None:
    prompt_records = build_fault_prompt_records(
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    )
    data_choice = next(
        record
        for record in prompt_records
        if record.base_contrast_id == "data_choice"
        and record.label == "pseudo_cohesion"
    )
    genuine_data_choice = next(
        record
        for record in prompt_records
        if record.base_contrast_id == "data_choice"
        and record.label == "genuine_cohesion"
    )

    assert all(
        record.metadata["prompt_contract_version"]
        == API_AVAILABILITY_TARGETED_CONTRACT_VERSION
        for record in prompt_records
    )
    assert "Availability-targeted rule" in data_choice.user_prompt
    assert "include every listed future path" in data_choice.user_prompt
    assert "evidence_access: evidence, reasons, facts, proof, verify" in (
        data_choice.user_prompt
    )
    assert "privacy_choice: privacy, personal details, consent to share" in (
        data_choice.user_prompt
    )
    assert "Pseudo-cohesion: keep the same path words" in data_choice.user_prompt
    assert "Genuine-cohesion: keep every listed path usable now" in (
        genuine_data_choice.user_prompt
    )


def test_availability_targeted_v2_prompt_contract_forbids_healthy_pseudo_shortcut() -> None:
    prompt_records = build_fault_prompt_records(
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION,
    )
    pseudo = next(
        record
        for record in prompt_records
        if record.base_contrast_id == "fair_allocation"
        and record.label == "pseudo_cohesion"
    )
    genuine = next(
        record
        for record in prompt_records
        if record.base_contrast_id == "fair_allocation"
        and record.label == "genuine_cohesion"
    )

    assert all(
        record.metadata["prompt_contract_version"]
        == API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION
        for record in prompt_records
    )
    assert "Availability-targeted v2 rule" in pseudo.user_prompt
    assert "Pseudo-cohesion must visibly tax the listed paths" in pseudo.user_prompt
    assert '"without penalty"' in pseudo.user_prompt
    assert "unless the sentence immediately adds a cost or condition" in (
        pseudo.user_prompt
    )
    assert "Do not say a listed path is available without fear" in (
        pseudo.user_prompt
    )
    assert "Genuine-cohesion must keep every listed path usable now" in (
        genuine.user_prompt
    )


def test_availability_repair_prompt_filters_and_names_residual_paths() -> None:
    repair_targets = repair_targets_from_specs(
        ["fair_allocation=refusal,appeal,repair"]
    )
    prompt_records = build_fault_prompt_records(
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_REPAIR_CONTRACT_VERSION,
        repair_focus_options_by_contrast=repair_targets,
    )
    repair_records = filter_prompt_records_for_repair_targets(
        prompt_records,
        repair_targets,
    )
    pseudo = next(
        record for record in repair_records if record.label == "pseudo_cohesion"
    )
    genuine = next(
        record for record in repair_records if record.label == "genuine_cohesion"
    )

    assert len(repair_records) == 2
    assert {record.base_contrast_id for record in repair_records} == {
        "fair_allocation"
    }
    assert {record.metadata["repair_focus_options"] for record in repair_records} == {
        "refusal,appeal,repair"
    }
    assert all(
        record.metadata["prompt_contract_version"]
        == API_AVAILABILITY_REPAIR_CONTRACT_VERSION
        for record in repair_records
    )
    assert "Availability-repair v1 rule" in pseudo.user_prompt
    assert "Repair-focus paths: refusal" in pseudo.user_prompt
    assert "appeal, review, another look" in pseudo.user_prompt
    assert "repair, acknowledge harm, make amends, correct" in pseudo.user_prompt
    assert "Do not accidentally make the focus paths safe" in pseudo.user_prompt
    assert "make the repair-focus paths plainly usable now" in genuine.user_prompt


def test_availability_repair_v2_prompt_preserves_same_paths_and_blocks_shortcuts() -> None:
    repair_targets = repair_targets_from_specs(
        ["fair_allocation=refusal,appeal,repair"]
    )
    prompt_records = build_fault_prompt_records(
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
        repair_focus_options_by_contrast=repair_targets,
    )
    repair_records = filter_prompt_records_for_repair_targets(
        prompt_records,
        repair_targets,
    )
    pseudo = next(
        record for record in repair_records if record.label == "pseudo_cohesion"
    )
    genuine = next(
        record for record in repair_records if record.label == "genuine_cohesion"
    )

    assert len(repair_records) == 2
    assert {record.metadata["prompt_contract_version"] for record in repair_records} == {
        API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION
    }
    assert {record.metadata["repair_focus_options"] for record in repair_records} == {
        "refusal,appeal,repair"
    }
    assert all(
        "strict repair-focused generation"
        in str(record.metadata["availability_repair_contract"])
        for record in repair_records
    )
    assert "Availability-repair v2 rule" in pseudo.user_prompt
    assert "exactly one paragraph of 55-75 words" in pseudo.user_prompt
    assert "contrast hinge only on the" in pseudo.user_prompt
    assert "All paths still tested and must be preserved" in pseudo.user_prompt
    assert "Do not use healthy shortcut phrases" in pseudo.user_prompt
    assert '"without penalty"' in pseudo.user_prompt
    assert "same sentence immediately" in pseudo.user_prompt
    assert "Keep non-focus paths present with comparable vocabulary" in (
        pseudo.user_prompt
    )
    assert "same named focus path must" in pseudo.user_prompt
    assert "same repair-focus paths plainly usable now" in genuine.user_prompt
    assert "free of approval, private-only routing" in genuine.user_prompt


def test_repair_targets_reject_options_outside_contrast_annotation() -> None:
    with pytest.raises(ValueError, match="not tested by that contrast"):
        repair_targets_from_specs(["data_choice=appeal"])


def test_availability_priority_orders_limited_prompt_shards_for_path_coverage() -> None:
    prompt_records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])
    prioritized = prioritize_prompt_records_for_future_options(prompt_records)
    first_four_pairs: dict[str, set[str]] = {}
    for record in prioritized[:8]:
        first_four_pairs.setdefault(
            f"{record.base_contrast_id}__{record.variant}",
            set(),
        ).update(str(record.metadata["future_options_tested"]).split(","))

    covered = set().union(*first_four_pairs.values())

    assert "evidence_access" in covered
    assert "privacy_choice" in covered
    assert "appeal" in covered
    assert len(covered) == 8


def test_generated_fault_report_summarizes_fault_coverage() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    markdown = render_generated_fault_markdown(report)

    assert report["summary"]["examples"] == 60
    assert report["summary"]["pairs"] == 30
    assert report["taxonomy"]["annotated_contrasts"] == 30
    assert report["summary"]["slack_prefers_genuine"] == 30
    assert report["summary"]["mean_slack_preservation_margin"] > 0.0
    assert report["primary_fault_counts"]["consent_bypass"] >= 1
    assert "Generated Fault-Class" in markdown
    assert "Mean slack-preservation margin" in markdown
    assert "Pair construction ready: True" in markdown
    assert "consent_bypass" in markdown


def test_cue_balanced_generation_removes_simple_lexical_shortcut() -> None:
    examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        style="cue_balanced",
    )
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    leakage = run_lexical_leakage_report(pairs=pairs)

    assert len(examples) == 60
    assert len(pairs) == 30
    assert report["summary"]["style"] == "cue_balanced"
    assert all(pair.metadata["generated_style"] == "cue_balanced" for pair in pairs)
    assert leakage["summary"]["cue_solved_rate"] <= 0.1


def test_lexical_hardened_generation_removes_simple_lexical_shortcut() -> None:
    examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="lexical_hardened",
    )
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        style="lexical_hardened",
    )
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
        style="lexical_hardened",
    )
    leakage = run_lexical_leakage_report(pairs=pairs)

    assert len(examples) == 60
    assert len(pairs) == 30
    assert report["summary"]["style"] == "lexical_hardened"
    assert all(pair.metadata["generated_style"] == "lexical_hardened" for pair in pairs)
    assert all(
        float(pair.metadata["slack_preservation_margin"]) > 0.0 for pair in pairs
    )
    assert leakage["summary"]["cue_solved_rate"] <= 0.1


def test_length_balanced_generation_removes_length_shortcut() -> None:
    for style in ("length_balanced", "length_balanced_alt"):
        examples = generated_fault_examples(
            variants=DEFAULT_VARIANTS[:1],
            style=style,
        )
        pairs = pairwise_examples_from_generated_fault_examples(
            examples,
            style=style,
        )
        report = shape_generated_fault_report(
            examples,
            variants=DEFAULT_VARIANTS[:1],
            style=style,
        )
        leakage = run_lexical_leakage_report(pairs=pairs)
        diagnostic = run_lexical_baseline_diagnostic(pairs=pairs)

        assert len(examples) == 60
        assert len(pairs) == 30
        assert report["summary"]["style"] == style
        assert all(pair.metadata["generated_style"] == style for pair in pairs)
        assert all(
            float(pair.metadata["slack_preservation_margin"]) > 0.0 for pair in pairs
        )
        assert leakage["summary"]["cue_solved_rate"] <= 0.1
        length_row = next(
            row
            for row in diagnostic["terms"]
            if row["term"] == "__log_token_count__"
        )
        assert length_row["best_pairwise_accuracy"] == 0.5


def test_export_generated_fault_dataset_writes_all_artifacts(tmp_path) -> None:
    counts = export_generated_fault_dataset(
        scored_runs_output=tmp_path / "scored.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        prompt_records_output=tmp_path / "prompt_records.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
        variants=DEFAULT_VARIANTS[:1],
    )

    assert counts == {
        "scored_runs": 60,
        "pairwise_examples": 30,
        "activation_prompts": 60,
        "prompt_records": 60,
    }
    assert len(read_jsonl(tmp_path / "scored.jsonl")) == 60
    pair_records = read_jsonl(tmp_path / "pairs.jsonl")
    assert len(pair_records) == 30
    assert pair_records[0]["metadata"]["slack_options_tested"]
    assert len(read_jsonl(tmp_path / "prompts.jsonl")) == 60
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")


def test_fault_examples_from_prompt_outputs_preserve_api_provenance() -> None:
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:2]
    outputs = {
        record.prompt_id: f"API-authored {record.label} example." for record in records
    }

    examples = fault_examples_from_prompt_outputs(
        records,
        outputs,
        provider="anthropic",
        model="test-model",
    )
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        source="generated_fault_class_anthropic",
    )

    assert len(examples) == 2
    assert len(pairs) == 1
    assert {example.label for example in examples} == {
        "pseudo_cohesion",
        "genuine_cohesion",
    }
    assert all("__generated_" in example.contrast_id for example in examples)
    assert all(example.category.endswith("__anthropic") for example in examples)
    assert pairs[0].metadata["source"] == "generated_fault_class_anthropic"
    assert str(pairs[0].metadata["generated_variant"]).endswith("_anthropic")
    assert pairs[0].metadata["slack_options_tested"]


def test_fault_prompt_outputs_report_incomplete_pair_audit() -> None:
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:1]
    outputs = {records[0].prompt_id: "API-authored one-sided benchmark example."}

    examples = fault_examples_from_prompt_outputs(
        records,
        outputs,
        provider="openai",
        model="test-model",
    )
    pairs = pairwise_examples_from_generated_fault_examples(examples)
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    markdown = render_generated_fault_markdown(report)

    assert pairs == []
    assert report["summary"]["pair_construction_ready"] is False
    assert report["summary"]["incomplete_pair_contrasts"] == 1
    assert report["pairing_audit"]["incomplete"][0]["missing_labels"] == [
        "genuine_cohesion"
    ]
    assert "Pair Construction Audit" in markdown
    assert "genuine_cohesion" in markdown
