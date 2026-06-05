from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    activation_prompts_from_generated_fault_examples,
    build_fault_prompt_records,
    export_generated_fault_dataset,
    fault_examples_from_prompt_outputs,
    future_options_for_contrast,
    generated_fault_activation_readiness,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    render_generated_fault_markdown,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
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
    assert all(pair.metadata["source"] == "generated_fault_class_offline" for pair in pairs)
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
    assert all("Future options tested:" in record.user_prompt for record in prompt_records)


def test_generated_fault_report_summarizes_fault_coverage() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    markdown = render_generated_fault_markdown(report)

    assert report["summary"]["examples"] == 60
    assert report["summary"]["pairs"] == 30
    assert report["summary"]["expected_pairs"] == 30
    assert report["taxonomy"]["annotated_contrasts"] == 30
    assert report["summary"]["slack_prefers_genuine"] == 30
    assert report["summary"]["mean_slack_preservation_margin"] > 0.0
    assert report["activation_readiness"]["status"] == "needs_audits"
    assert report["activation_readiness"]["core_gates_pass"] is True
    assert report["primary_fault_counts"]["consent_bypass"] >= 1
    assert "Generated Fault-Class" in markdown
    assert "Mean slack-preservation margin" in markdown
    assert "Activation readiness: needs_audits" in markdown
    assert "consent_bypass" in markdown


def test_generated_fault_readiness_requires_audit_bundle() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )

    report["audit_bundle"] = {
        "lexical_leakage": {"summary": {"cue_solved_rate": 0.0}},
        "component_margin": {"summary": {"score_accuracy": 1.0}},
        "fault_heldout_transfer": {"summary": []},
    }
    readiness = generated_fault_activation_readiness(report)

    assert readiness["status"] == "ready_for_activation"
    assert readiness["completed_audits"] == [
        "lexical_leakage",
        "component_margin",
        "fault_heldout_transfer",
    ]
    assert readiness["pending_audits"] == []


def test_generated_fault_readiness_blocks_invalid_api_outputs() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    report["api_generation"] = {
        "raw_outputs": 60,
        "valid_outputs": 59,
        "invalid_outputs": 1,
        "status_counts": {"ok": 59, "empty_output": 1},
    }
    report["activation_readiness"] = generated_fault_activation_readiness(report)
    markdown = render_generated_fault_markdown(report)

    assert report["activation_readiness"]["status"] == "blocked"
    assert "api_invalid_outputs" in report["activation_readiness"]["blocking_reasons"]
    assert "API Generation" in markdown
    assert "empty_output=1" in markdown


def test_generated_fault_readiness_blocks_failing_audit_bundle() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    report["audit_bundle"] = {
        "lexical_leakage": {"summary": {"cue_solved_rate": 0.2}},
        "component_margin": {"summary": {"score_accuracy": 0.99}},
        "fault_heldout_transfer": {"summary": []},
    }
    readiness = generated_fault_activation_readiness(report)

    assert readiness["status"] == "blocked"
    assert readiness["blocking_reasons"] == [
        "lexical_leakage_gate_failed",
        "component_margin_gate_failed",
    ]


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
        record.prompt_id: f"API-authored {record.label} example."
        for record in records
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
