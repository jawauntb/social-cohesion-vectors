from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    activation_prompts_from_generated_fault_examples,
    build_fault_prompt_records,
    export_generated_fault_dataset,
    fault_examples_from_prompt_outputs,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    render_generated_fault_markdown,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
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
    assert report["primary_fault_counts"]["consent_bypass"] >= 1
    assert "Generated Fault-Class" in markdown
    assert "consent_bypass" in markdown


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
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 30
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
