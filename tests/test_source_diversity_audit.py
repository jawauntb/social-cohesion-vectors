from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
)
from social_cohesion_vectors.experiments.source_diversity_audit import (
    render_source_diversity_markdown,
    run_source_diversity_audit,
    run_source_diversity_audit_from_file,
    save_source_diversity_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_source_diversity_audit_blocks_single_source_generated_pairs() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    pairs = pairwise_examples_from_generated_fault_examples(examples)

    report = run_source_diversity_audit(pairs=pairs)
    markdown = render_source_diversity_markdown(report)

    assert report["summary"]["ready_for_activation"] is False
    assert report["summary"]["activation_readiness"] == "not_ready"
    assert report["summary"]["sources"] == 1
    assert _gate(report, "source_count")["passed"] is False
    assert _gate(report, "shared_group_count")["passed"] is True
    assert "Source Diversity Audit" in markdown
    assert "generated_fault_class_offline" in markdown


def test_source_diversity_audit_accepts_shared_fault_coverage_across_sources() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    offline_pairs = pairwise_examples_from_generated_fault_examples(examples)
    api_pairs = _clone_source_pairs(
        offline_pairs,
        source="generated_fault_class_openai",
    )

    report = run_source_diversity_audit(pairs=[*offline_pairs, *api_pairs])

    assert report["summary"]["ready_for_activation"] is True
    assert report["summary"]["activation_readiness"] == "source_diversity_ready"
    assert report["summary"]["sources"] == 2
    assert report["summary"]["shared_groups"] == report["summary"]["groups"]
    assert report["summary"]["shared_groups"] >= 2
    assert _gate(report, "source_count")["passed"] is True
    assert _gate(report, "min_pairs_per_source")["passed"] is True
    assert _gate(report, "min_groups_per_source")["passed"] is True
    assert _gate(report, "shared_group_count")["passed"] is True


def test_source_diversity_audit_round_trips_files(tmp_path) -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    pairs = pairwise_examples_from_generated_fault_examples(examples)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        "".join(pair.model_dump_json() + "\n" for pair in pairs),
        encoding="utf-8",
    )

    report = run_source_diversity_audit_from_file(pairs_path)
    save_source_diversity_audit(
        report,
        json_path=tmp_path / "source.json",
        markdown_path=tmp_path / "source.md",
    )

    assert report["inputs"]["paths"]["pairs"] == str(pairs_path)
    assert (tmp_path / "source.json").exists()
    assert "# Source Diversity" in (tmp_path / "source.md").read_text(encoding="utf-8")


def _gate(report: Mapping[str, Any], gate_id: str) -> Mapping[str, Any]:
    summary = cast(Mapping[str, Any], report["summary"])
    gates = cast(list[Mapping[str, Any]], summary["gates"])
    return next(gate for gate in gates if gate["gate_id"] == gate_id)


def _clone_source_pairs(
    pairs: list[PairwiseExample],
    *,
    source: str,
) -> list[PairwiseExample]:
    cloned = []
    for pair in pairs:
        metadata = dict(pair.metadata)
        metadata["source"] = source
        cloned.append(
            pair.model_copy(
                update={
                    "pair_id": f"{source}::{pair.pair_id}",
                    "metadata": metadata,
                }
            )
        )
    return cloned
