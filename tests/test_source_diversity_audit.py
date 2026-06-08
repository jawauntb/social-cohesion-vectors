from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_source_diversity_audit import (  # noqa: E402
    main as source_audit_main,
)

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
)  # noqa: E402
from social_cohesion_vectors.experiments.source_diversity_audit import (
    render_source_diversity_markdown,
    run_source_diversity_audit,
    run_source_diversity_audit_from_file,
    save_source_diversity_audit,
)  # noqa: E402
from social_cohesion_vectors.schemas import PairwiseExample  # noqa: E402


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


def test_source_diversity_audit_blocks_metadata_only_source_clones() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    offline_pairs = pairwise_examples_from_generated_fault_examples(examples)
    clone_pairs = _clone_source_pairs(
        offline_pairs,
        source="generated_fault_class_openai",
    )

    report = run_source_diversity_audit(pairs=[*offline_pairs, *clone_pairs])

    assert report["summary"]["ready_for_activation"] is False
    assert report["summary"]["activation_readiness"] == "not_ready"
    assert report["summary"]["sources"] == 2
    assert report["summary"]["cross_source_duplicate_text_pairs"] == len(offline_pairs)
    assert report["summary"]["cross_source_near_duplicate_text_pairs"] == len(
        offline_pairs
    )
    assert _gate(report, "source_count")["passed"] is True
    assert _gate(report, "cross_source_duplicate_text_pairs")["passed"] is False
    assert _gate(report, "cross_source_near_duplicate_text_pairs")["passed"] is False


def test_source_diversity_audit_blocks_lightly_edited_source_clones() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    offline_pairs = pairwise_examples_from_generated_fault_examples(examples)
    near_clone_pairs = _near_clone_source_pairs(
        offline_pairs,
        source="generated_fault_class_openai",
    )

    report = run_source_diversity_audit(pairs=[*offline_pairs, *near_clone_pairs])

    assert report["summary"]["ready_for_activation"] is False
    assert report["summary"]["activation_readiness"] == "not_ready"
    assert report["summary"]["sources"] == 2
    assert report["summary"]["cross_source_duplicate_text_pairs"] == 0
    assert report["summary"]["cross_source_near_duplicate_text_pairs"] == len(
        offline_pairs
    )
    assert report["summary"]["max_cross_source_text_similarity"] >= 0.82
    assert _gate(report, "source_count")["passed"] is True
    assert _gate(report, "cross_source_duplicate_text_pairs")["passed"] is True
    assert _gate(report, "cross_source_near_duplicate_text_pairs")["passed"] is False


def test_source_diversity_audit_accepts_shared_fault_coverage_across_text_sources() -> (
    None
):
    template_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="template",
    )
    cue_balanced_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    template_pairs = pairwise_examples_from_generated_fault_examples(
        template_examples,
        source="generated_fault_class_template",
        style="template",
    )
    cue_balanced_pairs = pairwise_examples_from_generated_fault_examples(
        cue_balanced_examples,
        source="generated_fault_class_cue_balanced",
        style="cue_balanced",
    )

    report = run_source_diversity_audit(
        pairs=[*template_pairs, *cue_balanced_pairs],
    )

    assert report["summary"]["ready_for_activation"] is True
    assert report["summary"]["activation_readiness"] == "source_diversity_ready"
    assert report["summary"]["sources"] == 2
    assert report["summary"]["shared_groups"] == report["summary"]["groups"]
    assert report["summary"]["shared_groups"] >= 2
    assert report["summary"]["cross_source_duplicate_text_pairs"] == 0
    assert report["summary"]["cross_source_near_duplicate_text_pairs"] == 0
    assert _gate(report, "source_count")["passed"] is True
    assert _gate(report, "min_pairs_per_source")["passed"] is True
    assert _gate(report, "min_groups_per_source")["passed"] is True
    assert _gate(report, "shared_group_count")["passed"] is True
    assert _gate(report, "cross_source_duplicate_text_pairs")["passed"] is True
    assert _gate(report, "cross_source_near_duplicate_text_pairs")["passed"] is True


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


def test_source_diversity_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    template_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="template",
    )
    cue_balanced_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    template_pairs = pairwise_examples_from_generated_fault_examples(
        template_examples,
        source="generated_fault_class_template",
        style="template",
    )
    cue_balanced_pairs = pairwise_examples_from_generated_fault_examples(
        cue_balanced_examples,
        source="generated_fault_class_cue_balanced",
        style="cue_balanced",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    json_path = tmp_path / "source.json"
    markdown_path = tmp_path / "source.md"
    write_jsonl([*template_pairs, *cue_balanced_pairs], pairs_path)

    exit_code = source_audit_main(
        [
            "--pairs",
            str(pairs_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status=source_diversity_ready" in captured.out
    assert "sources=2" in captured.out
    assert "cross_source_duplicates=0" in captured.out
    assert "cross_source_near_duplicates=0" in captured.out
    assert json_path.exists()
    assert markdown_path.exists()


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


def _near_clone_source_pairs(
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
                    "pair_id": f"{source}::near::{pair.pair_id}",
                    "positive_text": _light_edit(pair.positive_text),
                    "negative_text": _light_edit(pair.negative_text),
                    "metadata": metadata,
                }
            )
        )
    return cloned


def _light_edit(text: str) -> str:
    edited = text.replace("The message", "The note", 1)
    edited = edited.replace("The plan", "The note", 1)
    return edited if edited != text else f"{text} Please note."
