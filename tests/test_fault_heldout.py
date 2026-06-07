from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    scored_runs_from_generated_fault_examples,
)
from social_cohesion_vectors.experiments.fault_heldout import (
    render_fault_heldout_markdown,
    run_fault_heldout_transfer,
    run_fault_heldout_transfer_from_files,
    save_fault_heldout_reports,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun


def test_fault_heldout_transfer_groups_by_primary_fault_class() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(examples)

    report = run_fault_heldout_transfer(scored_runs=scored_runs, pairs=pairs)
    markdown = render_fault_heldout_markdown(report)

    assert report["inputs"]["pairs"] == 30
    assert report["inputs"]["fault_classes"] >= 10
    assert report["inputs"]["missing_metadata_pairs"] == 0
    assert report["inputs"]["source_counts"]["generated_fault_class_offline"] == 30
    assert report["inputs"]["source_groups"] == 1
    assert report["readiness"]["status"] == "transfer_ready"
    assert report["readiness"]["ready"] is True
    assert report["source_transfer"]["readiness"]["ready"] is False
    assert _gate(report, "metadata_group_count")["passed"] is True
    assert _gate(report, "min_test_pairs_per_fold")["passed"] is True
    assert _source_gate(report, "metadata_group_count")["passed"] is False
    assert len(report["summary"]) == 3
    assert {row["baseline"] for row in report["summary"]} == {
        "strategy_prior",
        "metrics_only",
        "lexical_only",
    }
    assert all(row["split"] == "fault_class" for row in report["folds"])
    assert "Fault-Held-Out Transfer" in markdown
    assert "Readiness Gates" in markdown
    assert "Source Coverage" in markdown
    assert "Source-Held-Out Transfer" in markdown
    assert "consent_bypass" in markdown


def test_fault_heldout_transfer_holds_out_generated_api_sources() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    offline_runs = scored_runs_from_generated_fault_examples(examples)
    offline_pairs = pairwise_examples_from_generated_fault_examples(examples)
    api_runs, api_pairs = _clone_source_records(
        scored_runs=offline_runs,
        pairs=offline_pairs,
        source="generated_fault_class_anthropic",
    )

    report = run_fault_heldout_transfer(
        scored_runs=[*offline_runs, *api_runs],
        pairs=[*offline_pairs, *api_pairs],
    )
    markdown = render_fault_heldout_markdown(report)
    source_transfer = cast(Mapping[str, Any], report["source_transfer"])

    assert report["inputs"]["source_groups"] == 2
    assert report["inputs"]["source_counts"] == {
        "generated_fault_class_anthropic": 30,
        "generated_fault_class_offline": 30,
    }
    assert source_transfer["readiness"]["status"] == "transfer_ready"
    assert source_transfer["readiness"]["ready"] is True
    assert len(source_transfer["summary"]) == 3
    assert len(source_transfer["folds"]) == 6
    assert {fold["held_out"] for fold in source_transfer["folds"]} == {
        "generated_fault_class_anthropic",
        "generated_fault_class_offline",
    }
    assert "generated_fault_class_anthropic" in markdown
    assert "Ready for source-transfer claims: True" in markdown


def test_fault_heldout_transfer_blocks_missing_metadata() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(examples)
    first_pair = pairs[0]
    metadata = dict(first_pair.metadata)
    metadata.pop("primary_fault_class")
    incomplete_pairs = [
        first_pair.model_copy(update={"metadata": metadata}),
        *pairs[1:],
    ]

    report = run_fault_heldout_transfer(
        scored_runs=scored_runs,
        pairs=incomplete_pairs,
    )
    markdown = render_fault_heldout_markdown(report)

    assert report["inputs"]["missing_metadata_pairs"] == 1
    assert report["readiness"]["status"] == "not_ready_for_transfer_claims"
    assert report["readiness"]["ready"] is False
    assert _gate(report, "missing_metadata_pairs")["passed"] is False
    assert "Ready for downstream claims: False" in markdown


def test_fault_heldout_transfer_blocks_undercovered_folds() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(examples)

    report = run_fault_heldout_transfer(
        scored_runs=scored_runs,
        pairs=pairs,
        min_test_pairs_per_fold=2,
    )
    markdown = render_fault_heldout_markdown(report)

    assert report["readiness"]["status"] == "not_ready_for_transfer_claims"
    assert report["readiness"]["ready"] is False
    assert _gate(report, "min_test_pairs_per_fold")["passed"] is False
    assert report["readiness"]["failed_metadata_values"]
    assert "Not ready for downstream claims" in markdown


def test_fault_heldout_transfer_round_trips_files(tmp_path) -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(examples)
    scored_path = tmp_path / "scored.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    scored_path.write_text(
        "".join(run.model_dump_json() + "\n" for run in scored_runs),
        encoding="utf-8",
    )
    pairs_path.write_text(
        "".join(pair.model_dump_json() + "\n" for pair in pairs),
        encoding="utf-8",
    )

    report = run_fault_heldout_transfer_from_files(
        scored_runs_path=scored_path,
        pairs_path=pairs_path,
    )
    save_fault_heldout_reports(
        report,
        json_path=tmp_path / "heldout.json",
        markdown_path=tmp_path / "heldout.md",
    )

    assert report["inputs"]["paths"]["scored_runs"] == str(scored_path)
    assert report["readiness"]["ready"] is True
    assert (tmp_path / "heldout.json").read_text(encoding="utf-8").startswith("{")
    assert "# Fault-Held-Out" in (tmp_path / "heldout.md").read_text(
        encoding="utf-8"
    )


def _gate(report: Mapping[str, Any], gate_id: str) -> Mapping[str, Any]:
    readiness = cast(Mapping[str, Any], report["readiness"])
    return _readiness_gate(readiness, gate_id)


def _source_gate(report: Mapping[str, Any], gate_id: str) -> Mapping[str, Any]:
    source_transfer = cast(Mapping[str, Any], report["source_transfer"])
    readiness = cast(Mapping[str, Any], source_transfer["readiness"])
    return _readiness_gate(readiness, gate_id)


def _readiness_gate(
    readiness: Mapping[str, Any],
    gate_id: str,
) -> Mapping[str, Any]:
    gates = cast(list[Mapping[str, Any]], readiness["gates"])
    return next(gate for gate in gates if gate["gate_id"] == gate_id)


def _clone_source_records(
    *,
    scored_runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    source: str,
) -> tuple[list[ScoredRun], list[PairwiseExample]]:
    run_id_map = {
        run.run_id: f"{source}::{run.run_id}"
        for run in scored_runs
    }
    cloned_runs = [
        run.model_copy(update={"run_id": run_id_map[run.run_id]})
        for run in scored_runs
    ]
    cloned_pairs = []
    for pair in pairs:
        metadata = dict(pair.metadata)
        metadata["source"] = source
        cloned_pairs.append(
            pair.model_copy(
                update={
                    "pair_id": f"{source}::{pair.pair_id}",
                    "positive_run_id": run_id_map[pair.positive_run_id],
                    "negative_run_id": run_id_map[pair.negative_run_id],
                    "metadata": metadata,
                }
            )
        )
    return cloned_runs, cloned_pairs
