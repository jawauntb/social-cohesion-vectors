from __future__ import annotations

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


def test_fault_heldout_transfer_groups_by_primary_fault_class() -> None:
    examples = generated_fault_examples(variants=DEFAULT_VARIANTS[:1])
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(examples)

    report = run_fault_heldout_transfer(scored_runs=scored_runs, pairs=pairs)
    markdown = render_fault_heldout_markdown(report)

    assert report["inputs"]["pairs"] == 30
    assert report["inputs"]["fault_classes"] >= 10
    assert len(report["summary"]) == 3
    assert {row["baseline"] for row in report["summary"]} == {
        "strategy_prior",
        "metrics_only",
        "lexical_only",
    }
    assert all(row["split"] == "fault_class" for row in report["folds"])
    assert "Fault-Held-Out Transfer" in markdown
    assert "consent_bypass" in markdown


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
    assert (tmp_path / "heldout.json").read_text(encoding="utf-8").startswith("{")
    assert "# Fault-Held-Out" in (tmp_path / "heldout.md").read_text(
        encoding="utf-8"
    )
