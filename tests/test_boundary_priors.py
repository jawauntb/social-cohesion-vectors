from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.boundary_priors import (
    boundary_prior_activation_prompts,
    boundary_prior_pairwise_examples,
    boundary_prior_scored_runs,
    export_boundary_prior_artifacts,
    render_boundary_prior_markdown,
    shape_boundary_prior_report,
)


def test_boundary_prior_benchmark_scores_all_pairs_correctly() -> None:
    runs = boundary_prior_scored_runs()
    pairs = boundary_prior_pairwise_examples()
    prompts = boundary_prior_activation_prompts()

    assert len(runs) == 24
    assert len(pairs) == 12
    assert len(prompts) == 24
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert {pair.metadata["negative_pole"] for pair in pairs} == {
        "rigid_boundary_reification",
        "coercive_boundary_collapse",
    }
    assert all(pair.metadata["mechanism"] for pair in pairs)
    assert all(pair.positive_score > pair.negative_score for pair in pairs)
    assert all(
        float(pair.metadata["autonomy_safety_margin"]) > 0.0 for pair in pairs
    )


def test_boundary_prior_report_and_export(tmp_path) -> None:
    report = shape_boundary_prior_report()
    markdown = render_boundary_prior_markdown(report)

    assert report["summary"]["mechanisms"] == 6
    assert report["summary"]["negative_poles"] == 2
    assert report["summary"]["scorer_pairwise_accuracy"] == 1.0
    assert "Boundary Prior Benchmark" in markdown

    counts = export_boundary_prior_artifacts(
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "scored_runs": 24,
        "pairwise_examples": 12,
        "activation_prompts": 24,
    }
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 12
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")
