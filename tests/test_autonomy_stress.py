from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.autonomy_stress import (
    autonomy_stress_activation_prompts,
    autonomy_stress_pairwise_examples,
    autonomy_stress_scored_runs,
    export_autonomy_stress_artifacts,
    render_autonomy_stress_markdown,
    shape_autonomy_stress_report,
)


def test_autonomy_stress_suite_scores_all_pairs_correctly() -> None:
    runs = autonomy_stress_scored_runs()
    pairs = autonomy_stress_pairwise_examples()
    prompts = autonomy_stress_activation_prompts()

    assert len(runs) == 32
    assert len(pairs) == 16
    assert len(prompts) == 32
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert all(pair.metadata["mechanism"] for pair in pairs)
    assert all(pair.positive_score > pair.negative_score for pair in pairs)
    assert all(
        float(pair.metadata["autonomy_safety_margin"]) > 0.0 for pair in pairs
    )


def test_autonomy_stress_report_and_export(tmp_path) -> None:
    report = shape_autonomy_stress_report()
    markdown = render_autonomy_stress_markdown(report)

    assert report["summary"]["mechanisms"] == 8
    assert report["summary"]["wording_styles"] == 2
    assert report["summary"]["scorer_pairwise_accuracy"] == 1.0
    assert "Autonomy Stress Suite" in markdown

    counts = export_autonomy_stress_artifacts(
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "scored_runs": 32,
        "pairwise_examples": 16,
        "activation_prompts": 32,
    }
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 16
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")
