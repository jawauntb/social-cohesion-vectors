from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.social_games import (
    export_social_game_artifacts,
    render_social_game_markdown,
    shape_social_game_report,
    social_game_activation_prompts,
    social_game_pairwise_examples,
    social_game_scored_runs,
)


def test_social_game_scaffold_exports_schema_valid_pairs() -> None:
    runs = social_game_scored_runs()
    pairs = social_game_pairwise_examples()
    prompts = social_game_activation_prompts()

    assert len(runs) == 10
    assert len(pairs) == 5
    assert len(prompts) == 10
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert all(pair.metadata["game_kind"] for pair in pairs)
    scorer_failures = [
        pair for pair in pairs if pair.positive_score <= pair.negative_score
    ]
    assert [pair.metadata["game_kind"] for pair in scorer_failures] == ["trust_game"]


def test_social_game_report_and_export(tmp_path) -> None:
    report = shape_social_game_report()
    markdown = render_social_game_markdown(report)

    assert report["summary"]["game_kinds"] == 5
    assert report["summary"]["scorer_pairwise_accuracy"] == 0.8
    assert "dictator_game" in markdown

    counts = export_social_game_artifacts(
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "scored_runs": 10,
        "pairwise_examples": 5,
        "activation_prompts": 10,
    }
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 5
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")
