from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.affect_controls import (
    AFFECT_LABELS,
    affect_control_activation_prompts,
    affect_control_contrasts,
    affect_control_pairwise_examples,
    affect_control_scored_runs,
    affect_feature_vector,
    export_affect_control_artifacts,
    render_affect_control_markdown,
    shape_affect_control_report,
)


def test_affect_control_crosses_boundary_priors_with_six_affect_frames() -> None:
    contrasts = affect_control_contrasts()
    runs = affect_control_scored_runs(contrasts)
    pairs = affect_control_pairwise_examples(contrasts)
    prompts = affect_control_activation_prompts(contrasts)

    assert len(contrasts) == 72
    assert len(runs) == 144
    assert len(pairs) == 72
    assert len(prompts) == 144
    assert {contrast.affect_label for contrast in contrasts} == set(AFFECT_LABELS)
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert all(pair.metadata["affect_label"] for pair in pairs)
    assert all(pair.positive_score > pair.negative_score for pair in pairs)


def test_affect_features_count_coarse_nova_style_labels() -> None:
    features = affect_feature_vector(
        "anger and fear were named in the neutral record, but hope remained."
    )

    assert features["anger"] == 1.0
    assert features["fear"] == 1.0
    assert features["neutral"] == 2.0
    assert features["happy"] == 1.0


def test_affect_control_report_and_export(tmp_path) -> None:
    report = shape_affect_control_report()
    markdown = render_affect_control_markdown(report)

    assert report["summary"]["affect_classes"] == 6
    assert report["summary"]["mechanisms"] == 6
    assert report["summary"]["negative_poles"] == 2
    assert report["summary"]["scorer_pairwise_accuracy"] == 1.0
    assert 0.0 <= report["summary"]["affect_only_pair_loo_accuracy"] <= 1.0
    assert 0.0 <= report["summary"]["residualized_pair_loo_accuracy"] <= 1.0
    assert "Affect-Control Residualization Benchmark" in markdown

    counts = export_affect_control_artifacts(
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "scored_runs": 144,
        "pairwise_examples": 72,
        "activation_prompts": 144,
    }
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 72
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")
