from __future__ import annotations

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.ck7_candidate_trials import (
    CK7_CANDIDATE_TRIALS,
    CLAIM_BOUNDARY,
    ck7_activation_prompts,
    ck7_pairwise_examples,
    ck7_prompt_records,
    ck7_scored_runs,
    export_ck7_candidate_trial_artifacts,
    render_ck7_candidate_trial_markdown,
    shape_ck7_candidate_trial_report,
)


def test_ck7_candidate_trials_cover_pressure_modes_and_failure_targets() -> None:
    records = ck7_prompt_records()
    runs = ck7_scored_runs()
    pairs = ck7_pairwise_examples()
    prompts = ck7_activation_prompts()

    assert len(records) == 12
    assert len(runs) == 24
    assert len(pairs) == 12
    assert len(prompts) == 24
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert {trial.pressure_mode for trial in CK7_CANDIDATE_TRIALS} == {
        "urgent_help",
        "authority_status",
        "emotional_dependency",
        "group_unity",
        "social_debt",
        "high_stakes_uncertainty",
    }
    assert {trial.failure_target for trial in CK7_CANDIDATE_TRIALS} == {
        "sycophancy",
        "hallucination",
        "coercion",
        "dependency_lock_in",
        "privacy_exit_erosion",
        "boundary_collapse",
    }
    assert all(record["claim_boundary"] == CLAIM_BOUNDARY for record in records)


def test_ck7_candidate_trials_score_boundary_preserving_side_higher() -> None:
    pairs = ck7_pairwise_examples()

    assert all(pair.positive_score > pair.negative_score for pair in pairs)
    assert all(float(pair.metadata["score_margin"]) > 0.0 for pair in pairs)
    assert all(float(pair.metadata["autonomy_safety_margin"]) > 0.0 for pair in pairs)
    assert all(float(pair.metadata["truthfulness_margin"]) > 0.0 for pair in pairs)


def test_ck7_candidate_trial_report_and_export(tmp_path) -> None:
    report = shape_ck7_candidate_trial_report()
    markdown = render_ck7_candidate_trial_markdown(report)

    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["summary"]["pressure_modes"] == 6
    assert report["summary"]["failure_targets"] == 6
    assert report["summary"]["scorer_pairwise_accuracy"] == 1.0
    assert "CK-7 Candidate Trials" in markdown
    assert "Claim Boundary" in markdown

    counts = export_ck7_candidate_trial_artifacts(
        prompt_records_output=tmp_path / "prompt_records.jsonl",
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "prompt_records": 12,
        "scored_runs": 24,
        "pairwise_examples": 12,
        "activation_prompts": 24,
    }
    assert len(read_jsonl(tmp_path / "prompt_records.jsonl")) == 12
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 12
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")
