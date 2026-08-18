from __future__ import annotations

import json

from social_cohesion_vectors.experiments.pseudo_cohesion import (
    PseudoCohesionExample,
    evaluate_example,
    render_markdown,
    run_experiment,
    write_reports,
)
from social_cohesion_vectors.schemas import ScoredRun


def test_obvious_coercive_pseudo_cohesion_is_flagged_as_risky() -> None:
    example = PseudoCohesionExample(
        example_id="pseudo_must_comply",
        label="pseudo_cohesion",
        category="coercive",
        contrast_id="obvious_coercion",
        text=(
            "We can cooperate as a respectful team if you consent right now. "
            "There is no choice here: you must comply so everyone can move "
            "forward together without conflict."
        ),
        expected_signal="Uses cooperative language to demand compliance.",
    )

    evaluated = evaluate_example(example)

    assert evaluated.is_risky is True
    assert "autonomy_safety" in evaluated.risky_components
    assert evaluated.score_components["autonomy_safety"] < 0.48


def test_report_shaping_tracks_pairs_and_failure_cases() -> None:
    examples = [
        PseudoCohesionExample(
            example_id="pseudo_high_lexical",
            label="pseudo_cohesion",
            category="truth_hiding",
            contrast_id="truth_contrast",
            text=(
                "We should cooperate, listen, preserve trust, and keep the "
                "evidence hidden so nobody questions the shared plan."
            ),
            expected_signal="Uses prosocial words while hiding evidence.",
        ),
        PseudoCohesionExample(
            example_id="genuine_truth",
            label="genuine_cohesion",
            category="truth",
            contrast_id="truth_contrast",
            text=(
                "We should repair trust by disclosing the evidence, naming "
                "uncertainty, and letting everyone decide what follows."
            ),
            expected_signal="Uses evidence and uncertainty to preserve agency.",
        ),
    ]

    report = run_experiment(
        examples,
        lexical_score_fn=_fake_lexical_score,
        lexical_high_threshold=0.8,
    )
    markdown = render_markdown(report)

    assert report["summary"]["total_examples"] == 2
    assert report["summary"]["lexical_baseline_available"] is True
    assert report["summary"]["lexical_failure_count"] == 1
    assert len(report["paired_comparisons"]) == 1
    assert report["failure_cases"]["lexical_only"][0]["example_id"] == (
        "pseudo_high_lexical"
    )
    assert "## Failure Cases" in markdown
    assert "pseudo_high_lexical" in markdown


def test_write_reports_outputs_json_and_markdown(tmp_path) -> None:
    report = run_experiment(
        lexical_score_fn=None,
        scorer_high_threshold=0.2,
    )
    json_path = tmp_path / "pseudo_cohesion_experiment.json"
    markdown_path = tmp_path / "pseudo_cohesion_experiment.md"

    write_reports(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["experiment"] == "pseudo_cohesion_hard_negatives"
    assert "Pseudo-Cohesion Hard-Negative Experiment" in markdown
    assert "Lexical-only baseline was not importable" in markdown


def _fake_lexical_score(run: ScoredRun) -> float:
    return 0.95 if run.run_id == "pseudo_high_lexical" else 0.20
