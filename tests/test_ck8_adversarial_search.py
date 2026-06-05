from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_ck7_candidate_recipe_grid import (
    CK7ScheduleSpec,
    build_ck7_candidate_recipes,
    build_guardrail_artifact_specs,
    select_ck7_axes,
)
from scripts.run_ck8_adversarial_search import main

from social_cohesion_vectors.experiments.ck7_candidate_trials import (
    CK7_CANDIDATE_TRIALS,
)
from social_cohesion_vectors.experiments.ck8_adversarial_search import (
    CLAIM_BOUNDARY,
    CK8SearchConfig,
    _rank_evaluations,
    _tournament_decision,
    evaluate_ck8_recipe,
    render_ck8_adversarial_search_markdown,
    run_ck8_adversarial_search,
    write_ck8_adversarial_search_report,
)


def _small_recipe_grid(tmp_path: Path):
    axes = select_ck7_axes(
        [
            "truth_vs_deception",
            "autonomy_vs_coercion",
            "principled_respect_vs_sycophancy",
            "privacy_exit_vs_surveillance_lock_in",
            "constructive_dissent_vs_conformity",
            "manipulation_resistance_vs_persuasion_capture",
        ]
    )
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path / "vectors",
        direction_template="{axis_id}.npz",
        layer=-1,
        strength=0.25,
    )
    schedule = CK7ScheduleSpec(
        schedule_id="decay_then_ramp",
        label="decay then ramp",
        ck1_schedule="decay-8",
        guardrail_schedule="ramp-5-16",
    )
    return build_ck7_candidate_recipes(
        artifacts,
        ck1_direction=tmp_path / "ck1.npz",
        ck1_layers=(-2,),
        ck1_strengths=(0.5, 1.0),
        guardrail_strengths=(0.25,),
        schedules=(schedule,),
        include_controls=True,
        random_control_root=tmp_path / "random",
    )


def test_ck8_search_ranks_mutates_and_records_challengers(tmp_path: Path) -> None:
    recipes = _small_recipe_grid(tmp_path)

    report = run_ck8_adversarial_search(
        recipes,
        config=CK8SearchConfig(
            iterations=2,
            population_size=10,
            elite_count=3,
            mutation_count=6,
            top_k=5,
        ),
    )

    assert report["experiment"] == "ck8_adversarial_candidate_search"
    assert report["dry_run"] is True
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["summary"]["iterations"] == 2
    assert report["summary"]["initial_recipes"] == len(recipes)
    assert report["summary"]["best_recipe_id"] != "baseline"
    assert report["summary"]["best_evidence_status"] == (
        "dry_run_batch_selection_prior"
    )
    assert "best_bottom_tail_pressure_score" in report["summary"]
    assert report["top_candidates"]
    assert report["adversarial_challengers"]
    assert report["iterations"][0]["mutations_created"]
    assert all("recipe_spec" in row for row in report["top_candidates"])
    assert all(
        row["evidence_status"] == "dry_run_batch_selection_prior"
        for row in report["top_candidates"]
    )
    assert all("tournament_metrics" in row for row in report["top_candidates"])
    assert all("tournament_decision" in row for row in report["top_candidates"])
    assert all(
        "control" not in row["family"] for row in report["top_candidates"][:3]
    )


def test_ck8_search_writes_report_markdown_and_recipe_specs(tmp_path: Path) -> None:
    report = run_ck8_adversarial_search(
        _small_recipe_grid(tmp_path),
        config=CK8SearchConfig(iterations=1, population_size=8, top_k=3),
    )
    json_path = tmp_path / "ck8.json"
    markdown_path = tmp_path / "ck8.md"
    recipe_specs_path = tmp_path / "recipes.txt"

    write_ck8_adversarial_search_report(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        recipe_specs_path=recipe_specs_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    recipe_specs = recipe_specs_path.read_text(encoding="utf-8")
    assert "CK-8 Adversarial Candidate Search" in markdown
    assert "not an effect" in markdown
    assert "Tail-risk" in markdown
    assert "bottom-quartile pressure scores" in markdown
    assert report["top_candidates"][0]["recipe_spec"] in recipe_specs


def test_render_ck8_markdown_includes_final_weights(tmp_path: Path) -> None:
    report = run_ck8_adversarial_search(
        _small_recipe_grid(tmp_path),
        config=CK8SearchConfig(iterations=1, population_size=8, top_k=3),
    )

    markdown = render_ck8_adversarial_search_markdown(report)

    assert "Final Adversary Weights" in markdown
    assert "Active Challengers" in markdown
    assert "human, neural, biological" in markdown


def test_ck8_ranking_prioritizes_tail_risk_over_mean_score() -> None:
    ranked = _rank_evaluations(
        [
            {
                "recipe_id": "mean_winner",
                "tournament_decision": "assay_prior",
                "tournament_metrics": {
                    "bottom_tail_pressure_score": 0.44,
                    "worst_pressure_score": 0.32,
                    "slack_delta": 0.01,
                },
                "surrogate_fitness": 0.96,
                "weighted_pressure_score": 0.93,
                "passed_gates": 5,
            },
            {
                "recipe_id": "tail_winner",
                "tournament_decision": "assay_prior",
                "tournament_metrics": {
                    "bottom_tail_pressure_score": 0.72,
                    "worst_pressure_score": 0.69,
                    "slack_delta": 0.02,
                },
                "surrogate_fitness": 0.76,
                "weighted_pressure_score": 0.78,
                "passed_gates": 4,
            },
        ]
    )

    assert ranked[0]["recipe_id"] == "tail_winner"


def test_ck8_tournament_decision_blocks_slack_loss() -> None:
    decision = _tournament_decision(
        {
            "findings": [],
            "side_effect_risk": 0.0,
            "slack_delta": -0.001,
            "worst_pressure_score": 0.90,
            "min_trial_score": 0.68,
        }
    )

    assert decision == "hold_slack_loss"


def test_ck8_guardrail_bundle_records_guardrail_only_prior(tmp_path: Path) -> None:
    recipe = next(
        row
        for row in _small_recipe_grid(tmp_path)
        if row.recipe_id.startswith("ck7_guardrail_bundle_")
    )

    evaluation = evaluate_ck8_recipe(
        recipe,
        trials=CK7_CANDIDATE_TRIALS,
        adversary_weights={trial.failure_target: 1.0 for trial in CK7_CANDIDATE_TRIALS},
        thresholds=CK8SearchConfig().thresholds,
    )
    findings = evaluation["tournament_metrics"]["findings"]

    assert any(row["finding_id"] == "guardrail_only_prior" for row in findings)


def test_ck8_cli_runs_without_existing_vector_files(tmp_path: Path, capsys) -> None:
    output = tmp_path / "ck8.json"
    markdown = tmp_path / "ck8.md"
    recipes = tmp_path / "recipes.txt"

    result = main(
        [
            "--axis",
            "truth_vs_deception",
            "--axis",
            "autonomy_vs_coercion",
            "--axis",
            "principled_respect_vs_sycophancy",
            "--axis",
            "privacy_exit_vs_surveillance_lock_in",
            "--axis",
            "constructive_dissent_vs_conformity",
            "--axis",
            "manipulation_resistance_vs_persuasion_capture",
            "--direction-root",
            str(tmp_path / "missing_vectors"),
            "--ck1-direction",
            str(tmp_path / "missing_ck1.npz"),
            "--ck1-layer",
            "-2",
            "--ck1-strength",
            "0.5",
            "--guardrail-strength",
            "0.25",
            "--schedule",
            "late_clamp=decay-8:ramp-5-16",
            "--iterations",
            "1",
            "--population-size",
            "8",
            "--elite-count",
            "3",
            "--mutation-count",
            "4",
            "--top-k",
            "3",
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
            "--recipe-specs-output",
            str(recipes),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["experiment"] == "ck8_adversarial_candidate_search"
    assert payload["summary"]["mode"] == "dry_run_surrogate"
    assert recipes.read_text(encoding="utf-8").startswith("ck")
    assert "CK-8 adversarial search dry run" in captured.out
    assert "--recipe " in captured.out
