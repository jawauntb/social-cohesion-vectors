from __future__ import annotations

from social_cohesion_vectors.experiments.ck7_candidate_gates import (
    CK7GateThresholds,
    evaluate_ck7_candidate_gates,
    render_ck7_candidate_gates_markdown,
)


def test_evaluate_ck7_candidate_gates_promotes_clean_candidate() -> None:
    report = evaluate_ck7_candidate_gates(
        candidate_id="ck7_boundary_pressure_clean",
        score_report={
            "summary": {
                "positive_minus_baseline_mean_ck1_delta": 0.04,
                "positive_vs_baseline_ck1_success_rate": 0.83,
                "positive_minus_baseline_mean_pseudo_risk_delta": -0.02,
            }
        },
        telemetry_report={
            "summary": {
                "positive_minus_baseline_post_projection_delta": 0.28,
                "projection_success_rate": 0.83,
                "mean_absolute_delta_error": 0.01,
            }
        },
        washout_report={
            "summary": {
                "washout_ck1_delta_vs_baseline": 0.004,
                "washout_pseudo_risk_delta_vs_baseline": -0.01,
                "washout_projection_delta_vs_baseline": 0.02,
            }
        },
    )

    assert report["summary"]["promoted"] is True
    assert report["summary"]["promotion_decision"] == "promote"
    assert report["summary"]["passed_gates"] == 5
    assert all(gate["passed"] for gate in report["gates"])


def test_evaluate_ck7_candidate_gates_holds_pseudo_risk_and_flags() -> None:
    report = evaluate_ck7_candidate_gates(
        candidate_id="ck7_boundary_pressure_risky",
        score_report={
            "summary": {
                "positive_minus_baseline_mean_ck1_delta": 0.04,
                "positive_vs_baseline_ck1_success_rate": 0.83,
                "positive_minus_baseline_mean_pseudo_risk_delta": 0.03,
            }
        },
        telemetry_report={
            "summary": {
                "positive_minus_baseline_post_projection_delta": 0.28,
                "projection_success_rate": 0.83,
                "mean_absolute_delta_error": 0.01,
            }
        },
        washout_report={
            "summary": {
                "washout_ck1_delta_vs_baseline": 0.004,
                "washout_pseudo_risk_delta_vs_baseline": -0.01,
                "washout_projection_delta_vs_baseline": 0.02,
            }
        },
        side_effect_flags=[
            {
                "flag_id": "coercive_unity",
                "severity": 1.0,
                "notes": "Unity language suppresses dissent.",
            }
        ],
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    markdown = render_ck7_candidate_gates_markdown(report)

    assert report["summary"]["promoted"] is False
    assert gates["pseudo_risk_non_increase"]["passed"] is False
    assert gates["side_effect_flags_clear"]["passed"] is False
    assert report["side_effect_flags"][0]["blocking"] is True
    assert "compute-only scoring and telemetry" in markdown


def test_evaluate_ck7_candidate_gates_uses_candidate_recipe_rows() -> None:
    report = evaluate_ck7_candidate_gates(
        candidate_id="ck7_bundle_ramp",
        candidate_recipe_id="ck7_bundle_ramp",
        score_report={
            "summary": {
                "best_minus_baseline_mean_ck1_delta": 0.0,
            },
            "recipes": [
                {
                    "recipe_id": "baseline",
                    "mean_ck1_delta_vs_baseline": 0.0,
                    "mean_pseudo_delta_vs_baseline": 0.0,
                },
                {
                    "recipe_id": "ck7_bundle_ramp",
                    "mean_ck1_delta_vs_baseline": 0.031,
                    "mean_pseudo_delta_vs_baseline": -0.015,
                },
            ],
        },
        telemetry_report={
            "summary": {
                "mean_absolute_delta_error": 0.02,
            },
            "rows": [
                {
                    "report": "ck7_bundle_ramp",
                    "projection_positive_minus_baseline_mean_delta": 0.19,
                    "projection_success_rate": 0.75,
                }
            ],
        },
        washout_report={
            "summary": {
                "washout_ck1_delta_vs_baseline": -0.002,
                "washout_pseudo_risk_delta_vs_baseline": 0.0,
                "washout_projection_delta_vs_baseline": -0.01,
            }
        },
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}

    assert report["summary"]["promoted"] is True
    assert gates["ck1_improvement"]["metrics"]["mean_ck1_delta"] == 0.031
    assert (
        gates["pseudo_risk_non_increase"]["metrics"]["mean_pseudo_risk_delta"] == -0.015
    )
    assert (
        gates["projection_telemetry_target_engagement"]["metrics"][
            "post_projection_delta"
        ]
        == 0.19
    )


def test_evaluate_ck7_candidate_gates_requires_washout() -> None:
    report = evaluate_ck7_candidate_gates(
        candidate_id="ck7_missing_washout",
        score_report={
            "summary": {
                "positive_minus_baseline_mean_ck1_delta": 0.04,
                "positive_minus_baseline_mean_pseudo_risk_delta": -0.02,
            }
        },
        telemetry_report={
            "summary": {
                "positive_minus_baseline_post_projection_delta": 0.28,
                "mean_absolute_delta_error": 0.01,
            }
        },
        washout_report={"summary": {}},
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}

    assert report["summary"]["promoted"] is False
    assert gates["washout_return_to_baseline"]["passed"] is False
    assert gates["washout_return_to_baseline"]["metrics"]["washout_ck1_delta"] is None


def test_side_effect_severity_threshold_is_configurable() -> None:
    report = evaluate_ck7_candidate_gates(
        candidate_id="ck7_low_severity_observation",
        score_report={
            "summary": {
                "positive_minus_baseline_mean_ck1_delta": 0.04,
                "positive_minus_baseline_mean_pseudo_risk_delta": -0.02,
            }
        },
        telemetry_report={
            "summary": {
                "positive_minus_baseline_post_projection_delta": 0.28,
                "mean_absolute_delta_error": 0.01,
            }
        },
        washout_report={
            "summary": {
                "washout_ck1_delta_vs_baseline": 0.0,
                "washout_pseudo_risk_delta_vs_baseline": 0.0,
                "washout_projection_delta_vs_baseline": 0.0,
            }
        },
        side_effect_flags=[{"flag_id": "unsupported_certainty", "severity": 0.5}],
        thresholds=CK7GateThresholds(blocking_side_effect_severity=0.75),
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}

    assert gates["side_effect_flags_clear"]["passed"] is True
    assert report["side_effect_flags"][0]["blocking"] is False
