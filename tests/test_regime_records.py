from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_regime_transition_records import main  # noqa: E402

from social_cohesion_vectors.regime_records import (  # noqa: E402
    DEFAULT_CLAIM_BOUNDARY,
    build_regime_transition_record,
    load_regime_transition_records,
    render_regime_transition_markdown,
    summarize_regime_transition_records,
    write_regime_transition_markdown,
    write_regime_transition_records,
)


def test_regime_transition_record_captures_new_verifier_and_residual() -> None:
    record = build_regime_transition_record(
        record_id="scoring-v2-slack-preservation",
        title="Add slack preservation as a scorer component",
        old_regime={
            "score_components": [
                "cooperation",
                "repair",
                "fairness",
                "hostility_inverse",
                "truthfulness",
                "autonomy_safety",
            ]
        },
        new_regime={
            "score_components": [
                "cooperation",
                "repair",
                "fairness",
                "hostility_inverse",
                "truthfulness",
                "autonomy_safety",
                "slack_preservation",
            ]
        },
        preserved_artifacts=[
            "generated_fault_class_pairwise_probe_dataset",
            "generated_fault_class_pairwise_probe_dataset",
            "autonomy_stress_suite",
        ],
        new_artifact_types=["slack_preservation_component"],
        new_verifiers=["slack_component_margin"],
        rejected_alternatives=[
            {
                "alternative_id": "fold_slack_into_autonomy_safety",
                "reason": "Autonomy safety is broader than future-option closure.",
            }
        ],
        gates=[
            {
                "gate_id": "targeted_scorer_tests",
                "status": "passed",
                "metric": "pytest",
                "observed": 9,
                "threshold": 9,
            }
        ],
        residual_findings=[
            {
                "finding_id": "pseudo_cohesion_closes_futures",
                "description": (
                    "Warm unity language can remove refusal, appeal, evidence, "
                    "privacy, exit, dissent, and repair paths."
                ),
            }
        ],
        source_id="docs/plans/2026-06-05-generated-pseudo-cohesion-slack-regime.md",
    )

    assert record.status == "accepted"
    assert record.claim_boundary == DEFAULT_CLAIM_BOUNDARY
    assert record.preserved_artifacts == [
        "autonomy_stress_suite",
        "generated_fault_class_pairwise_probe_dataset",
    ]
    assert record.new_artifact_types == ["slack_preservation_component"]
    assert record.gates[0].status == "passed"
    assert record.residual_count == 1
    assert record.residual_content == record.residual_findings
    assert record.rejected_alternatives[0]["alternative_id"] == (
        "fold_slack_into_autonomy_safety"
    )


def test_regime_transition_summary_counts_gate_status_and_residuals() -> None:
    records = [
        build_regime_transition_record(
            record_id="r1",
            title="Accepted transition",
            old_regime={},
            new_regime={},
            new_artifact_types=["slack_preservation_component"],
            new_verifiers=["component_margin"],
            gates=[{"gate_id": "tests", "status": "passed"}],
            residual_findings=[{"finding_id": "residual-a"}],
        ),
        build_regime_transition_record(
            record_id="r2",
            title="Rejected transition",
            old_regime={},
            new_regime={},
            status="rejected",
            new_verifiers=["component_margin"],
            gates=[{"gate_id": "leakage", "status": "failed"}],
            residual_findings=[
                {"finding_id": "residual-b"},
                {"finding_id": "residual-c"},
            ],
        ),
    ]

    assert summarize_regime_transition_records(records) == {
        "records": 2,
        "status": {"accepted": 1, "rejected": 1},
        "gate_status": {"failed": 1, "passed": 1},
        "residual_content": 3,
        "residual_findings": 3,
        "new_artifact_types": ["slack_preservation_component"],
        "new_verifiers": ["component_margin"],
        "claim_boundaries": [DEFAULT_CLAIM_BOUNDARY],
    }


def test_regime_records_round_trip_through_jsonl(tmp_path) -> None:
    output_path = tmp_path / "regime_records.jsonl"
    records = [
        build_regime_transition_record(
            record_id="r1",
            title="Accepted transition",
            old_regime={"schema": "old"},
            new_regime={"schema": "new"},
            gates=[{"gate_id": "tests", "status": "passed"}],
        )
    ]

    count = write_regime_transition_records(records, output_path)
    loaded = load_regime_transition_records(output_path)

    assert count == 1
    assert loaded[0].record_id == "r1"
    assert loaded[0].gates[0].gate_id == "tests"
    dumped = json.loads(output_path.read_text(encoding="utf-8"))
    assert "residual_content" in dumped
    assert "claim_boundary" in dumped
    assert "residual_findings" not in dumped


def test_regime_records_accept_legacy_residual_findings_input(tmp_path) -> None:
    input_path = tmp_path / "legacy_regime_records.json"
    input_path.write_text(
        json.dumps(
            {
                "record_id": "legacy",
                "title": "Legacy residual name",
                "old_regime": {},
                "new_regime": {},
                "residual_findings": [{"finding_id": "old-name"}],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_regime_transition_records(input_path)

    assert loaded[0].residual_content == [{"finding_id": "old-name"}]
    assert loaded[0].residual_findings == [{"finding_id": "old-name"}]


def test_regime_markdown_renders_rejections_gates_and_claim_boundary() -> None:
    records = [_benchmark_upgrade_record(), _steering_bottleneck_record()]

    markdown = render_regime_transition_markdown(records)

    assert "# Regime Transition Records" in markdown
    assert "Generated API-authored pseudo-cohesion benchmark" in markdown
    assert "`keyword_only_pseudo_detector`" in markdown
    assert "`source_transfer_readiness` | `passed`" in markdown
    assert "Representation-real, not yet generation-control-real" in markdown
    assert "hidden projection movement survived" in markdown
    assert "Does not claim human, neural" in markdown


def test_export_regime_records_script_normalizes_json_input(tmp_path) -> None:
    input_path = tmp_path / "regime_records.json"
    output_path = tmp_path / "regime_records.jsonl"
    markdown_path = tmp_path / "regime_records.md"
    input_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "r1",
                        "title": "Accepted transition",
                        "old_regime": {"schema": "old"},
                        "new_regime": {"schema": "new"},
                        "gates": [{"gate_id": "tests", "status": "passed"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["record_id"] == "r1"
    assert "Regime Transition Records" in markdown_path.read_text(encoding="utf-8")


def test_regime_markdown_writer_outputs_report(tmp_path) -> None:
    output_path = tmp_path / "regime_records.md"

    write_regime_transition_markdown([_benchmark_upgrade_record()], output_path)

    assert "Generated API-authored pseudo-cohesion benchmark" in (
        output_path.read_text(encoding="utf-8")
    )


def _benchmark_upgrade_record():
    return build_regime_transition_record(
        record_id="benchmark-v3-generated-api-hard-negatives",
        title="Generated API-authored pseudo-cohesion benchmark",
        old_regime={
            "benchmark": "deterministic_cue_balanced_fault_pairs",
            "source_groups": ["generated_fault_class_offline"],
        },
        new_regime={
            "benchmark": "generated_api_authored_fault_pairs",
            "source_groups": [
                "generated_fault_class_offline",
                "generated_fault_class_anthropic",
            ],
        },
        preserved_artifacts=[
            "generated_fault_class_pairwise_probe_dataset",
            "lexical_leakage_report",
            "component_margin_audit",
        ],
        new_artifact_types=[
            "api_fault_prompt_record",
            "source_held_out_transfer_report",
        ],
        new_verifiers=[
            "source_transfer_readiness",
            "missing_metadata_pair_gate",
        ],
        rejected_alternatives=[
            {
                "alternative_id": "keyword_only_pseudo_detector",
                "reason": (
                    "Warm/cohesive wording is not enough; generated source "
                    "transfer must survive held-out provenance groups."
                ),
            }
        ],
        gates=[
            {
                "gate_id": "source_transfer_readiness",
                "status": "passed",
                "metric": "source_groups",
                "observed": 2,
                "threshold": 2,
            }
        ],
        residual_content=[
            {
                "finding_id": "source_generalization_required",
                "description": (
                    "API-authored examples become held-out source opponents "
                    "rather than just provenance counts."
                ),
            }
        ],
    )


def _steering_bottleneck_record():
    return build_regime_transition_record(
        record_id="steering-v1-hidden-movement-behavior-bottleneck",
        title="Representation-real, not yet generation-control-real",
        old_regime={"steering_claim": "hidden projection movement"},
        new_regime={
            "steering_claim": (
                "hidden projection movement plus behavior and anti-compliance gates"
            )
        },
        preserved_artifacts=[
            "activation_projection_telemetry",
            "steered_generation_outputs",
        ],
        new_artifact_types=["behavior_movement_gate", "anti_compliance_control"],
        new_verifiers=["slack_improvement_gate", "pseudo_risk_side_effect_gate"],
        rejected_alternatives=[
            {
                "alternative_id": "projection_only_causal_claim",
                "reason": (
                    "Projection movement alone did not establish generated "
                    "behavior control."
                ),
            }
        ],
        gates=[
            {
                "gate_id": "hidden_projection_movement",
                "status": "passed",
                "metric": "signed_projection_delta",
                "observed": 1.0,
                "threshold": 0.0,
            },
            {
                "gate_id": "behavior_movement",
                "status": "failed",
                "metric": "paired_output_improvement",
                "observed": 0.0,
                "threshold": 1.0,
            },
        ],
        residual_content=[
            {
                "finding_id": "hidden_without_behavior",
                "description": (
                    "hidden projection movement survived, but generated behavior "
                    "barely improved or became noisy."
                ),
            }
        ],
    )
