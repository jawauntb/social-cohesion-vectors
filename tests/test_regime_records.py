from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_regime_transition_records import main  # noqa: E402

from social_cohesion_vectors.regime_records import (  # noqa: E402
    build_regime_transition_record,
    load_regime_transition_records,
    summarize_regime_transition_records,
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
    assert record.preserved_artifacts == [
        "autonomy_stress_suite",
        "generated_fault_class_pairwise_probe_dataset",
    ]
    assert record.new_artifact_types == ["slack_preservation_component"]
    assert record.gates[0].status == "passed"
    assert record.residual_count == 1
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
        "residual_findings": 3,
        "new_artifact_types": ["slack_preservation_component"],
        "new_verifiers": ["component_margin"],
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


def test_export_regime_records_script_normalizes_json_input(tmp_path) -> None:
    input_path = tmp_path / "regime_records.json"
    output_path = tmp_path / "regime_records.jsonl"
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

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["record_id"] == "r1"
