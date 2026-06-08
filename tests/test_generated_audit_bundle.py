from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_generated_benchmark_audit_bundle import (
    main as bundle_main,  # noqa: E402
)

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    DEFAULT_VARIANTS,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    scored_runs_from_generated_fault_examples,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    render_generated_benchmark_audit_bundle_markdown,
    run_generated_benchmark_audit_bundle,
)  # noqa: E402
from social_cohesion_vectors.regime_records import (  # noqa: E402
    load_regime_transition_records,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun  # noqa: E402


def test_generated_benchmark_audit_bundle_marks_missing_activation_as_skipped(
    tmp_path: Path,
) -> None:
    scored_path, pairs_path, _activation_path = _write_bundle_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_path,
        pairs_path=pairs_path,
        output_dir=output_dir,
    )
    markdown = render_generated_benchmark_audit_bundle_markdown(manifest)

    assert manifest["summary"]["status"] == "bundle_incomplete"
    assert manifest["summary"]["ready"] is False
    assert manifest["summary"]["ready_steps"] == 5
    assert manifest["summary"]["skipped_steps"] == 1
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"
    assert _step(manifest, "slack_preservation_audit")["ready"] is True
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "fault_heldout_transfer")["ready"] is True
    assert "activation_npz_not_provided" in markdown
    assert (output_dir / "generated_benchmark_audit_bundle.json").exists()
    assert (output_dir / "generated_benchmark_audit_bundle.md").exists()
    saved_manifest = json.loads(
        (output_dir / "generated_benchmark_audit_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_manifest["manifest_markdown_path"].endswith(
        "generated_benchmark_audit_bundle.md"
    )


def test_generated_benchmark_audit_bundle_accepts_two_lexically_safe_sources(
    tmp_path: Path,
) -> None:
    scored_path, pairs_path = _write_lexically_safe_two_source_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_path,
        pairs_path=pairs_path,
        output_dir=output_dir,
    )

    assert manifest["summary"]["status"] == "bundle_incomplete"
    assert manifest["summary"]["ready"] is False
    assert manifest["summary"]["ready_steps"] == 5
    assert manifest["summary"]["not_ready_steps"] == 0
    assert manifest["summary"]["skipped_steps"] == 1
    assert _step(manifest, "lexical_leakage")["ready"] is True
    assert _step(manifest, "component_margin_audit")["ready"] is True
    assert _step(manifest, "slack_preservation_audit")["ready"] is True
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "fault_heldout_transfer")["ready"] is True
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"


def test_generated_benchmark_audit_bundle_writes_activation_regime_record(
    tmp_path: Path,
) -> None:
    scored_path, pairs_path, activation_path = _write_bundle_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_path,
        pairs_path=pairs_path,
        activation_npz=activation_path,
        output_dir=output_dir,
    )

    assert manifest["summary"]["status"] == "bundle_ready"
    assert manifest["summary"]["ready"] is True
    assert manifest["summary"]["ready_steps"] == 7
    assert manifest["summary"]["skipped_steps"] == 0
    assert _step(manifest, "slack_preservation_audit")["ready"] is True
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "activation_metadata_transfer")["ready"] is True
    assert _step(manifest, "activation_transfer_regime_record")["ready"] is True
    records = load_regime_transition_records(
        output_dir / "activation_transfer_regime_record.jsonl"
    )
    assert records[0].status == "accepted"
    assert "activation_transfer_readiness" in records[0].new_verifiers
    assert (output_dir / "activation_metadata_transfer.md").exists()
    assert (output_dir / "activation_transfer_regime_record.md").exists()


def test_generated_benchmark_audit_bundle_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    scored_path, pairs_path, activation_path = _write_bundle_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    exit_code = bundle_main(
        [
            "--scored-runs",
            str(scored_path),
            "--pairs",
            str(pairs_path),
            "--activation-npz",
            str(activation_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status=bundle_ready" in captured.out
    assert (output_dir / "generated_benchmark_audit_bundle.json").exists()
    assert (output_dir / "generated_benchmark_audit_bundle.md").exists()


def _write_bundle_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    runs, pairs = _bundle_fixture_records()
    scored_path = tmp_path / "generated_fault_class_scored_runs.jsonl"
    pairs_path = tmp_path / "generated_fault_class_pairs.jsonl"
    activation_path = tmp_path / "activations.npz"
    write_jsonl(runs, scored_path)
    write_jsonl(pairs, pairs_path)
    _write_activation_payload(activation_path, pairs)
    return scored_path, pairs_path, activation_path


def _write_lexically_safe_two_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    cue_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    hardened_examples = generated_fault_examples(
        variants=DEFAULT_VARIANTS[:1],
        style="lexical_hardened",
    )
    runs = [
        *scored_runs_from_generated_fault_examples(cue_examples),
        *scored_runs_from_generated_fault_examples(hardened_examples),
    ]
    pairs = [
        *pairwise_examples_from_generated_fault_examples(
            cue_examples,
            source="generated_fault_class_cue_balanced",
            style="cue_balanced",
        ),
        *pairwise_examples_from_generated_fault_examples(
            hardened_examples,
            source="generated_fault_class_lexical_hardened",
            style="lexical_hardened",
        ),
    ]
    scored_path = tmp_path / "generated_fault_class_scored_runs.jsonl"
    pairs_path = tmp_path / "generated_fault_class_pairs.jsonl"
    write_jsonl(runs, scored_path)
    write_jsonl(pairs, pairs_path)
    return scored_path, pairs_path


def _bundle_fixture_records() -> tuple[list[ScoredRun], list[PairwiseExample]]:
    specs = [
        ("p1", "consent_bypass", "offline", "cue_balanced"),
        ("p2", "consent_bypass", "anthropic", "api_authored"),
        ("p3", "exit_rights", "offline", "cue_balanced"),
        ("p4", "exit_rights", "anthropic", "api_authored"),
    ]
    runs: list[ScoredRun] = []
    pairs: list[PairwiseExample] = []
    for index, (pair_id, fault_class, provider, style) in enumerate(specs, start=1):
        positive = _run(pair_id, label="positive", score=0.9, seed=index)
        negative = _run(pair_id, label="negative", score=0.1, seed=index)
        runs.extend([positive, negative])
        pairs.append(
            PairwiseExample(
                pair_id=pair_id,
                scenario_id=f"scenario-{pair_id}",
                positive_run_id=positive.run_id,
                negative_run_id=negative.run_id,
                positive_text=f"Record {pair_id}: logs stay visible; appeal and exit remain open.",
                negative_text=f"Record {pair_id}: logs stay visible; appeal and exit are closed.",
                positive_score=positive.cohesion_score,
                negative_score=negative.cohesion_score,
                metadata={
                    "primary_fault_class": fault_class,
                    "fault_classes": f"{fault_class},shared_fault",
                    "slack_options_tested": "refusal,appeal,evidence_access,privacy_choice,exit,dissent,repair,proportional_review",
                    "positive_slack_preservation": 0.9,
                    "negative_slack_preservation": 0.1,
                    "slack_preservation_margin": 0.8,
                    "source": f"generated_fault_class_{provider}",
                    "provider": provider,
                    "generated_style": style,
                },
            )
        )
    return runs, pairs


def _run(
    pair_id: str,
    *,
    label: str,
    score: float,
    seed: int,
) -> ScoredRun:
    is_positive = label == "positive"
    components = _components(high=is_positive)
    return ScoredRun(
        run_id=f"{pair_id}:{label}",
        scenario_id=f"scenario-{pair_id}",
        intervention="truth_first" if is_positive else "none",
        strategy_profile="cooperative" if is_positive else "adversarial",
        seed=seed,
        transcript=(
            f"Record {pair_id} {label}: channel status remains neutral; "
            "participants can inspect logs and leave the room."
        ),
        events=[],
        metrics={
            "cooperation_rate": 0.9 if is_positive else 0.1,
            "repair_attempt_rate": 0.8 if is_positive else 0.1,
            "fairness_score": 0.85 if is_positive else 0.2,
            "hostility_rate": 0.05 if is_positive else 0.7,
            "joint_payoff": 0.9 if is_positive else 0.2,
            "defection_rate": 0.05 if is_positive else 0.8,
        },
        cohesion_score=score,
        score_components=components,
    )


def _components(*, high: bool) -> dict[str, float]:
    positive_value = 0.9 if high else 0.1
    inverse_value = 0.95 if high else 0.2
    return {
        "cooperation": positive_value,
        "repair": positive_value,
        "fairness": positive_value,
        "hostility_inverse": inverse_value,
        "truthfulness": positive_value,
        "autonomy_safety": positive_value,
        "slack_preservation": positive_value,
    }


def _write_activation_payload(
    activation_path: Path,
    pairs: list[PairwiseExample],
) -> None:
    activations: list[list[float]] = []
    pair_ids: list[str] = []
    labels: list[str] = []
    for index, pair in enumerate(pairs):
        group_offset = float(index) / 10.0
        activations.extend([[2.0, group_offset], [0.0, group_offset]])
        pair_ids.extend([pair.pair_id, pair.pair_id])
        labels.extend(["positive", "negative"])
    np.savez(
        activation_path,
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
    )


def _step(manifest: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in manifest["steps"] if step["step_id"] == step_id)
