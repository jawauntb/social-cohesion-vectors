from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.activation_pair_manifest_recovery import (
    recover_pair_manifest_from_activation_npz,
    write_recovered_pair_manifest,
)


def test_recovers_generated_fault_manifest_metadata(tmp_path: Path) -> None:
    activation_npz = tmp_path / "generated.npz"
    pair_id = "generated-fault::accountability_after_harm__generated_neighborhood_forum_modal_hf"
    _write_activation_payload(
        activation_npz,
        pair_ids=[pair_id, pair_id],
        texts=["usable appeal and repair", "appeal waits for approval"],
        scores=[0.7, 0.2],
    )

    result = recover_pair_manifest_from_activation_npz(
        activation_npz,
        dataset_kind="generated_fault",
    )

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.pair_id == pair_id
    assert pair.scenario_id == (
        "accountability_after_harm__generated_neighborhood_forum_modal_hf"
    )
    assert pair.metadata["source"] == "generated_fault_class_primary"
    assert pair.metadata["metadata_recovery"] == "inferred_from_activation_payload"
    assert pair.metadata["primary_fault_class"] == "punitive_accountability"
    assert pair.metadata["base_contrast_id"] == "accountability_after_harm"
    assert pair.metadata["slack_options_tested"] == (
        "refusal,appeal,exit,dissent,repair,proportional_review"
    )
    assert pair.metadata["score_margin"] == 0.5


def test_recovers_procedural_control_manifest_metadata(tmp_path: Path) -> None:
    activation_npz = tmp_path / "control.npz"
    pair_id = "voice_under_pressure::hand_authored_case_notes_v1"
    _write_activation_payload(
        activation_npz,
        pair_ids=[pair_id, pair_id],
        texts=["residents can refuse and inspect evidence", "evidence waits"],
        scores=[0.86, 0.24],
    )

    result = recover_pair_manifest_from_activation_npz(
        activation_npz,
        dataset_kind="procedural_control",
    )

    pair = result.pairs[0]
    assert pair.scenario_id == "procedural_justice_control::voice_under_pressure"
    assert pair.metadata["source"] == "hand_authored_case_notes_v1"
    assert pair.metadata["primary_fault_class"] == "voice_under_pressure"
    assert pair.metadata["slack_options_tested"] == (
        "refusal,dissent,evidence_access,exit"
    )
    assert pair.metadata["claim_boundary"] == "generated_text_excluded_non_human_control"


def test_recovery_writer_and_cli_emit_pairs_and_report(tmp_path: Path) -> None:
    activation_npz = tmp_path / "generated.npz"
    pair_id = "generated-fault::belonging_norms__generated_neighborhood_forum_modal_hf_tournament"
    _write_activation_payload(
        activation_npz,
        pair_ids=[pair_id, pair_id],
        texts=["dissent and exit remain visible", "dissent waits for alignment"],
        scores=[0.62, 0.52],
    )
    pairs_output = tmp_path / "pairs.jsonl"
    report_output = tmp_path / "report.json"

    result = write_recovered_pair_manifest(
        activation_npz,
        dataset_kind="generated_fault",
        pairs_output=pairs_output,
        json_report_output=report_output,
    )

    assert result.report["summary"]["written_pairwise_examples"] == 1
    assert load_pairwise_examples_jsonl(pairs_output)[0].metadata["source"] == (
        "generated_fault_class_repair_v2_tournament"
    )
    assert report_output.exists()

    script = _load_script("recover_activation_pair_manifest.py")
    cli_pairs = tmp_path / "cli_pairs.jsonl"
    assert (
        script.main(
            [
                "--activation-npz",
                str(activation_npz),
                "--dataset-kind",
                "generated_fault",
                "--pairs-output",
                str(cli_pairs),
            ]
        )
        == 0
    )
    assert len(load_pairwise_examples_jsonl(cli_pairs)) == 1


def _write_activation_payload(
    path: Path,
    *,
    pair_ids: list[str],
    texts: list[str],
    scores: list[float],
) -> None:
    labels = ["positive", "negative"]
    sample_ids = [
        f"{pair_id}:{label}"
        for pair_id, label in zip(pair_ids, labels, strict=True)
    ]
    np.savez(
        path,
        activations=np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        sample_ids=np.asarray(sample_ids, dtype=str),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
        target_scores=np.asarray(scores, dtype=np.float32),
        texts=np.asarray(texts, dtype=str),
        model_id=np.asarray("fixture-model"),
        layer=np.asarray(-2),
        pooling=np.asarray("mean"),
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
