from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_generated_fault_source_bundle import (  # noqa: E402
    main as source_bundle_main,
)

from social_cohesion_vectors.datasets import read_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    DEFAULT_VARIANTS,
)
from social_cohesion_vectors.experiments.generated_fault_source_bundle import (  # noqa: E402
    render_generated_fault_source_bundle_pipeline_markdown,
    run_generated_fault_source_bundle_pipeline,
)


def test_generated_fault_source_bundle_exports_two_ready_text_sources(
    tmp_path: Path,
) -> None:
    paths = _source_bundle_paths(tmp_path)

    manifest = _run_source_bundle_pipeline(
        paths,
        variants=DEFAULT_VARIANTS[:1],
    )
    markdown = render_generated_fault_source_bundle_pipeline_markdown(manifest)
    pairs = read_jsonl(paths["pairs_output"])
    prompts = read_jsonl(paths["prompts_output"])

    assert manifest["summary"]["status"] == "not_ready_for_activation_claims"
    assert manifest["summary"]["ready"] is False
    assert manifest["summary"]["styles"] == [
        "length_balanced",
        "length_balanced_alt",
    ]
    assert manifest["summary"]["sources"] == 2
    assert manifest["summary"]["pairwise_examples"] == 60
    assert manifest["summary"]["activation_prompts"] == 120
    assert manifest["summary"]["prompt_records"] == 60
    assert manifest["summary"]["audit_ready_steps"] == 6
    assert manifest["summary"]["audit_not_ready_steps"] == 1
    assert manifest["summary"]["audit_skipped_steps"] == 1
    assert manifest["summary"]["audit_warning_count"] == 0
    assert _step(manifest, "lexical_leakage")["ready"] is True
    assert _step(manifest, "lexical_baseline_diagnostic")["ready"] is True
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "availability_audit")["ready"] is False
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"
    assert len(pairs) == 60
    assert len(prompts) == 120
    assert {row["metadata"]["provider"] for row in pairs} == {"offline"}
    assert {row["metadata"]["source"] for row in pairs} == {
        "generated_fault_class_length_balanced",
        "generated_fault_class_length_balanced_alt",
    }
    assert "Generated Fault Source-Bundle Pipeline" in markdown
    assert "Audit Warnings" not in markdown
    assert paths["dataset_markdown_report"].exists()
    assert paths["pipeline_markdown_report"].exists()


def test_generated_fault_source_bundle_accepts_synthetic_activation_payload(
    tmp_path: Path,
) -> None:
    paths = _source_bundle_paths(tmp_path)
    _run_source_bundle_pipeline(
        paths,
        variants=DEFAULT_VARIANTS[:1],
    )
    activation_path = tmp_path / "activations.npz"
    _write_activation_payload(activation_path, paths["pairs_output"])

    manifest = _run_source_bundle_pipeline(
        paths,
        variants=DEFAULT_VARIANTS[:1],
        activation_npz=activation_path,
    )

    assert manifest["summary"]["status"] == "not_ready_for_activation_claims"
    assert manifest["summary"]["ready"] is False
    assert manifest["summary"]["audit_ready_steps"] == 8
    assert manifest["summary"]["audit_not_ready_steps"] == 1
    assert manifest["summary"]["audit_skipped_steps"] == 0
    assert manifest["summary"]["audit_warning_count"] == 0
    assert _step(manifest, "availability_audit")["ready"] is False
    assert _step(manifest, "activation_metadata_transfer")["ready"] is True
    assert _step(manifest, "activation_transfer_regime_record")["ready"] is True


def test_generated_fault_source_bundle_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _source_bundle_paths(tmp_path)

    exit_code = source_bundle_main(
        [
            "--scored-runs-output",
            str(paths["scored_runs_output"]),
            "--pairs-output",
            str(paths["pairs_output"]),
            "--prompts-output",
            str(paths["prompts_output"]),
            "--prompt-records-output",
            str(paths["prompt_records_output"]),
            "--dataset-json-report",
            str(paths["dataset_json_report"]),
            "--dataset-markdown-report",
            str(paths["dataset_markdown_report"]),
            "--audit-output-dir",
            str(paths["audit_output_dir"]),
            "--pipeline-json-report",
            str(paths["pipeline_json_report"]),
            "--pipeline-markdown-report",
            str(paths["pipeline_markdown_report"]),
            "--variants",
            DEFAULT_VARIANTS[0].name,
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "generated fault source-bundle pipeline" in captured.out
    assert "audit_not_ready=1" in captured.out
    loaded = json.loads(paths["pipeline_json_report"].read_text(encoding="utf-8"))
    assert loaded["summary"]["pairwise_examples"] == 60
    assert loaded["summary"]["audit_warning_count"] == 0
    assert loaded["audit_bundle"]["summary"]["not_ready_steps"] == 1
    assert loaded["audit_bundle"]["summary"]["skipped_steps"] == 1


def _source_bundle_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "scored_runs_output": tmp_path / "processed" / "scored.jsonl",
        "pairs_output": tmp_path / "training" / "pairs.jsonl",
        "prompts_output": tmp_path / "training" / "prompts.jsonl",
        "prompt_records_output": tmp_path / "raw" / "prompt_records.jsonl",
        "dataset_json_report": tmp_path / "reports" / "dataset.json",
        "dataset_markdown_report": tmp_path / "reports" / "dataset.md",
        "audit_output_dir": tmp_path / "reports" / "audit_bundle",
        "pipeline_json_report": tmp_path / "reports" / "pipeline.json",
        "pipeline_markdown_report": tmp_path / "reports" / "pipeline.md",
    }


def _run_source_bundle_pipeline(
    paths: Mapping[str, Path],
    **kwargs: Any,
) -> dict[str, Any]:
    return run_generated_fault_source_bundle_pipeline(
        scored_runs_output=paths["scored_runs_output"],
        pairs_output=paths["pairs_output"],
        prompts_output=paths["prompts_output"],
        prompt_records_output=paths["prompt_records_output"],
        dataset_json_report=paths["dataset_json_report"],
        dataset_markdown_report=paths["dataset_markdown_report"],
        audit_output_dir=paths["audit_output_dir"],
        pipeline_json_report=paths["pipeline_json_report"],
        pipeline_markdown_report=paths["pipeline_markdown_report"],
        **kwargs,
    )


def _write_activation_payload(activation_path: Path, pairs_path: Path) -> None:
    activations: list[list[float]] = []
    pair_ids: list[str] = []
    labels: list[str] = []
    for index, row in enumerate(read_jsonl(pairs_path)):
        group_offset = float(index) / 100.0
        activations.extend([[2.0, group_offset], [0.0, group_offset]])
        pair_ids.extend([str(row["pair_id"]), str(row["pair_id"])])
        labels.extend(["positive", "negative"])
    np.savez(
        activation_path,
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
    )


def _step(manifest: dict[str, object], step_id: str) -> dict[str, object]:
    audit_bundle = manifest["audit_bundle"]
    assert isinstance(audit_bundle, dict)
    steps = audit_bundle["steps"]
    assert isinstance(steps, list)
    return next(step for step in steps if step["step_id"] == step_id)
