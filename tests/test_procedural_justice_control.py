from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.procedural_justice_control import (
    CONTROL_CONTRACT_VERSION,
    render_procedural_justice_control_pipeline_markdown,
    run_procedural_justice_control_pipeline,
)


def test_procedural_justice_control_pipeline_exports_non_generated_control(
    tmp_path: Path,
) -> None:
    paths = _control_paths(tmp_path)

    manifest = run_procedural_justice_control_pipeline(
        scored_runs_output=paths["scored_runs_output"],
        pairs_output=paths["pairs_output"],
        prompts_output=paths["prompts_output"],
        dataset_json_report=paths["dataset_json_report"],
        dataset_markdown_report=paths["dataset_markdown_report"],
        audit_output_dir=paths["audit_output_dir"],
        pipeline_json_report=paths["pipeline_json_report"],
        pipeline_markdown_report=paths["pipeline_markdown_report"],
    )
    markdown = render_procedural_justice_control_pipeline_markdown(manifest)
    pairs = read_jsonl(paths["pairs_output"])
    prompts = read_jsonl(paths["prompts_output"])

    assert manifest["summary"]["status"] == (
        "control_ready_for_activation_extraction"
    )
    assert manifest["summary"]["ready_for_activation_extraction"] is True
    assert manifest["summary"]["ready_for_activation_claims"] is False
    assert manifest["summary"]["pairwise_examples"] == 16
    assert manifest["summary"]["activation_prompts"] == 32
    assert manifest["summary"]["sources"] == 4
    assert manifest["summary"]["audit_not_ready_steps"] == 0
    assert manifest["summary"]["audit_skipped_steps"] == 1
    assert manifest["summary"]["audit_warning_count"] == 0
    assert _step(manifest, "lexical_leakage")["ready"] is True
    assert _step(manifest, "component_margin_audit")["ready"] is True
    assert _step(manifest, "slack_preservation_audit")["ready"] is True
    assert _step(manifest, "availability_audit")["ready"] is True
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "fault_heldout_transfer")["ready"] is True
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"
    assert len(pairs) == 16
    assert len(prompts) == 32
    assert {row["metadata"]["artifact_class"] for row in pairs} == {
        "non_generated_control"
    }
    assert {row["metadata"]["control_contract_version"] for row in pairs} == {
        CONTROL_CONTRACT_VERSION
    }
    assert "Procedural-Justice Control Pipeline" in markdown
    assert paths["pipeline_markdown_report"].exists()


def test_procedural_justice_control_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    paths = _control_paths(tmp_path)

    exit_code = script.main(
        [
            "--scored-runs-output",
            str(paths["scored_runs_output"]),
            "--pairs-output",
            str(paths["pairs_output"]),
            "--prompts-output",
            str(paths["prompts_output"]),
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
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "procedural-justice control pipeline" in captured.out
    loaded = json.loads(paths["pipeline_json_report"].read_text(encoding="utf-8"))
    assert loaded["summary"]["status"] == "control_ready_for_activation_extraction"
    assert loaded["summary"]["audit_warning_count"] == 0


def _control_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "scored_runs_output": tmp_path / "processed" / "scored.jsonl",
        "pairs_output": tmp_path / "training" / "pairs.jsonl",
        "prompts_output": tmp_path / "training" / "prompts.jsonl",
        "dataset_json_report": tmp_path / "reports" / "dataset.json",
        "dataset_markdown_report": tmp_path / "reports" / "dataset.md",
        "audit_output_dir": tmp_path / "reports" / "audit_bundle",
        "pipeline_json_report": tmp_path / "reports" / "pipeline.json",
        "pipeline_markdown_report": tmp_path / "reports" / "pipeline.md",
    }


def _step(manifest: Mapping[str, object], step_id: str) -> Mapping[str, object]:
    audit_bundle = manifest["audit_bundle"]
    assert isinstance(audit_bundle, dict)
    steps = audit_bundle["steps"]
    assert isinstance(steps, list)
    return next(step for step in steps if step["step_id"] == step_id)


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_procedural_justice_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_procedural_justice_control",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
