from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_generated_fault_audit_pipeline import (
    main as pipeline_main,  # noqa: E402
)

from social_cohesion_vectors.datasets import read_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    DEFAULT_VARIANTS,
)
from social_cohesion_vectors.experiments.generated_fault_audit_pipeline import (  # noqa: E402
    render_generated_fault_audit_pipeline_markdown,
    run_generated_fault_audit_pipeline,
)


def test_generated_fault_audit_pipeline_exports_and_audits_cue_balanced_dataset(
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)

    manifest = run_generated_fault_audit_pipeline(
        **paths,
        variants=DEFAULT_VARIANTS[:1],
        style="cue_balanced",
    )
    markdown = render_generated_fault_audit_pipeline_markdown(manifest)

    assert manifest["summary"]["style"] == "cue_balanced"
    assert manifest["summary"]["pairwise_examples"] == 30
    assert manifest["summary"]["activation_prompts"] == 60
    assert manifest["summary"]["status"] == "not_ready_for_activation_claims"
    assert manifest["summary"]["ready"] is False
    assert manifest["summary"]["audit_not_ready_steps"] == 3
    assert manifest["summary"]["audit_skipped_steps"] == 1
    assert _step(manifest, "slack_preservation_audit")["ready"] is True
    assert _step(manifest, "availability_audit")["ready"] is False
    assert _step(manifest, "source_diversity_audit")["ready"] is False
    assert _step(manifest, "fault_heldout_transfer")["ready"] is False
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"
    assert len(read_jsonl(paths["pairs_output"])) == 30
    assert len(read_jsonl(paths["prompts_output"])) == 60
    assert paths["pipeline_json_report"].exists()
    assert paths["pipeline_markdown_report"].exists()
    assert paths["dataset_markdown_report"].exists()
    assert "Generated Fault-Class Audit Pipeline" in markdown
    assert "not_ready_for_activation_claims" in markdown


def test_generated_fault_audit_pipeline_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _pipeline_paths(tmp_path)

    exit_code = pipeline_main(
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
    assert "generated fault audit pipeline" in captured.out
    assert "status=not_ready_for_activation_claims" in captured.out
    loaded = json.loads(paths["pipeline_json_report"].read_text(encoding="utf-8"))
    assert loaded["summary"]["pairwise_examples"] == 30
    assert loaded["audit_bundle"]["summary"]["not_ready_steps"] == 3
    assert loaded["audit_bundle"]["summary"]["skipped_steps"] == 1


def _pipeline_paths(tmp_path: Path) -> dict[str, Path]:
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


def _step(manifest: dict[str, object], step_id: str) -> dict[str, object]:
    audit_bundle = manifest["audit_bundle"]
    assert isinstance(audit_bundle, dict)
    steps = audit_bundle["steps"]
    assert isinstance(steps, list)
    return next(step for step in steps if step["step_id"] == step_id)
