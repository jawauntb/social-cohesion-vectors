"""End-to-end generated fault-class benchmark export and audit pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    FaultGenerationStyle,
    FaultGenerationVariant,
    export_generated_fault_dataset,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)


def run_generated_fault_audit_pipeline(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    prompt_records_output: str | Path,
    dataset_json_report: str | Path,
    dataset_markdown_report: str | Path,
    audit_output_dir: str | Path,
    pipeline_json_report: str | Path,
    pipeline_markdown_report: str | Path,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    style: FaultGenerationStyle = "cue_balanced",
    activation_npz: str | Path | None = None,
) -> dict[str, Any]:
    """Export generated fault-class artifacts, then run the audit bundle."""

    counts = export_generated_fault_dataset(
        scored_runs_output=scored_runs_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        prompt_records_output=prompt_records_output,
        json_report_output=dataset_json_report,
        markdown_report_output=dataset_markdown_report,
        variants=variants,
        style=style,
    )
    audit_manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_runs_output,
        pairs_path=pairs_output,
        output_dir=audit_output_dir,
        activation_npz=activation_npz,
    )
    manifest = _pipeline_manifest(
        counts=counts,
        variants=variants,
        style=style,
        scored_runs_output=Path(scored_runs_output),
        pairs_output=Path(pairs_output),
        prompts_output=Path(prompts_output),
        prompt_records_output=Path(prompt_records_output),
        dataset_json_report=Path(dataset_json_report),
        dataset_markdown_report=Path(dataset_markdown_report),
        audit_output_dir=Path(audit_output_dir),
        activation_npz=Path(activation_npz) if activation_npz is not None else None,
        audit_manifest=audit_manifest,
    )
    manifest["pipeline_json_report"] = str(pipeline_json_report)
    manifest["pipeline_markdown_report"] = str(pipeline_markdown_report)
    write_generated_fault_audit_pipeline_manifest(
        manifest,
        json_path=pipeline_json_report,
        markdown_path=pipeline_markdown_report,
    )
    return manifest


def write_generated_fault_audit_pipeline_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write generated fault-class pipeline manifest artifacts."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_generated_fault_audit_pipeline_markdown(manifest),
        encoding="utf-8",
    )


def render_generated_fault_audit_pipeline_markdown(
    manifest: Mapping[str, Any],
) -> str:
    """Render a generated fault-class pipeline manifest."""

    summary = _mapping(manifest.get("summary"))
    artifacts = _mapping(manifest.get("artifacts"))
    audit = _mapping(manifest.get("audit_bundle"))
    lines = [
        "# Generated Fault-Class Audit Pipeline",
        "",
        str(manifest.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready: {bool(summary.get('ready', False))}",
        f"- Style: `{summary.get('style', '')}`",
        f"- Variants: {', '.join(_strings(summary.get('variants')))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        f"- Audit bundle status: `{summary.get('audit_bundle_status', 'unknown')}`",
        f"- Audit skipped steps: {int(summary.get('audit_skipped_steps', 0))}",
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
    ]
    for key, value in artifacts.items():
        lines.append(f"| `{key}` | {value or ''} |")
    lines.extend(
        [
            "",
            "## Audit Bundle",
            "",
            f"- Manifest JSON: {audit.get('manifest_json_path', '')}",
            f"- Manifest Markdown: {audit.get('manifest_markdown_path', '')}",
            "",
            "| Step | Status | Readiness |",
            "| --- | --- | --- |",
        ]
    )
    for raw_step in _sequence(audit.get("steps")):
        step = _mapping(raw_step)
        lines.append(
            "| "
            f"`{step.get('step_id', '')}` | "
            f"`{step.get('status', '')}` | "
            f"`{step.get('readiness_status', '')}` |"
        )
    return "\n".join(lines) + "\n"


def _pipeline_manifest(
    *,
    counts: Mapping[str, int],
    variants: Sequence[FaultGenerationVariant],
    style: FaultGenerationStyle,
    scored_runs_output: Path,
    pairs_output: Path,
    prompts_output: Path,
    prompt_records_output: Path,
    dataset_json_report: Path,
    dataset_markdown_report: Path,
    audit_output_dir: Path,
    activation_npz: Path | None,
    audit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit_summary = _mapping(audit_manifest.get("summary"))
    audit_status = str(audit_summary.get("status", "unknown"))
    return {
        "experiment": "generated_fault_audit_pipeline",
        "description": (
            "Exports the cue-balanced generated fault-class benchmark and runs "
            "the generated benchmark audit bundle against the produced artifacts."
        ),
        "summary": {
            "status": audit_status,
            "ready": bool(audit_summary.get("ready", False)),
            "style": style,
            "variants": [variant.name for variant in variants],
            "scored_runs": int(counts.get("scored_runs", 0)),
            "pairwise_examples": int(counts.get("pairwise_examples", 0)),
            "activation_prompts": int(counts.get("activation_prompts", 0)),
            "prompt_records": int(counts.get("prompt_records", 0)),
            "audit_bundle_status": audit_status,
            "audit_ready_steps": int(audit_summary.get("ready_steps", 0)),
            "audit_not_ready_steps": int(audit_summary.get("not_ready_steps", 0)),
            "audit_skipped_steps": int(audit_summary.get("skipped_steps", 0)),
        },
        "artifacts": {
            "scored_runs": str(scored_runs_output),
            "pairwise_examples": str(pairs_output),
            "activation_prompts": str(prompts_output),
            "prompt_records": str(prompt_records_output),
            "dataset_json_report": str(dataset_json_report),
            "dataset_markdown_report": str(dataset_markdown_report),
            "audit_output_dir": str(audit_output_dir),
            "activation_npz": str(activation_npz)
            if activation_npz is not None
            else None,
        },
        "audit_bundle": audit_manifest,
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return [str(item) for item in _sequence(value)]
