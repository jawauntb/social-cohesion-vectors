"""Generated benchmark audit bundle orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.activation_metadata_transfer import (
    run_activation_metadata_transfer_from_files,
    save_activation_metadata_transfer_report,
)
from social_cohesion_vectors.experiments.component_audit import (
    run_component_margin_audit_from_files,
    save_component_margin_audit,
)
from social_cohesion_vectors.experiments.fault_heldout import (
    run_fault_heldout_transfer_from_files,
    save_fault_heldout_reports,
)
from social_cohesion_vectors.experiments.lexical_leakage import (
    run_lexical_leakage_report_from_file,
    save_lexical_leakage_report,
)
from social_cohesion_vectors.experiments.slack_preservation_audit import (
    run_slack_preservation_audit_from_file,
    save_slack_preservation_audit,
)
from social_cohesion_vectors.experiments.source_diversity_audit import (
    run_source_diversity_audit_from_file,
    save_source_diversity_audit,
)
from social_cohesion_vectors.regime_records import (
    build_activation_metadata_transfer_regime_record,
    write_regime_transition_markdown,
    write_regime_transition_records,
)


def run_generated_benchmark_audit_bundle(
    *,
    scored_runs_path: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    activation_npz: str | Path | None = None,
    group_metadata_key: str = "primary_fault_class",
    source_metadata_key: str = "source",
) -> dict[str, Any]:
    """Run the generated benchmark audit bundle and return its manifest."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scored_runs = Path(scored_runs_path)
    pairs = Path(pairs_path)
    steps: list[dict[str, Any]] = []

    lexical_report = run_lexical_leakage_report_from_file(
        pairs,
        group_metadata_key=group_metadata_key,
    )
    lexical_json = output_path / "generated_benchmark_lexical_leakage.json"
    lexical_markdown = output_path / "generated_benchmark_lexical_leakage.md"
    save_lexical_leakage_report(
        lexical_report,
        json_path=lexical_json,
        markdown_path=lexical_markdown,
    )
    steps.append(
        _step(
            step_id="lexical_leakage",
            report=lexical_report,
            ready=bool(
                _mapping(lexical_report.get("summary")).get("ready_for_activation")
            ),
            readiness_status=str(
                _mapping(lexical_report.get("summary")).get(
                    "activation_readiness",
                    "not_ready",
                )
            ),
            json_path=lexical_json,
            markdown_path=lexical_markdown,
        )
    )

    component_report = run_component_margin_audit_from_files(
        scored_runs_path=scored_runs,
        pairs_path=pairs,
        group_metadata_key=group_metadata_key,
    )
    component_json = output_path / "generated_benchmark_component_audit.json"
    component_markdown = output_path / "generated_benchmark_component_audit.md"
    save_component_margin_audit(
        component_report,
        json_path=component_json,
        markdown_path=component_markdown,
    )
    steps.append(
        _step(
            step_id="component_margin_audit",
            report=component_report,
            ready=bool(
                _mapping(component_report.get("summary")).get("ready_for_activation")
            ),
            readiness_status=str(
                _mapping(component_report.get("summary")).get(
                    "activation_readiness",
                    "not_ready",
                )
            ),
            json_path=component_json,
            markdown_path=component_markdown,
        )
    )

    slack_report = run_slack_preservation_audit_from_file(
        pairs,
        group_metadata_key=group_metadata_key,
    )
    slack_json = output_path / "generated_benchmark_slack_preservation_audit.json"
    slack_markdown = output_path / "generated_benchmark_slack_preservation_audit.md"
    save_slack_preservation_audit(
        slack_report,
        json_path=slack_json,
        markdown_path=slack_markdown,
    )
    steps.append(
        _step(
            step_id="slack_preservation_audit",
            report=slack_report,
            ready=bool(
                _mapping(slack_report.get("summary")).get("ready_for_activation")
            ),
            readiness_status=str(
                _mapping(slack_report.get("summary")).get(
                    "activation_readiness",
                    "not_ready",
                )
            ),
            json_path=slack_json,
            markdown_path=slack_markdown,
        )
    )

    source_report = run_source_diversity_audit_from_file(
        pairs,
        source_metadata_key=source_metadata_key,
        group_metadata_key=group_metadata_key,
    )
    source_json = output_path / "generated_benchmark_source_diversity_audit.json"
    source_markdown = output_path / "generated_benchmark_source_diversity_audit.md"
    save_source_diversity_audit(
        source_report,
        json_path=source_json,
        markdown_path=source_markdown,
    )
    steps.append(
        _step(
            step_id="source_diversity_audit",
            report=source_report,
            ready=bool(
                _mapping(source_report.get("summary")).get("ready_for_activation")
            ),
            readiness_status=str(
                _mapping(source_report.get("summary")).get(
                    "activation_readiness",
                    "not_ready",
                )
            ),
            json_path=source_json,
            markdown_path=source_markdown,
        )
    )

    fault_report = run_fault_heldout_transfer_from_files(
        scored_runs_path=scored_runs,
        pairs_path=pairs,
        metadata_key=group_metadata_key,
        source_metadata_key=source_metadata_key,
    )
    fault_json = output_path / "generated_benchmark_fault_heldout_transfer.json"
    fault_markdown = output_path / "generated_benchmark_fault_heldout_transfer.md"
    save_fault_heldout_reports(
        fault_report,
        json_path=fault_json,
        markdown_path=fault_markdown,
    )
    fault_ready = bool(_mapping(fault_report.get("readiness")).get("ready"))
    source_ready = bool(
        _mapping(_mapping(fault_report.get("source_transfer")).get("readiness")).get(
            "ready"
        )
    )
    steps.append(
        _step(
            step_id="fault_heldout_transfer",
            report=fault_report,
            ready=fault_ready and source_ready,
            readiness_status=_combined_status(
                str(_mapping(fault_report.get("readiness")).get("status", "not_ready")),
                str(
                    _mapping(
                        _mapping(fault_report.get("source_transfer")).get("readiness")
                    ).get("status", "not_ready")
                ),
            ),
            json_path=fault_json,
            markdown_path=fault_markdown,
        )
    )

    if activation_npz is None:
        steps.append(
            {
                "step_id": "activation_metadata_transfer",
                "status": "skipped",
                "ready": False,
                "readiness_status": "activation_npz_not_provided",
                "json_path": None,
                "markdown_path": None,
            }
        )
    else:
        activation_report = run_activation_metadata_transfer_from_files(
            activation_npz=activation_npz,
            pairs_path=pairs,
            metadata_key=group_metadata_key,
        )
        activation_json = output_path / "activation_metadata_transfer.json"
        activation_markdown = output_path / "activation_metadata_transfer.md"
        save_activation_metadata_transfer_report(
            activation_report,
            json_path=activation_json,
            markdown_path=activation_markdown,
        )
        activation_ready = bool(
            _mapping(activation_report.get("summary")).get(
                "ready_for_metadata_coverage_claims"
            )
        ) and bool(
            _mapping(activation_report.get("summary")).get("ready_for_transfer_claims")
        )
        steps.append(
            _step(
                step_id="activation_metadata_transfer",
                report=activation_report,
                ready=activation_ready,
                readiness_status=_combined_status(
                    str(
                        _mapping(activation_report.get("summary")).get(
                            "metadata_coverage_readiness",
                            "not_ready",
                        )
                    ),
                    str(
                        _mapping(activation_report.get("summary")).get(
                            "transfer_readiness",
                            "not_ready",
                        )
                    ),
                ),
                json_path=activation_json,
                markdown_path=activation_markdown,
            )
        )
        regime_record = build_activation_metadata_transfer_regime_record(
            activation_report,
            source_id=str(activation_json),
        )
        regime_jsonl = output_path / "activation_transfer_regime_record.jsonl"
        regime_markdown = output_path / "activation_transfer_regime_record.md"
        write_regime_transition_records([regime_record], regime_jsonl)
        write_regime_transition_markdown([regime_record], regime_markdown)
        steps.append(
            {
                "step_id": "activation_transfer_regime_record",
                "status": "ready"
                if regime_record.status == "accepted"
                else "not_ready",
                "ready": regime_record.status == "accepted",
                "readiness_status": regime_record.status,
                "json_path": str(regime_jsonl),
                "markdown_path": str(regime_markdown),
            }
        )

    manifest = _manifest(
        scored_runs_path=scored_runs,
        pairs_path=pairs,
        activation_npz=Path(activation_npz) if activation_npz is not None else None,
        output_dir=output_path,
        steps=steps,
    )
    manifest_json = output_path / "generated_benchmark_audit_bundle.json"
    manifest_markdown = output_path / "generated_benchmark_audit_bundle.md"
    manifest["manifest_json_path"] = str(manifest_json)
    manifest["manifest_markdown_path"] = str(manifest_markdown)
    write_generated_benchmark_audit_bundle_manifest(
        manifest,
        json_path=manifest_json,
        markdown_path=manifest_markdown,
    )
    return manifest


def write_generated_benchmark_audit_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write generated benchmark audit bundle manifest artifacts."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_generated_benchmark_audit_bundle_markdown(manifest),
        encoding="utf-8",
    )


def render_generated_benchmark_audit_bundle_markdown(
    manifest: Mapping[str, Any],
) -> str:
    """Render a generated benchmark audit bundle manifest."""

    summary = _mapping(manifest.get("summary"))
    lines = [
        "# Generated Benchmark Audit Bundle",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready: {bool(summary.get('ready', False))}",
        f"- Ready steps: {int(summary.get('ready_steps', 0))}",
        f"- Not-ready steps: {int(summary.get('not_ready_steps', 0))}",
        f"- Skipped steps: {int(summary.get('skipped_steps', 0))}",
        "",
        "## Steps",
        "",
        "| Step | Status | Readiness | JSON | Markdown |",
        "| --- | --- | --- | --- | --- |",
    ]
    for raw_step in _sequence(manifest.get("steps")):
        step = _mapping(raw_step)
        lines.append(
            "| "
            f"`{step.get('step_id', '')}` | "
            f"`{step.get('status', '')}` | "
            f"`{step.get('readiness_status', '')}` | "
            f"{step.get('json_path') or ''} | "
            f"{step.get('markdown_path') or ''} |"
        )
    return "\n".join(lines) + "\n"


def _manifest(
    *,
    scored_runs_path: Path,
    pairs_path: Path,
    activation_npz: Path | None,
    output_dir: Path,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    skipped = [step for step in steps if step["status"] == "skipped"]
    not_ready = [step for step in steps if step["status"] == "not_ready"]
    ready = [step for step in steps if step["status"] == "ready"]
    status = "bundle_ready"
    if skipped:
        status = "bundle_incomplete"
    if not_ready:
        status = "not_ready_for_activation_claims"
    return {
        "experiment": "generated_benchmark_audit_bundle",
        "description": (
            "Runs the generated benchmark audit reports as one provenance bundle "
            "and records every output artifact in a manifest."
        ),
        "inputs": {
            "scored_runs_path": str(scored_runs_path),
            "pairs_path": str(pairs_path),
            "activation_npz": str(activation_npz)
            if activation_npz is not None
            else None,
            "output_dir": str(output_dir),
        },
        "summary": {
            "status": status,
            "ready": status == "bundle_ready",
            "steps": len(steps),
            "ready_steps": len(ready),
            "not_ready_steps": len(not_ready),
            "skipped_steps": len(skipped),
        },
        "steps": steps,
    }


def _step(
    *,
    step_id: str,
    report: Mapping[str, Any],
    ready: bool,
    readiness_status: str,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "readiness_status": readiness_status,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "experiment": str(report.get("experiment", "")),
    }


def _combined_status(*statuses: str) -> str:
    return "+".join(statuses)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
