"""Multi-source generated fault-class benchmark export and audit pipeline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    FaultGenerationStyle,
    FaultGenerationVariant,
    build_fault_prompt_records,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)
from social_cohesion_vectors.schemas import PairwiseExample

DEFAULT_SOURCE_BUNDLE_STYLES: tuple[FaultGenerationStyle, ...] = (
    "length_balanced",
    "length_balanced_alt",
)


def run_generated_fault_source_bundle_pipeline(
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
    styles: Sequence[FaultGenerationStyle] = DEFAULT_SOURCE_BUNDLE_STYLES,
    activation_npz: str | Path | None = None,
) -> dict[str, Any]:
    """Export a multi-source generated benchmark, then run the audit bundle."""

    counts, dataset_report = export_generated_fault_source_bundle(
        scored_runs_output=scored_runs_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        prompt_records_output=prompt_records_output,
        json_report_output=dataset_json_report,
        markdown_report_output=dataset_markdown_report,
        variants=variants,
        styles=styles,
    )
    audit_manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_runs_output,
        pairs_path=pairs_output,
        output_dir=audit_output_dir,
        activation_npz=activation_npz,
    )
    manifest = _pipeline_manifest(
        counts=counts,
        dataset_report=dataset_report,
        variants=variants,
        styles=styles,
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
    write_generated_fault_source_bundle_manifest(
        manifest,
        json_path=pipeline_json_report,
        markdown_path=pipeline_markdown_report,
    )
    return manifest


def export_generated_fault_source_bundle(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    prompt_records_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    styles: Sequence[FaultGenerationStyle] = DEFAULT_SOURCE_BUNDLE_STYLES,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write combined generated-fault artifacts for multiple source styles."""

    style_tuple = _validate_styles(styles)
    scored_runs = []
    pairs: list[PairwiseExample] = []
    style_reports: dict[str, Mapping[str, Any]] = {}
    for style in style_tuple:
        examples = generated_fault_examples(variants=variants, style=style)
        scored_runs.extend(scored_runs_from_generated_fault_examples(examples))
        pairs.extend(
            pairwise_examples_from_generated_fault_examples(
                examples,
                source=_source_for_style(style),
                provider="offline",
                style=style,
            )
        )
        style_reports[style] = shape_generated_fault_report(
            examples,
            variants=variants,
            style=style,
        )

    prompts = activation_prompts_from_pairs(pairs)
    prompt_records = build_fault_prompt_records(variants=variants)
    report = shape_generated_fault_source_bundle_report(
        pairs=pairs,
        variants=variants,
        styles=style_tuple,
        style_reports=style_reports,
    )
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
        "prompt_records": write_jsonl(prompt_records, prompt_records_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_generated_fault_source_bundle_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def shape_generated_fault_source_bundle_report(
    *,
    pairs: Sequence[PairwiseExample],
    variants: Sequence[FaultGenerationVariant],
    styles: Sequence[FaultGenerationStyle],
    style_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize combined generated-fault source coverage."""

    sources = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    styles_by_source: dict[str, set[str]] = {}
    for pair in pairs:
        source = str(pair.metadata.get("source", ""))
        style = str(pair.metadata.get("generated_style", ""))
        styles_by_source.setdefault(source, set()).add(style)
    return {
        "experiment": "generated_fault_source_bundle",
        "description": (
            "Combines multiple deterministic generated-fault text styles into "
            "a single source-diverse benchmark for audit and activation runs."
        ),
        "summary": {
            "styles": list(styles),
            "variants": [variant.name for variant in variants],
            "sources": len([source for source in sources if source]),
            "pairs": len(pairs),
            "activation_prompts": len(pairs) * 2,
            "source_counts": dict(sorted(sources.items())),
            "styles_by_source": {
                source: sorted(style_values)
                for source, style_values in sorted(styles_by_source.items())
            },
        },
        "style_reports": {
            style: _summary_only(report)
            for style, report in sorted(style_reports.items())
        },
    }


def render_generated_fault_source_bundle_markdown(report: Mapping[str, Any]) -> str:
    """Render combined source-bundle coverage as markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Generated Fault Source Bundle",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Styles: {', '.join(_strings(summary.get('styles')))}",
        f"- Variants: {', '.join(_strings(summary.get('variants')))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        "",
        "## Source Counts",
        "",
        "| Source | Pairs | Styles |",
        "| --- | ---: | --- |",
    ]
    styles_by_source = _mapping(summary.get("styles_by_source"))
    for source, count in _mapping(summary.get("source_counts")).items():
        source_styles = _strings(styles_by_source.get(source))
        lines.append(f"| `{source}` | {int(count)} | {', '.join(source_styles)} |")
    lines.extend(
        [
            "",
            "## Style Reports",
            "",
            "| Style | Pairs | Scorer accuracy | Slack-prefers-genuine |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for style, raw_report in _mapping(report.get("style_reports")).items():
        style_summary = _mapping(_mapping(raw_report).get("summary"))
        lines.append(
            "| "
            f"`{style}` | "
            f"{int(style_summary.get('pairs', 0))} | "
            f"{float(style_summary.get('scorer_accuracy', 0.0)):.3f} | "
            f"{int(style_summary.get('slack_prefers_genuine', 0))} |"
        )
    return "\n".join(lines) + "\n"


def write_generated_fault_source_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write source-bundle pipeline manifest artifacts."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_generated_fault_source_bundle_pipeline_markdown(manifest),
        encoding="utf-8",
    )


def render_generated_fault_source_bundle_pipeline_markdown(
    manifest: Mapping[str, Any],
) -> str:
    """Render a source-bundle pipeline manifest."""

    summary = _mapping(manifest.get("summary"))
    artifacts = _mapping(manifest.get("artifacts"))
    audit = _mapping(manifest.get("audit_bundle"))
    lines = [
        "# Generated Fault Source-Bundle Pipeline",
        "",
        str(manifest.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready: {bool(summary.get('ready', False))}",
        f"- Styles: {', '.join(_strings(summary.get('styles')))}",
        f"- Variants: {', '.join(_strings(summary.get('variants')))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        f"- Audit bundle status: `{summary.get('audit_bundle_status', 'unknown')}`",
        f"- Audit skipped steps: {int(summary.get('audit_skipped_steps', 0))}",
        f"- Audit warnings: {int(summary.get('audit_warning_count', 0))}",
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
    warnings = _sequence(audit.get("warnings"))
    if warnings:
        lines.extend(["", "## Audit Warnings", ""])
        for raw_warning in warnings:
            warning = _mapping(raw_warning)
            lines.append(
                "- "
                f"`{warning.get('warning_id', '')}`: "
                f"{warning.get('message', '')}"
            )
    return "\n".join(lines) + "\n"


def _pipeline_manifest(
    *,
    counts: Mapping[str, int],
    dataset_report: Mapping[str, Any],
    variants: Sequence[FaultGenerationVariant],
    styles: Sequence[FaultGenerationStyle],
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
    dataset_summary = _mapping(dataset_report.get("summary"))
    audit_status = str(audit_summary.get("status", "unknown"))
    return {
        "experiment": "generated_fault_source_bundle_pipeline",
        "description": (
            "Exports a source-diverse generated fault-class benchmark and runs "
            "the generated benchmark audit bundle against the produced artifacts."
        ),
        "summary": {
            "status": audit_status,
            "ready": bool(audit_summary.get("ready", False)),
            "styles": list(styles),
            "variants": [variant.name for variant in variants],
            "sources": int(dataset_summary.get("sources", 0)),
            "scored_runs": int(counts.get("scored_runs", 0)),
            "pairwise_examples": int(counts.get("pairwise_examples", 0)),
            "activation_prompts": int(counts.get("activation_prompts", 0)),
            "prompt_records": int(counts.get("prompt_records", 0)),
            "audit_bundle_status": audit_status,
            "audit_ready_steps": int(audit_summary.get("ready_steps", 0)),
            "audit_not_ready_steps": int(audit_summary.get("not_ready_steps", 0)),
            "audit_skipped_steps": int(audit_summary.get("skipped_steps", 0)),
            "audit_warning_count": int(audit_summary.get("warning_count", 0)),
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
        "dataset_report": dataset_report,
        "audit_bundle": audit_manifest,
    }


def _validate_styles(
    styles: Sequence[FaultGenerationStyle],
) -> tuple[FaultGenerationStyle, ...]:
    style_tuple = tuple(styles)
    if not style_tuple:
        raise ValueError("at least one style is required")
    duplicates = sorted(
        style for style in set(style_tuple) if style_tuple.count(style) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate styles are not allowed: {', '.join(duplicates)}")
    return style_tuple


def _source_for_style(style: FaultGenerationStyle) -> str:
    return f"generated_fault_class_{style}"


def _summary_only(report: Mapping[str, Any]) -> dict[str, Any]:
    return {"summary": dict(_mapping(report.get("summary")))}


def _write_json(value: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return [str(item) for item in _sequence(value)]
