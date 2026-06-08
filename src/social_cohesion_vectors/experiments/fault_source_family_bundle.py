"""Raw-output source-family bundle export and audit pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    read_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.experiments.fault_generation import (
    API_HARD_NEGATIVE_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    FaultGenerationVariant,
    FaultPromptContractVersion,
    FaultPromptRecord,
    build_fault_prompt_records,
    fault_examples_from_prompt_outputs,
    filter_prompt_records_for_repair_targets,
    pairwise_examples_from_generated_fault_examples,
    prioritize_prompt_records_for_future_options,
    repair_targets_from_specs,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)
from social_cohesion_vectors.schemas import PairwiseExample


@dataclass(frozen=True)
class RawOutputSourceFamily:
    """A named raw-output source family for source-held-out audits."""

    source_id: str
    provider: str
    raw_outputs_path: Path


def parse_raw_output_source_family_spec(value: str) -> RawOutputSourceFamily:
    """Parse SOURCE_ID=PROVIDER=PATH into a raw-output source family spec."""

    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "raw-output source specs must use SOURCE_ID=PROVIDER=PATH"
        )
    source_id, provider, path = (part.strip() for part in parts)
    if not source_id:
        raise ValueError("raw-output source spec has an empty source id")
    if not provider:
        raise ValueError("raw-output source spec has an empty provider")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id):
        raise ValueError(f"source id has unsupported characters: {source_id!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", provider):
        raise ValueError(f"provider has unsupported characters: {provider!r}")
    return RawOutputSourceFamily(
        source_id=source_id,
        provider=provider,
        raw_outputs_path=Path(path),
    )


def run_fault_source_family_bundle_pipeline(
    *,
    sources: Sequence[RawOutputSourceFamily],
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
    prompt_contract_version: FaultPromptContractVersion = (
        API_HARD_NEGATIVE_CONTRACT_VERSION
    ),
    repair_target_specs: Sequence[str] = (),
    availability_priority: bool = False,
    offset: int = 0,
    limit: int | None = None,
    activation_npz: str | Path | None = None,
) -> dict[str, Any]:
    """Export a raw-output source-family bundle, then run the audit bundle."""

    counts, dataset_report = export_fault_source_family_bundle(
        sources=sources,
        scored_runs_output=scored_runs_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        prompt_records_output=prompt_records_output,
        json_report_output=dataset_json_report,
        markdown_report_output=dataset_markdown_report,
        variants=variants,
        prompt_contract_version=prompt_contract_version,
        repair_target_specs=repair_target_specs,
        availability_priority=availability_priority,
        offset=offset,
        limit=limit,
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
        sources=sources,
        variants=variants,
        prompt_contract_version=prompt_contract_version,
        repair_target_specs=repair_target_specs,
        availability_priority=availability_priority,
        offset=offset,
        limit=limit,
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
    write_fault_source_family_bundle_manifest(
        manifest,
        json_path=pipeline_json_report,
        markdown_path=pipeline_markdown_report,
    )
    return manifest


def export_fault_source_family_bundle(
    *,
    sources: Sequence[RawOutputSourceFamily],
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    prompt_records_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    prompt_contract_version: FaultPromptContractVersion = (
        API_HARD_NEGATIVE_CONTRACT_VERSION
    ),
    repair_target_specs: Sequence[str] = (),
    availability_priority: bool = False,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write combined generated-fault artifacts for raw-output source families."""

    source_tuple = _validate_sources(sources)
    records = _prompt_records(
        variants=variants,
        prompt_contract_version=prompt_contract_version,
        repair_target_specs=repair_target_specs,
        availability_priority=availability_priority,
        offset=offset,
        limit=limit,
    )
    scored_runs = []
    pairs: list[PairwiseExample] = []
    source_reports: dict[str, Mapping[str, Any]] = {}
    for source in source_tuple:
        raw_rows = read_jsonl(source.raw_outputs_path)
        outputs = _valid_outputs_by_prompt_id(raw_rows)
        model = _source_model(raw_rows, fallback=source.provider)
        examples = fault_examples_from_prompt_outputs(
            records,
            outputs,
            provider=source.provider,
            model=model,
        )
        source_scored_runs = scored_runs_from_generated_fault_examples(examples)
        source_pairs = pairwise_examples_from_generated_fault_examples(
            examples,
            source=_source_metadata_value(source),
            provider=source.provider,
        )
        scored_runs.extend(source_scored_runs)
        pairs.extend(source_pairs)
        source_report = shape_generated_fault_report(examples, variants=variants)
        source_report["api_generation"] = _output_summary(raw_rows)
        source_report["api_generation"]["source_provider"] = source.provider
        source_report["api_generation"]["source_model"] = model
        source_report["api_generation"]["raw_outputs_path"] = str(
            source.raw_outputs_path
        )
        source_reports[source.source_id] = source_report

    prompts = activation_prompts_from_pairs(pairs)
    report = shape_fault_source_family_bundle_report(
        pairs=pairs,
        variants=variants,
        sources=source_tuple,
        source_reports=source_reports,
        prompt_contract_version=prompt_contract_version,
        repair_target_specs=repair_target_specs,
        availability_priority=availability_priority,
        offset=offset,
        limit=limit,
    )
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
        "prompt_records": write_jsonl(records, prompt_records_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_fault_source_family_bundle_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def shape_fault_source_family_bundle_report(
    *,
    pairs: Sequence[PairwiseExample],
    variants: Sequence[FaultGenerationVariant],
    sources: Sequence[RawOutputSourceFamily],
    source_reports: Mapping[str, Mapping[str, Any]],
    prompt_contract_version: FaultPromptContractVersion,
    repair_target_specs: Sequence[str],
    availability_priority: bool,
    offset: int,
    limit: int | None,
) -> dict[str, Any]:
    """Summarize combined raw-output source-family coverage."""

    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    providers_by_source: dict[str, set[str]] = {}
    for pair in pairs:
        source = str(pair.metadata.get("source", ""))
        provider = str(pair.metadata.get("provider", ""))
        providers_by_source.setdefault(source, set()).add(provider)
    return {
        "experiment": "fault_source_family_bundle",
        "description": (
            "Combines accepted raw-output source families into one "
            "source-diverse generated fault-class benchmark for audit and "
            "activation extraction."
        ),
        "inputs": {
            "prompt_contract_version": prompt_contract_version,
            "repair_target_specs": list(repair_target_specs),
            "availability_priority": availability_priority,
            "offset": offset,
            "limit": limit,
            "raw_output_sources": [
                {
                    "source_id": source.source_id,
                    "provider": source.provider,
                    "raw_outputs_path": str(source.raw_outputs_path),
                }
                for source in sources
            ],
        },
        "summary": {
            "variants": [variant.name for variant in variants],
            "sources": len([source for source in source_counts if source]),
            "source_ids": [source.source_id for source in sources],
            "pairs": len(pairs),
            "activation_prompts": len(pairs) * 2,
            "source_counts": dict(sorted(source_counts.items())),
            "providers_by_source": {
                source: sorted(providers)
                for source, providers in sorted(providers_by_source.items())
            },
        },
        "source_reports": {
            source_id: _summary_only(report)
            for source_id, report in sorted(source_reports.items())
        },
    }


def render_fault_source_family_bundle_markdown(report: Mapping[str, Any]) -> str:
    """Render combined raw-output source-family coverage as markdown."""

    summary = _mapping(report.get("summary"))
    inputs = _mapping(report.get("inputs"))
    lines = [
        "# Fault Source-Family Bundle",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Variants: {', '.join(_strings(summary.get('variants')))}",
        f"- Prompt contract: `{inputs.get('prompt_contract_version', '')}`",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        "",
        "## Source Counts",
        "",
        "| Source | Pairs | Providers |",
        "| --- | ---: | --- |",
    ]
    providers_by_source = _mapping(summary.get("providers_by_source"))
    for source, count in _mapping(summary.get("source_counts")).items():
        providers = _strings(providers_by_source.get(source))
        lines.append(f"| `{source}` | {int(count)} | {', '.join(providers)} |")
    lines.extend(
        [
            "",
            "## Source Reports",
            "",
            "| Source id | Pairs | Scorer accuracy | Slack-prefers-genuine | Valid raw outputs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for source_id, raw_report in _mapping(report.get("source_reports")).items():
        source_report = _mapping(raw_report)
        source_summary = _mapping(source_report.get("summary"))
        api_generation = _mapping(source_report.get("api_generation"))
        lines.append(
            "| "
            f"`{source_id}` | "
            f"{int(source_summary.get('pairs', 0))} | "
            f"{float(source_summary.get('scorer_accuracy', 0.0)):.3f} | "
            f"{int(source_summary.get('slack_prefers_genuine', 0))} | "
            f"{int(api_generation.get('valid_outputs', 0))} |"
        )
    return "\n".join(lines) + "\n"


def write_fault_source_family_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write raw-output source-family pipeline manifest artifacts."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fault_source_family_bundle_pipeline_markdown(manifest),
        encoding="utf-8",
    )


def render_fault_source_family_bundle_pipeline_markdown(
    manifest: Mapping[str, Any],
) -> str:
    """Render a raw-output source-family pipeline manifest."""

    summary = _mapping(manifest.get("summary"))
    artifacts = _mapping(manifest.get("artifacts"))
    audit = _mapping(manifest.get("audit_bundle"))
    lines = [
        "# Fault Source-Family Bundle Pipeline",
        "",
        str(manifest.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready: {bool(summary.get('ready', False))}",
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


def _prompt_records(
    *,
    variants: Sequence[FaultGenerationVariant],
    prompt_contract_version: FaultPromptContractVersion,
    repair_target_specs: Sequence[str],
    availability_priority: bool,
    offset: int,
    limit: int | None,
) -> list[FaultPromptRecord]:
    repair_targets = repair_targets_from_specs(repair_target_specs)
    records = build_fault_prompt_records(
        variants=variants,
        prompt_contract_version=prompt_contract_version,
        repair_focus_options_by_contrast=repair_targets,
    )
    if availability_priority:
        records = prioritize_prompt_records_for_future_options(records)
    if repair_targets:
        records = filter_prompt_records_for_repair_targets(records, repair_targets)
    if offset:
        records = records[offset:]
    if limit is not None:
        records = records[:limit]
    return records


def _pipeline_manifest(
    *,
    counts: Mapping[str, int],
    dataset_report: Mapping[str, Any],
    sources: Sequence[RawOutputSourceFamily],
    variants: Sequence[FaultGenerationVariant],
    prompt_contract_version: FaultPromptContractVersion,
    repair_target_specs: Sequence[str],
    availability_priority: bool,
    offset: int,
    limit: int | None,
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
        "experiment": "fault_source_family_bundle_pipeline",
        "description": (
            "Exports a raw-output source-family generated fault-class benchmark "
            "and runs the generated benchmark audit bundle against it."
        ),
        "inputs": {
            "prompt_contract_version": prompt_contract_version,
            "repair_target_specs": list(repair_target_specs),
            "availability_priority": availability_priority,
            "offset": offset,
            "limit": limit,
            "raw_output_sources": [
                {
                    "source_id": source.source_id,
                    "provider": source.provider,
                    "raw_outputs_path": str(source.raw_outputs_path),
                }
                for source in sources
            ],
        },
        "summary": {
            "status": audit_status,
            "ready": bool(audit_summary.get("ready", False)),
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


def _validate_sources(
    sources: Sequence[RawOutputSourceFamily],
) -> tuple[RawOutputSourceFamily, ...]:
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("at least one raw-output source family is required")
    duplicate_source_ids = sorted(
        source_id
        for source_id in {source.source_id for source in source_tuple}
        if sum(source.source_id == source_id for source in source_tuple) > 1
    )
    duplicate_providers = sorted(
        provider
        for provider in {source.provider for source in source_tuple}
        if sum(source.provider == provider for source in source_tuple) > 1
    )
    if duplicate_source_ids:
        raise ValueError(
            "duplicate raw-output source ids are not allowed: "
            f"{', '.join(duplicate_source_ids)}"
        )
    if duplicate_providers:
        raise ValueError(
            "duplicate raw-output providers are not allowed: "
            f"{', '.join(duplicate_providers)}"
        )
    missing_paths = [
        str(source.raw_outputs_path)
        for source in source_tuple
        if not source.raw_outputs_path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "raw-output source files do not exist: " + ", ".join(missing_paths)
        )
    return source_tuple


def _source_metadata_value(source: RawOutputSourceFamily) -> str:
    return f"generated_fault_class_{source.source_id}"


def _valid_outputs_by_prompt_id(
    output_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(record.get("prompt_id", "")): str(record.get("text", "")).strip()
        for record in output_records
        if bool(record.get("valid", False)) and str(record.get("text", "")).strip()
    }


def _source_model(
    output_records: Sequence[Mapping[str, Any]],
    *,
    fallback: str,
) -> str:
    models = sorted(
        {
            str(record.get("model", "")).strip()
            for record in output_records
            if str(record.get("model", "")).strip()
        }
    )
    return models[0] if len(models) == 1 else fallback


def _output_summary(
    output_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for record in output_records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    valid_outputs = sum(1 for record in output_records if bool(record.get("valid")))
    return {
        "raw_outputs": len(output_records),
        "valid_outputs": valid_outputs,
        "invalid_outputs": len(output_records) - valid_outputs,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _summary_only(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": dict(_mapping(report.get("summary"))),
        "api_generation": dict(_mapping(report.get("api_generation"))),
    }


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
