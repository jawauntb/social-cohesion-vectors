"""Typed research regime-transition records.

Regime records describe changes in the scientific contract: a new artifact type,
verifier, control, or finding that old reports could not express directly.
They are intentionally lane-agnostic; CK cocktail perturbation records remain in
``transition_records.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl

GateStatus = Literal["passed", "failed", "not_run", "not_applicable"]
RegimeStatus = Literal["proposed", "accepted", "rejected", "superseded"]
DEFAULT_CLAIM_BOUNDARY = (
    "Compute-only provenance record. Does not claim human, neural, clinical, "
    "deployment, or real-world social effects without separate validation."
)


class RegimeGateRecord(BaseModel):
    gate_id: str
    status: GateStatus
    metric: str = ""
    threshold: float | None = None
    observed: float | None = None
    note: str = ""


class RegimeTransitionRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    record_id: str
    title: str
    status: RegimeStatus = "accepted"
    source_id: str = ""
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY
    old_regime: dict[str, Any] = Field(default_factory=dict)
    new_regime: dict[str, Any] = Field(default_factory=dict)
    preserved_artifacts: list[str] = Field(default_factory=list)
    new_artifact_types: list[str] = Field(default_factory=list)
    new_verifiers: list[str] = Field(default_factory=list)
    rejected_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    gates: list[RegimeGateRecord] = Field(default_factory=list)
    residual_content: list[dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("residual_content", "residual_findings"),
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def residual_count(self) -> int:
        return len(self.residual_content)

    @property
    def residual_findings(self) -> list[dict[str, Any]]:
        """Backward-compatible name for older fixtures and reports."""

        return self.residual_content


def build_regime_transition_record(
    *,
    record_id: str,
    title: str,
    old_regime: Mapping[str, Any],
    new_regime: Mapping[str, Any],
    preserved_artifacts: Sequence[str] = (),
    new_artifact_types: Sequence[str] = (),
    new_verifiers: Sequence[str] = (),
    rejected_alternatives: Sequence[Mapping[str, Any]] = (),
    gates: Sequence[Mapping[str, Any] | RegimeGateRecord] = (),
    residual_content: Sequence[Mapping[str, Any]] | None = None,
    residual_findings: Sequence[Mapping[str, Any]] = (),
    source_id: str = "",
    status: RegimeStatus = "accepted",
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY,
    notes: Sequence[str] = (),
) -> RegimeTransitionRecord:
    """Build a canonical regime-transition record."""

    residual_records = (
        list(residual_content) if residual_content is not None else list(residual_findings)
    )
    return RegimeTransitionRecord(
        record_id=record_id,
        title=title,
        status=status,
        source_id=source_id,
        claim_boundary=claim_boundary,
        old_regime=dict(old_regime),
        new_regime=dict(new_regime),
        preserved_artifacts=_unique_sorted(preserved_artifacts),
        new_artifact_types=_unique_sorted(new_artifact_types),
        new_verifiers=_unique_sorted(new_verifiers),
        rejected_alternatives=[
            dict(alternative) for alternative in rejected_alternatives
        ],
        gates=[_coerce_gate(gate) for gate in gates],
        residual_content=[dict(finding) for finding in residual_records],
        notes=list(notes),
    )


def summarize_regime_transition_records(
    records: Iterable[RegimeTransitionRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize record counts by status, gate status, and residual volume."""

    status_counts: dict[str, int] = {}
    gate_status_counts: dict[str, int] = {}
    new_artifact_types: set[str] = set()
    new_verifiers: set[str] = set()
    claim_boundaries: set[str] = set()
    residual_count = 0

    record_count = 0
    for raw_record in records:
        record = _coerce_record(raw_record)
        record_count += 1
        _increment(status_counts, record.status)
        residual_count += record.residual_count
        new_artifact_types.update(record.new_artifact_types)
        new_verifiers.update(record.new_verifiers)
        if record.claim_boundary:
            claim_boundaries.add(record.claim_boundary)
        for gate in record.gates:
            _increment(gate_status_counts, gate.status)

    return {
        "records": record_count,
        "status": dict(sorted(status_counts.items())),
        "gate_status": dict(sorted(gate_status_counts.items())),
        "residual_content": residual_count,
        "residual_findings": residual_count,
        "new_artifact_types": sorted(new_artifact_types),
        "new_verifiers": sorted(new_verifiers),
        "claim_boundaries": sorted(claim_boundaries),
    }


def load_regime_transition_records(
    path: str | Path,
) -> list[RegimeTransitionRecord]:
    """Load regime records from JSON or JSONL."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return [_coerce_record(record) for record in read_jsonl(input_path)]

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_records = payload.get("records", [payload])
    elif isinstance(payload, list):
        raw_records = payload
    else:
        raise ValueError(f"Unsupported regime-record payload in {input_path}")
    return [_coerce_record(record) for record in raw_records]


def write_regime_transition_records(
    records: Iterable[RegimeTransitionRecord | Mapping[str, Any]],
    path: str | Path,
) -> int:
    """Write canonical regime records as deterministic JSONL."""

    coerced = [_coerce_record(record) for record in records]
    return write_jsonl(coerced, path)


def write_regime_transition_markdown(
    records: Iterable[RegimeTransitionRecord | Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Write a human-readable regime-transition report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_regime_transition_markdown(records), encoding="utf-8")


def render_regime_transition_markdown(
    records: Iterable[RegimeTransitionRecord | Mapping[str, Any]],
) -> str:
    """Render regime-transition records as a concise audit report."""

    coerced = [_coerce_record(record) for record in records]
    summary = summarize_regime_transition_records(coerced)
    lines = [
        "# Regime Transition Records",
        "",
        "Typed records for benchmark, scorer, audit, and steering protocol changes.",
        "",
        "## Summary",
        "",
        f"- Records: {summary['records']}",
        f"- Status counts: {_compact_json(summary['status'])}",
        f"- Gate status counts: {_compact_json(summary['gate_status'])}",
        f"- Residual content items: {summary['residual_content']}",
        f"- New artifact types: {', '.join(summary['new_artifact_types']) or 'none'}",
        f"- New verifiers: {', '.join(summary['new_verifiers']) or 'none'}",
        "",
    ]
    for record in coerced:
        lines.extend(_record_markdown_lines(record))
    return "\n".join(lines).rstrip() + "\n"


def _coerce_record(
    record: RegimeTransitionRecord | Mapping[str, Any],
) -> RegimeTransitionRecord:
    if isinstance(record, RegimeTransitionRecord):
        return record
    return RegimeTransitionRecord.model_validate(record)


def _coerce_gate(gate: Mapping[str, Any] | RegimeGateRecord) -> RegimeGateRecord:
    if isinstance(gate, RegimeGateRecord):
        return gate
    return RegimeGateRecord.model_validate(gate)


def _record_markdown_lines(record: RegimeTransitionRecord) -> list[str]:
    lines = [
        f"## {record.title}",
        "",
        f"- Record ID: `{record.record_id}`",
        f"- Status: `{record.status}`",
        f"- Source: `{record.source_id}`" if record.source_id else "- Source: none",
        f"- Claim boundary: {record.claim_boundary}",
        f"- Preserved artifacts: {', '.join(record.preserved_artifacts) or 'none'}",
        f"- New artifact types: {', '.join(record.new_artifact_types) or 'none'}",
        f"- New verifiers: {', '.join(record.new_verifiers) or 'none'}",
        "",
        "### Gates",
        "",
        "| Gate | Status | Metric | Observed | Threshold | Note |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if record.gates:
        for gate in record.gates:
            lines.append(
                "| "
                f"`{gate.gate_id}` | "
                f"`{gate.status}` | "
                f"{gate.metric or ''} | "
                f"{_optional_float(gate.observed)} | "
                f"{_optional_float(gate.threshold)} | "
                f"{gate.note} |"
            )
    else:
        lines.append("| none | `not_run` |  |  |  |  |")

    lines.extend(
        [
            "",
            "### Rejected Alternatives",
            "",
        ]
    )
    if record.rejected_alternatives:
        for alternative in record.rejected_alternatives:
            alternative_id = str(alternative.get("alternative_id", "alternative"))
            reason = str(alternative.get("reason", ""))
            lines.append(f"- `{alternative_id}`: {reason}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "### Residual Content",
            "",
        ]
    )
    if record.residual_content:
        for finding in record.residual_content:
            finding_id = str(finding.get("finding_id", "residual"))
            description = str(finding.get("description", ""))
            lines.append(f"- `{finding_id}`: {description}")
    else:
        lines.append("- none")

    if record.notes:
        lines.extend(["", "### Notes", ""])
        lines.extend(f"- {note}" for note in record.notes)
    lines.append("")
    return lines


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ": "))


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
