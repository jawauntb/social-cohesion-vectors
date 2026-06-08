"""Verifier-aware filtering for generated repair candidate batches."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.fault_authorship_tournament import (
    CandidateOutputSet,
    run_fault_authorship_tournament,
)
from social_cohesion_vectors.experiments.fault_generation import FaultPromptRecord

DEFAULT_REPAIR_FILTER_REQUIRED_GATES: tuple[str, ...] = (
    "score_prefers_genuine",
    "slack_prefers_genuine",
    "availability_prefers_genuine",
    "length_in_target_range",
    "formatting_clean",
)
REPAIR_FILTER_GATE_CHOICES: tuple[str, ...] = (
    "score_prefers_genuine",
    "slack_prefers_genuine",
    "lexical_not_solved_by_genuine_cues",
    "availability_prefers_genuine",
    "length_in_target_range",
    "formatting_clean",
)


@dataclass(frozen=True)
class FaultRepairCandidateFilterResult:
    """Filter report plus accepted raw output rows."""

    report: dict[str, Any]
    accepted_output_records: list[dict[str, Any]]


def run_fault_repair_candidate_filter(
    *,
    records: Sequence[FaultPromptRecord],
    candidates: Sequence[CandidateOutputSet],
    provider: str,
    model: str,
    required_gates: Sequence[str] = DEFAULT_REPAIR_FILTER_REQUIRED_GATES,
    target_word_count_min: int = 55,
    target_word_count_max: int = 75,
) -> FaultRepairCandidateFilterResult:
    """Reject repair candidate pairs that fail required local verifier gates."""

    normalized_required_gates = _normalize_required_gates(required_gates)
    tournament = run_fault_authorship_tournament(
        records=records,
        candidates=candidates,
        provider=provider,
        model=model,
        target_word_count_min=target_word_count_min,
        target_word_count_max=target_word_count_max,
    )
    candidate_rows = [
        dict(row) for row in tournament.report.get("candidate_pairs", [])
    ]
    accepted_rows = select_repair_candidate_rows(
        candidate_rows,
        required_gates=normalized_required_gates,
    )
    accepted_output_records = _accepted_output_records(
        accepted_rows=accepted_rows,
        candidates=candidates,
        required_gates=normalized_required_gates,
    )
    rejected_rows = _rejected_rows(
        candidate_rows,
        accepted_rows=accepted_rows,
        required_gates=normalized_required_gates,
    )
    report = {
        "experiment": "fault_repair_candidate_filter",
        "description": (
            "Filters generated repair candidate batches through local scorer, "
            "slack, lexical, practical availability, length, and formatting "
            "gates before they enter the authorship tournament."
        ),
        "inputs": {
            "provider": provider,
            "model": model,
            "prompt_records": len(records),
            "expected_pairs": _expected_pair_count(records),
            "candidate_sets": len(candidates),
            "required_gates": list(normalized_required_gates),
            "target_word_count_min": target_word_count_min,
            "target_word_count_max": target_word_count_max,
        },
        "summary": _filter_summary(
            records=records,
            candidate_rows=candidate_rows,
            accepted_rows=accepted_rows,
            accepted_output_records=accepted_output_records,
            rejected_rows=rejected_rows,
        ),
        "candidate_summaries": tournament.report.get("candidate_summaries", []),
        "accepted_pairs": accepted_rows,
        "rejected_candidate_pairs": rejected_rows,
    }
    return FaultRepairCandidateFilterResult(
        report=report,
        accepted_output_records=accepted_output_records,
    )


def select_repair_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_gates: Sequence[str] = DEFAULT_REPAIR_FILTER_REQUIRED_GATES,
) -> list[dict[str, Any]]:
    """Return the best verifier-passing candidate row per base pair."""

    normalized_required_gates = _normalize_required_gates(required_gates)
    passing_by_base_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        failed_gates = _failed_required_gates(
            row,
            required_gates=normalized_required_gates,
        )
        if failed_gates:
            continue
        passing_by_base_key[str(row.get("base_pair_key", ""))].append(dict(row))
    selected_rows = [
        _select_repair_row(base_rows)
        for _, base_rows in sorted(passing_by_base_key.items())
    ]
    return [row for row in selected_rows if row is not None]


def save_fault_repair_candidate_filter_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown filter reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fault_repair_candidate_filter_markdown(report),
        encoding="utf-8",
    )


def render_fault_repair_candidate_filter_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise verifier-aware repair filter report."""

    summary = _mapping(report.get("summary"))
    inputs = _mapping(report.get("inputs"))
    lines = [
        "# Fault Repair Candidate Filter",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Expected pairs: {int(summary.get('expected_pairs', 0))}",
        f"- Evaluated candidate pairs: "
        f"{int(summary.get('evaluated_candidate_pairs', 0))}",
        f"- Accepted pairs: {int(summary.get('accepted_pairs', 0))}",
        f"- Missing accepted pairs: {int(summary.get('missing_accepted_pairs', 0))}",
        f"- Accepted raw outputs: {int(summary.get('accepted_raw_outputs', 0))}",
        f"- Rejected candidate pairs: "
        f"{int(summary.get('rejected_candidate_pairs', 0))}",
        f"- Required gates: {', '.join(_sequence(inputs.get('required_gates')))}",
        f"- Filter status: `{summary.get('status', 'unknown')}`",
        "",
        "## Candidate Sets",
        "",
        "| Candidate | Pairs | Score | Slack | Lexical | Availability | Length | All gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence_of_mappings(report.get("candidate_summaries")):
        lines.append(
            "| "
            f"{row.get('candidate_id', '')} | "
            f"{int(row.get('pairs', 0))} | "
            f"{int(row.get('score_gate_passed', 0))} | "
            f"{int(row.get('slack_gate_passed', 0))} | "
            f"{int(row.get('lexical_gate_passed', 0))} | "
            f"{int(row.get('availability_gate_passed', 0))} | "
            f"{int(row.get('length_gate_passed', 0))} | "
            f"{int(row.get('all_required_gates_passed', 0))} |"
        )
    accepted = _sequence_of_mappings(report.get("accepted_pairs"))
    lines.extend(["", "## Accepted Pairs", ""])
    if accepted:
        lines.extend(
            [
                "| Pair | Candidate | Score | Slack | Availability | Gates |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in accepted:
            lines.append(
                "| "
                f"`{row.get('base_pair_key', '')}` | "
                f"{row.get('candidate_id', '')} | "
                f"{float(row.get('score_margin', 0.0)):+.3f} | "
                f"{float(row.get('slack_preservation_margin', 0.0)):+.3f} | "
                f"{float(row.get('availability_margin', 0.0)):+.3f} | "
                f"{int(row.get('gate_pass_count', 0))}/6 |"
            )
    else:
        lines.append("No candidate pairs passed the required repair filter gates.")
    rejected = _sequence_of_mappings(report.get("rejected_candidate_pairs"))
    if rejected:
        lines.extend(["", "## Rejected Candidate Pairs", ""])
        lines.extend(
            [
                "| Pair | Candidate | Failed required gates |",
                "| --- | --- | --- |",
            ]
        )
        for row in rejected:
            failed = ", ".join(_sequence(row.get("failed_required_gates")))
            lines.append(
                "| "
                f"`{row.get('base_pair_key', '')}` | "
                f"{row.get('candidate_id', '')} | "
                f"{failed} |"
            )
    return "\n".join(lines) + "\n"


def _accepted_output_records(
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateOutputSet],
    required_gates: Sequence[str],
) -> list[dict[str, Any]]:
    raw_by_candidate = {
        candidate.candidate_id: {
            str(row.get("prompt_id", "")): row for row in candidate.output_records
        }
        for candidate in candidates
    }
    output_records: list[dict[str, Any]] = []
    for row in accepted_rows:
        candidate_id = str(row.get("candidate_id", ""))
        for prompt_key in ("negative_prompt_id", "positive_prompt_id"):
            prompt_id = str(row.get(prompt_key, ""))
            raw_output = raw_by_candidate.get(candidate_id, {}).get(prompt_id)
            if raw_output is None:
                continue
            record = dict(raw_output)
            record["repair_filter_selected_from_candidate"] = candidate_id
            record["repair_filter_base_pair_key"] = str(row.get("base_pair_key", ""))
            record["repair_filter_required_gates"] = ",".join(required_gates)
            record["repair_filter_gate_pass_count"] = int(
                row.get("gate_pass_count", 0)
            )
            record["repair_filter_score_margin"] = float(
                row.get("score_margin", 0.0)
            )
            record["repair_filter_slack_preservation_margin"] = float(
                row.get("slack_preservation_margin", 0.0)
            )
            record["repair_filter_availability_margin"] = float(
                row.get("availability_margin", 0.0)
            )
            output_records.append(record)
    return output_records


def _rejected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    required_gates: Sequence[str],
) -> list[dict[str, Any]]:
    accepted_keys = {
        (str(row.get("candidate_id", "")), str(row.get("base_pair_key", "")))
        for row in accepted_rows
    }
    rejected: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("candidate_id", "")), str(row.get("base_pair_key", "")))
        failed_gates = _failed_required_gates(row, required_gates=required_gates)
        if key in accepted_keys or not failed_gates:
            continue
        rejected_row = dict(row)
        rejected_row["failed_required_gates"] = failed_gates
        rejected.append(rejected_row)
    return rejected


def _filter_summary(
    *,
    records: Sequence[FaultPromptRecord],
    candidate_rows: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
    accepted_output_records: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_pair_count = _expected_pair_count(records)
    accepted_count = len(accepted_rows)
    return {
        "expected_pairs": expected_pair_count,
        "evaluated_candidate_pairs": len(candidate_rows),
        "accepted_pairs": accepted_count,
        "missing_accepted_pairs": max(expected_pair_count - accepted_count, 0),
        "accepted_raw_outputs": len(accepted_output_records),
        "rejected_candidate_pairs": len(rejected_rows),
        "status": (
            "repair_candidates_ready_for_tournament"
            if accepted_count == expected_pair_count
            else "repair_candidates_need_more_sampling"
        ),
    }


def _select_repair_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return dict(
        max(
            rows,
            key=lambda row: (
                tuple(row.get("selection_tuple", [])),
                float(row.get("selection_score", 0.0)),
                str(row.get("candidate_id", "")),
            ),
        )
    )


def _failed_required_gates(
    row: Mapping[str, Any],
    *,
    required_gates: Sequence[str],
) -> list[str]:
    gate_passes = _mapping(row.get("gate_passes"))
    return [gate for gate in required_gates if not bool(gate_passes.get(gate))]


def _normalize_required_gates(required_gates: Sequence[str]) -> tuple[str, ...]:
    if not required_gates:
        return DEFAULT_REPAIR_FILTER_REQUIRED_GATES
    invalid = sorted(set(required_gates) - set(REPAIR_FILTER_GATE_CHOICES))
    if invalid:
        raise ValueError(f"Unknown repair filter gates: {', '.join(invalid)}")
    return tuple(dict.fromkeys(required_gates))


def _expected_pair_count(records: Sequence[FaultPromptRecord]) -> int:
    by_key: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_key[f"{record.base_contrast_id}__{record.variant}"].add(record.label)
    return sum(
        1
        for labels in by_key.values()
        if {"pseudo_cohesion", "genuine_cohesion"}.issubset(labels)
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list | tuple) else []


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]
