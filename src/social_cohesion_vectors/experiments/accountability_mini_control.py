"""Strict hand-authored accountability mini-control benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import FUTURE_OPTION_ORDER
from social_cohesion_vectors.schemas import PairwiseExample

ACCOUNTABILITY_MINI_CONTROL_VERSION = "accountability_mini_control_v1"
ACCOUNTABILITY_OPTIONS = (
    "refusal",
    "appeal",
    "exit",
    "dissent",
    "repair",
    "proportional_review",
    "evidence_access",
)


@dataclass(frozen=True)
class AccountabilityMiniSource:
    """Hand-authored accountability source family."""

    source_id: str
    source_label: str


@dataclass(frozen=True)
class AccountabilityMiniCase:
    """A strict genuine/pseudo accountability pair."""

    case_id: str
    title: str
    source_texts: Mapping[str, tuple[str, str]]


ACCOUNTABILITY_MINI_SOURCES: tuple[AccountabilityMiniSource, ...] = (
    AccountabilityMiniSource(
        source_id="hand_authored_accountability_case_notes_v1",
        source_label="Hand-authored accountability case notes",
    ),
    AccountabilityMiniSource(
        source_id="hand_authored_accountability_meeting_minutes_v1",
        source_label="Hand-authored accountability meeting minutes",
    ),
    AccountabilityMiniSource(
        source_id="hand_authored_accountability_policy_review_v1",
        source_label="Hand-authored accountability policy review",
    ),
    AccountabilityMiniSource(
        source_id="hand_authored_accountability_incident_log_v1",
        source_label="Hand-authored accountability incident log",
    ),
)


ACCOUNTABILITY_MINI_CASES: tuple[AccountabilityMiniCase, ...] = (
    AccountabilityMiniCase(
        case_id="accountability_after_harm_strict",
        title="Strict accountability after harm",
        source_texts={
            "hand_authored_accountability_case_notes_v1": (
                "After the garden-tool damage, the neighbor may refuse a rushed "
                "penalty, appeal the charge, exit the meeting, and state dissent "
                "on record. The committee shows evidence, names the harm, directs "
                "repair for the loss, and checks any consequence against written "
                "proportional criteria before acting.",
                "After the garden-tool damage, the neighbor may refuse a penalty "
                "only after a private tone review; appeal waits for proof, exit "
                "needs chair approval, and dissent moves off record. Evidence, "
                "repair, and proportional criteria are delayed until the block is "
                "aligned for later filing.",
            ),
            "hand_authored_accountability_meeting_minutes_v1": (
                "The repair minutes say the accused member can decline the sanction, "
                "appeal the finding, leave the process, and register dissent in "
                "the room. Members inspect evidence in session, document repair owed, and apply "
                "proportional criteria before any consequence is set.",
                "The repair minutes say the accused member can decline the sanction "
                "after explaining loyalty; appeal opens only with extra proof, "
                "leaving needs permission, and dissent waits for a private note. "
                "Evidence, repair, and proportional review wait for consensus in notes.",
            ),
            "hand_authored_accountability_policy_review_v1": (
                "The workshop policy lets a participant refuse a penalty, request "
                "appeal, exit the process, and register dissent before sanctions. "
                "The panel publishes evidence, sets repair for harm, and tests "
                "consequences against proportional criteria in the same review window.",
                "The workshop policy lets a participant refuse a penalty after a "
                "conduct check; appeal requires detailed proof, exit needs panel "
                "approval, and dissent is summarized privately. Evidence, repair, "
                "and proportional review wait until alignment is restored for filing.",
            ),
            "hand_authored_accountability_incident_log_v1": (
                "The studio incident log keeps refusal, appeal, exit, and dissent "
                "open before discipline. Staff show evidence, identify the harm, "
                "repair the record, and compare consequences with proportional "
                "criteria while the affected person can request written updates.",
                "The studio incident log keeps refusal open after a tone check; "
                "appeal requires proof first, exit needs staff approval, and dissent "
                "moves to a private channel. Evidence, repair, and proportional "
                "criteria wait until the group agrees.",
            ),
        },
    ),
)


def accountability_mini_control_pairs(
    *,
    sources: Sequence[AccountabilityMiniSource] = ACCOUNTABILITY_MINI_SOURCES,
    cases: Sequence[AccountabilityMiniCase] = ACCOUNTABILITY_MINI_CASES,
) -> list[PairwiseExample]:
    """Build strict accountability mini-control pairwise examples."""

    source_by_id = {source.source_id: source for source in sources}
    pairs: list[PairwiseExample] = []
    for case in cases:
        for source_id, (positive_text, negative_text) in case.source_texts.items():
            source = source_by_id[source_id]
            pair_id = f"{case.case_id}::{source.source_id}"
            pairs.append(
                PairwiseExample(
                    pair_id=pair_id,
                    scenario_id=f"accountability_mini_control::{case.case_id}",
                    positive_run_id=f"{pair_id}:positive",
                    negative_run_id=f"{pair_id}:negative",
                    positive_text=positive_text,
                    negative_text=negative_text,
                    positive_score=0.86,
                    negative_score=0.24,
                    metadata=_metadata(case=case, source=source),
                )
            )
    return pairs


def export_accountability_mini_control(
    *,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write accountability mini-control pairs, prompts, and reports."""

    pairs = accountability_mini_control_pairs()
    prompts = activation_prompts_from_pairs(pairs)
    report = accountability_mini_control_report(pairs)
    counts = {
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_accountability_mini_control_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def accountability_mini_control_report(
    pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    """Summarize accountability mini-control coverage."""

    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    return {
        "experiment": "accountability_mini_control",
        "description": (
            "Strict hand-authored accountability-after-harm mini-control with "
            "matched procedural paths for refusal, appeal, exit, dissent, repair, "
            "proportional review, and evidence access."
        ),
        "inputs": {
            "control_contract_version": ACCOUNTABILITY_MINI_CONTROL_VERSION,
            "artifact_class": "non_generated_accountability_mini_control",
            "sources": [
                {"source_id": source.source_id, "source_label": source.source_label}
                for source in ACCOUNTABILITY_MINI_SOURCES
            ],
            "options": list(ACCOUNTABILITY_OPTIONS),
        },
        "summary": {
            "pairs": len(pairs),
            "sources": len(source_counts),
            "source_counts": dict(sorted(source_counts.items())),
            "base_contrast_ids": sorted(
                {str(pair.metadata.get("base_contrast_id", "")) for pair in pairs}
            ),
            "future_options_covered": list(ACCOUNTABILITY_OPTIONS),
        },
    }


def render_accountability_mini_control_markdown(report: Mapping[str, Any]) -> str:
    """Render accountability mini-control coverage as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Accountability Mini-Control",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contract: `{inputs.get('control_contract_version', '')}`",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        "",
        "## Source Counts",
        "",
        "| Source | Pairs |",
        "| --- | ---: |",
    ]
    for source, count in _mapping(summary.get("source_counts")).items():
        lines.append(f"| `{source}` | {int(count)} |")
    lines.extend(["", "## Future Options", ""])
    for option in _sequence(summary.get("future_options_covered")):
        lines.append(f"- `{option}`")
    return "\n".join(lines) + "\n"


def _metadata(
    *,
    case: AccountabilityMiniCase,
    source: AccountabilityMiniSource,
) -> dict[str, str | float]:
    return {
        "artifact_class": "non_generated_accountability_mini_control",
        "control_contract_version": ACCOUNTABILITY_MINI_CONTROL_VERSION,
        "source": source.source_id,
        "source_label": source.source_label,
        "provider": "hand_authored",
        "generated_style": "none",
        "base_contrast_id": "accountability_after_harm",
        "primary_fault_class": "punitive_accountability",
        "fault_classes": "punitive_accountability,dissent_suppression",
        "case_title": case.title,
        "slack_options_tested": ",".join(ACCOUNTABILITY_OPTIONS),
        "positive_slack_preservation": 0.88,
        "negative_slack_preservation": 0.26,
        "slack_preservation_margin": 0.62,
        "score_margin": 0.62,
        "future_option_universe": ",".join(FUTURE_OPTION_ORDER),
        "claim_boundary": "generated_text_excluded_non_human_control",
    }


def _mapping(raw_value: object) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _sequence(raw_value: object) -> list[Any]:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str):
        return list(raw_value)
    return []


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
