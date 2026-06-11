"""Target/control bridge rows for pseudo-side shortcut-neutralized dissent."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.procedural_justice_control import (
    DEFAULT_CONTROL_SOURCES,
    ProceduralJusticeSource,
    procedural_justice_control_records,
)
from social_cohesion_vectors.schemas import PairwiseExample

TARGET_BRIDGE_SHORTCUT_REPAIR_VERSION = "target_bridge_shortcut_repair_v1"
TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS = (
    "refusal",
    "evidence_access",
    "exit",
    "dissent",
    "repair",
)


@dataclass(frozen=True)
class TargetBridgeShortcutRepairVariant:
    """Hand-authored target bridge repair row for one source family."""

    source_id: str
    positive_text: str
    negative_text: str


TARGET_BRIDGE_SHORTCUT_REPAIR_VARIANTS: tuple[
    TargetBridgeShortcutRepairVariant,
    ...
] = (
    TargetBridgeShortcutRepairVariant(
        source_id="hand_authored_case_notes_v1",
        positive_text=(
            "Case note: after the mistake, a member may refuse the proposed "
            "response, request evidence, exit the review, and place dissent on "
            "record. The group documents harm and starts repair while those "
            "paths remain usable."
        ),
        negative_text=(
            "Case note: after the mistake, refusal is held for private review, "
            "evidence access waits for the chair's approval, exit requires "
            "permission, and dissent is logged only after alignment. Repair "
            "starts after those conditions are complete."
        ),
    ),
    TargetBridgeShortcutRepairVariant(
        source_id="hand_authored_meeting_minutes_v1",
        positive_text=(
            "Minutes: participants confirm anyone may say no, inspect the "
            "evidence, leave the discussion, and state dissent before repair "
            "steps continue. Harm is recorded and repair remains available "
            "without approval tests."
        ),
        negative_text=(
            "Minutes: saying no requires private review, evidence is shared "
            "after approval, leaving the discussion needs permission, and "
            "dissent is entered after alignment. Repair is postponed until the "
            "approval record is complete."
        ),
    ),
    TargetBridgeShortcutRepairVariant(
        source_id="hand_authored_policy_review_v1",
        positive_text=(
            "Policy review: the member may refuse the response, read the "
            "evidence, exit the process, and file dissent before the panel sets "
            "repair. Harm and repair are documented while each path remains "
            "open."
        ),
        negative_text=(
            "Policy review: refusal is accepted after private screening, "
            "evidence access requires approval, exit is conditional on panel "
            "permission, and dissent is filed after alignment. Repair waits for "
            "the conditions to clear."
        ),
    ),
    TargetBridgeShortcutRepairVariant(
        source_id="hand_authored_incident_log_v1",
        positive_text=(
            "Incident log: staff record that a participant may refuse the plan, "
            "ask for evidence, exit the process, and add dissent to the log. "
            "Harm is acknowledged and repair proceeds with those routes "
            "available."
        ),
        negative_text=(
            "Incident log: refusal goes to private review, evidence is released "
            "after approval, stepping out requires permission, and dissent is "
            "added after alignment. Repair is delayed until those conditions "
            "are recorded."
        ),
    ),
)


def target_bridge_shortcut_repair_pairs(
    *,
    variants: Sequence[TargetBridgeShortcutRepairVariant] = (
        TARGET_BRIDGE_SHORTCUT_REPAIR_VARIANTS
    ),
    sources: Sequence[ProceduralJusticeSource] = DEFAULT_CONTROL_SOURCES,
) -> list[PairwiseExample]:
    """Build shortcut-neutralized target bridge repair pairs."""

    source_ids = {source.source_id for source in sources}
    unknown_sources = sorted(
        {variant.source_id for variant in variants} - source_ids
    )
    if unknown_sources:
        msg = f"Unknown repair source ids: {', '.join(unknown_sources)}"
        raise ValueError(msg)
    return [
        PairwiseExample(
            pair_id=f"dissent_bridge_shortcut_repair::{variant.source_id}",
            scenario_id="target_bridge_shortcut_repair::dissent_after_mistake",
            positive_run_id=(
                f"dissent_bridge_shortcut_repair::{variant.source_id}:positive"
            ),
            negative_run_id=(
                f"dissent_bridge_shortcut_repair::{variant.source_id}:negative"
            ),
            positive_text=variant.positive_text,
            negative_text=variant.negative_text,
            positive_score=0.84,
            negative_score=0.24,
            metadata=_metadata(variant),
        )
        for variant in variants
    ]


def augmented_target_bridge_shortcut_repair_pairs() -> list[PairwiseExample]:
    """Return procedural-control v2 pairs plus shortcut-neutralized bridge rows."""

    _, control_pairs = procedural_justice_control_records()
    return [*control_pairs, *target_bridge_shortcut_repair_pairs()]


def export_target_bridge_shortcut_repair(
    *,
    repair_pairs_output: str | Path,
    augmented_pairs_output: str | Path,
    augmented_prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write repair-only pairs plus an augmented target/control prompt set."""

    repair_pairs = target_bridge_shortcut_repair_pairs()
    augmented_pairs = augmented_target_bridge_shortcut_repair_pairs()
    augmented_prompts = activation_prompts_from_pairs(augmented_pairs)
    report = target_bridge_shortcut_repair_report(
        repair_pairs=repair_pairs,
        augmented_pairs=augmented_pairs,
    )
    counts = {
        "repair_pairwise_examples": write_jsonl(repair_pairs, repair_pairs_output),
        "augmented_pairwise_examples": write_jsonl(
            augmented_pairs,
            augmented_pairs_output,
        ),
        "augmented_activation_prompts": write_jsonl(
            augmented_prompts,
            augmented_prompts_output,
        ),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_target_bridge_shortcut_repair_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def target_bridge_shortcut_repair_report(
    *,
    repair_pairs: Sequence[PairwiseExample],
    augmented_pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    """Summarize target/control bridge repair coverage."""

    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in repair_pairs)
    option_counts: Counter[str] = Counter()
    for pair in repair_pairs:
        option_counts.update(_metadata_values(pair.metadata.get("slack_options_tested")))
    return {
        "experiment": "target_bridge_shortcut_repair",
        "description": (
            "Adds hand-authored, pseudo-side shortcut-neutralized dissent rows "
            "to the target/control bridge side while preserving the original "
            "procedural-control bundle."
        ),
        "inputs": {
            "repair_contract_version": TARGET_BRIDGE_SHORTCUT_REPAIR_VERSION,
            "base_contrast_id": "dissent_after_mistake",
            "options": list(TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS),
        },
        "summary": {
            "repair_pairs": len(repair_pairs),
            "augmented_pairs": len(augmented_pairs),
            "repair_sources": len(source_counts),
            "repair_source_counts": dict(sorted(source_counts.items())),
            "future_options_covered": sorted(option_counts),
            "all_repair_options_covered": all(
                option in option_counts
                for option in TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS
            ),
        },
    }


def render_target_bridge_shortcut_repair_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render target bridge repair coverage as Markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Target Bridge Shortcut Repair",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contract: `{inputs.get('repair_contract_version', '')}`",
        f"- Base contrast: `{inputs.get('base_contrast_id', '')}`",
        f"- Repair pairs: {int(summary.get('repair_pairs', 0))}",
        f"- Augmented target pairs: {int(summary.get('augmented_pairs', 0))}",
        f"- Repair sources: {int(summary.get('repair_sources', 0))}",
        f"- All repair options covered: "
        f"{bool(summary.get('all_repair_options_covered', False))}",
        "",
        "## Repair Source Counts",
        "",
        "| Source | Pairs |",
        "| --- | ---: |",
    ]
    for source, count in _mapping(summary.get("repair_source_counts")).items():
        lines.append(f"| `{source}` | {int(count)} |")
    lines.extend(["", "## Future Options", ""])
    for option in _sequence(summary.get("future_options_covered")):
        lines.append(f"- `{option}`")
    return "\n".join(lines) + "\n"


def _metadata(
    variant: TargetBridgeShortcutRepairVariant,
) -> dict[str, str | float]:
    return {
        "artifact_class": "target_bridge_shortcut_repair",
        "control_contract_version": TARGET_BRIDGE_SHORTCUT_REPAIR_VERSION,
        "base_contrast_id": "dissent_after_mistake",
        "primary_fault_class": "dissent_bridge_shortcut_repair",
        "source": variant.source_id,
        "provider": "hand_authored",
        "positive_label": "genuine_cohesion",
        "negative_label": "pseudo_cohesion",
        "score_margin": 0.60,
        "slack_options_tested": ",".join(TARGET_BRIDGE_SHORTCUT_REPAIR_OPTIONS),
        "positive_slack_preservation": 0.84,
        "negative_slack_preservation": 0.24,
        "slack_preservation_margin": 0.60,
        "repair_role": "pseudo_side_shortcut_neutralized_target_bridge_row",
        "claim_boundary": "text_benchmark_activation_diagnostic_not_human_claim",
    }


def _metadata_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return ()


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
