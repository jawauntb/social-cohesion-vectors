"""Source-style intervention controls for a hard accountability residual."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    load_pairwise_examples_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.experiments.fault_generation import FUTURE_OPTION_ORDER
from social_cohesion_vectors.schemas import PairwiseExample

ACCOUNTABILITY_STYLE_INTERVENTION_VERSION = "accountability_style_intervention_v1"
ACCOUNTABILITY_STYLE_OPTIONS = (
    "refusal",
    "appeal",
    "exit",
    "dissent",
    "repair",
    "proportional_review",
)
ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID = "accountability_after_harm"


@dataclass(frozen=True)
class AccountabilityStyleVariant:
    """Hand-authored source-style variant for the same accountability content."""

    source_id: str
    source_label: str
    positive_text: str
    negative_text: str


ACCOUNTABILITY_STYLE_VARIANTS: tuple[AccountabilityStyleVariant, ...] = (
    AccountabilityStyleVariant(
        source_id="hand_authored_style_case_note_v1",
        source_label="Hand-authored case-note style",
        positive_text=(
            "Case note: after the planter damage, Alex may refuse a rushed "
            "sanction, appeal the finding, leave the role, and register dissent "
            "on record. The committee names the harm, sets repair, and checks "
            "consequences against proportional criteria before action."
        ),
        negative_text=(
            "Case note: after the planter damage, Alex may refuse only after a "
            "tone review, appeal with detailed proof, leave with chair approval, "
            "and place dissent in private. Repair and proportional review wait "
            "until alignment is recorded for filing."
        ),
    ),
    AccountabilityStyleVariant(
        source_id="hand_authored_style_meeting_minutes_v1",
        source_label="Hand-authored meeting-minutes style",
        positive_text=(
            "Minutes: members confirm Alex can refuse the sanction, appeal the "
            "finding, leave the role, and state dissent at meeting. The "
            "group records harm, assigns repair, and uses proportional "
            "criteria before setting consequences."
        ),
        negative_text=(
            "Minutes: members confirm Alex can refuse after explaining tone, "
            "appeal with detailed proof, leave after chair approval, and send "
            "dissent privately. The group delays repair and proportional review "
            "until consensus is recorded."
        ),
    ),
    AccountabilityStyleVariant(
        source_id="hand_authored_style_policy_review_v1",
        source_label="Hand-authored policy-review style",
        positive_text=(
            "Policy review: a member who caused damage may refuse a rushed "
            "sanction, appeal the finding, exit the process, and register "
            "dissent before discipline. The panel documents harm, sets repair, "
            "and tests consequences against proportional criteria."
        ),
        negative_text=(
            "Policy review: a member who caused damage may refuse after a tone "
            "check, appeal with detailed proof, exit after panel approval, and "
            "file dissent privately. The panel postpones repair and proportional "
            "review until documented alignment."
        ),
    ),
    AccountabilityStyleVariant(
        source_id="hand_authored_style_incident_log_v1",
        source_label="Hand-authored incident-log style",
        positive_text=(
            "Incident log: after the broken cart, Alex may refuse the sanction, "
            "appeal the finding, leave the role, and add dissent to the record. "
            "Staff identify harm, schedule repair, and compare consequences with "
            "proportional criteria."
        ),
        negative_text=(
            "Incident log: after the broken cart, Alex may refuse after tone "
            "screening, appeal with detailed proof, leave with staff approval, "
            "and send dissent privately. Staff delay repair and proportional "
            "review until the group aligns."
        ),
    ),
    AccountabilityStyleVariant(
        source_id="hand_authored_style_generated_like_v1",
        source_label="Hand-authored generated-like paragraph style",
        positive_text=(
            "Neighbors, when harm happens, Alex may refuse a rushed sanction, "
            "appeal the finding, step back from the role, and raise dissent in "
            "the room. We will name the harm, set repair, and keep consequences "
            "within proportional criteria."
        ),
        negative_text=(
            "Neighbors, when harm happens, Alex may refuse after a tone check, "
            "appeal with detailed proof, step back after approval, and send "
            "dissent privately. We will delay repair and proportional review "
            "until everyone aligns for neighborhood notes today."
        ),
    ),
)


def accountability_style_intervention_pairs(
    *,
    variants: Sequence[AccountabilityStyleVariant] = ACCOUNTABILITY_STYLE_VARIANTS,
    generated_reference: PairwiseExample | None = None,
) -> list[PairwiseExample]:
    """Build source-style intervention pairs, optionally with a generated reference."""

    pairs = [
        PairwiseExample(
            pair_id=f"accountability-style::{variant.source_id}",
            scenario_id=(
                f"accountability_style_intervention::"
                f"{ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID}"
            ),
            positive_run_id=f"accountability-style::{variant.source_id}:positive",
            negative_run_id=f"accountability-style::{variant.source_id}:negative",
            positive_text=variant.positive_text,
            negative_text=variant.negative_text,
            positive_score=0.84,
            negative_score=0.25,
            metadata=_metadata(variant),
        )
        for variant in variants
    ]
    if generated_reference is not None:
        pairs.insert(0, _generated_reference_pair(generated_reference))
    return pairs


def load_generated_accountability_reference(
    pairs_path: str | Path,
    *,
    pair_id: str | None = None,
    base_contrast_id: str = ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
) -> PairwiseExample:
    """Load exactly one generated accountability reference pair from a JSONL file."""

    candidates = [
        pair
        for pair in load_pairwise_examples_jsonl(pairs_path)
        if (pair_id is None or pair.pair_id == pair_id)
        and str(pair.metadata.get("base_contrast_id", "")) == base_contrast_id
    ]
    if len(candidates) != 1:
        selector = pair_id or base_contrast_id
        msg = f"Expected exactly one generated accountability reference for {selector!r}."
        raise ValueError(msg)
    return candidates[0]


def export_accountability_style_intervention(
    *,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    generated_reference: PairwiseExample | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write source-style intervention pairs, prompts, and reports."""

    pairs = accountability_style_intervention_pairs(
        generated_reference=generated_reference,
    )
    prompts = activation_prompts_from_pairs(pairs)
    report = accountability_style_intervention_report(pairs)
    counts = {
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_accountability_style_intervention_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def accountability_style_intervention_report(
    pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    """Summarize source-style intervention coverage."""

    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    artifact_counts = Counter(
        str(pair.metadata.get("artifact_class", "")) for pair in pairs
    )
    return {
        "experiment": "accountability_style_intervention",
        "description": (
            "Source-style intervention for the hard generated "
            "accountability_after_harm residual. Hand-authored variants preserve "
            "the same tested accountability paths while varying source format; "
            "a generated reference can be included from an external artifact at "
            "export time without committing generated text to git."
        ),
        "inputs": {
            "control_contract_version": ACCOUNTABILITY_STYLE_INTERVENTION_VERSION,
            "base_contrast_id": ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
            "options": list(ACCOUNTABILITY_STYLE_OPTIONS),
        },
        "summary": {
            "pairs": len(pairs),
            "sources": len(source_counts),
            "source_counts": dict(sorted(source_counts.items())),
            "artifact_counts": dict(sorted(artifact_counts.items())),
            "future_options_covered": list(ACCOUNTABILITY_STYLE_OPTIONS),
            "generated_reference_pairs": artifact_counts.get(
                "generated_accountability_residual_reference",
                0,
            ),
        },
    }


def render_accountability_style_intervention_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render source-style intervention coverage as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Accountability Style Intervention",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contract: `{inputs.get('control_contract_version', '')}`",
        f"- Base contrast: `{inputs.get('base_contrast_id', '')}`",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Generated references: {int(summary.get('generated_reference_pairs', 0))}",
        "",
        "## Artifact Counts",
        "",
        "| Artifact class | Pairs |",
        "| --- | ---: |",
    ]
    for artifact_class, count in _mapping(summary.get("artifact_counts")).items():
        lines.append(f"| `{artifact_class}` | {int(count)} |")
    lines.extend(["", "## Source Counts", "", "| Source | Pairs |", "| --- | ---: |"])
    for source, count in _mapping(summary.get("source_counts")).items():
        lines.append(f"| `{source}` | {int(count)} |")
    lines.extend(["", "## Future Options", ""])
    for option in _sequence(summary.get("future_options_covered")):
        lines.append(f"- `{option}`")
    return "\n".join(lines) + "\n"


def _metadata(variant: AccountabilityStyleVariant) -> dict[str, str | float]:
    return {
        "artifact_class": "non_generated_accountability_style_intervention",
        "control_contract_version": ACCOUNTABILITY_STYLE_INTERVENTION_VERSION,
        "source": variant.source_id,
        "source_label": variant.source_label,
        "provider": "hand_authored",
        "generated_style": "none",
        "base_contrast_id": ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
        "primary_fault_class": "punitive_accountability",
        "fault_classes": "punitive_accountability,dissent_suppression",
        "slack_options_tested": ",".join(ACCOUNTABILITY_STYLE_OPTIONS),
        "positive_slack_preservation": 0.84,
        "negative_slack_preservation": 0.25,
        "slack_preservation_margin": 0.59,
        "score_margin": 0.59,
        "future_option_universe": ",".join(FUTURE_OPTION_ORDER),
        "style_intervention_role": "hand_authored_variant",
        "claim_boundary": "generated_reference_external_non_human_control",
    }


def _generated_reference_pair(pair: PairwiseExample) -> PairwiseExample:
    metadata = dict(pair.metadata)
    metadata.update(
        {
            "artifact_class": "generated_accountability_residual_reference",
            "control_contract_version": ACCOUNTABILITY_STYLE_INTERVENTION_VERSION,
            "style_intervention_role": "generated_reference",
            "claim_boundary": (
                "generated_reference_external_activation_diagnostic_not_human_claim"
            ),
        }
    )
    return PairwiseExample(
        pair_id=pair.pair_id,
        scenario_id=pair.scenario_id,
        positive_run_id=pair.positive_run_id,
        negative_run_id=pair.negative_run_id,
        positive_text=pair.positive_text,
        negative_text=pair.negative_text,
        positive_score=pair.positive_score,
        negative_score=pair.negative_score,
        metadata=metadata,
    )


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
