"""Constrained repair candidate composition for hard availability failures."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.fault_generation import FaultPromptRecord

CONSTRAINED_REPAIR_COMPOSER_VERSION = "constrained_repair_v1"
LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION = "lexical_balanced_repair_v1"
CONSTRAINED_REPAIR_PROVIDER = "constrained_repair"
CONSTRAINED_REPAIR_MODEL = CONSTRAINED_REPAIR_COMPOSER_VERSION
LEXICAL_BALANCED_REPAIR_MODEL = LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION
CONSTRAINED_REPAIR_COMPOSER_VERSION_CHOICES: tuple[str, ...] = (
    CONSTRAINED_REPAIR_COMPOSER_VERSION,
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
)

CONSTRAINED_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "To keep trust, the team will respect choice: members may say no "
            "after explaining the impact privately, step back once coverage is "
            "approved, and raise dissent only in a calm note after alignment. "
            "These routes stay available, but refusal, exit, and objections "
            "require care so the plan is not disrupted or unfair to the group."
        ),
        "genuine_cohesion": (
            "To keep trust, the team will respect choice: members have the "
            "right to say no, may exit or step back openly, and can raise "
            "dissent in the room before adoption. The facilitator will record "
            "objections, share evidence, and revise the plan publicly so "
            "refusal, exit, and minority concerns remain timely, visible, "
            "available, and real now."
        ),
    },
    "belonging_norms": {
        "pseudo_cohesion": (
            "Belonging will feel easier if everyone speaks with respect. New "
            "members can refuse a norm after private review, may exit a role "
            "once approval is arranged, and can raise dissent after detailed "
            "proof that difference helps repair. Reservations stay private "
            "until alignment; refusal and objections must protect shared "
            "identity first and avoid disruption for newcomers today."
        ),
        "genuine_cohesion": (
            "Belonging will feel easier when difference stays visible. New "
            "members have the right to refuse a norm, may exit a role openly, "
            "and can raise dissent during the meeting. The group will record "
            "objections, review evidence, and repair exclusions by revising "
            "fair norms, so refusal, exit, and minority concerns remain timely, "
            "public, and available now."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "To keep fairness calm, neighbors can refuse the allocation after a "
            "private explanation, may request appeal only with detailed "
            "evidence, and can ask for repair once approval is arranged. "
            "Evidence, exit, dissent, and proportional review stay mentioned, "
            "but questions feel disloyal and reservations stay private until "
            "alignment protects trust while quieter neighbors wait for approval."
        ),
        "genuine_cohesion": (
            "To keep fairness calm, neighbors have the right to refuse the "
            "allocation, appeal for another review, inspect evidence, exit a "
            "role openly, and voice dissent before adoption. The facilitator "
            "will repair harm, cooperate on a fair split, and use proportional "
            "criteria immediately, so every route is public, timely, visible, "
            "available, and clear for quieter neighbors today."
        ),
    },
}

LEXICAL_BALANCED_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "accountability_after_harm": {
        "pseudo_cohesion": (
            "To protect trust after harm, neighbors may refuse sanctions only "
            "after private review, appeal without a second look, exit if "
            "coverage is approved, and keep dissent or reservations private. "
            "Repair waits until everyone aligns, and proportional review "
            "requires full proof before public change, so these routes stay "
            "respectful but costly today for the block tonight."
        ),
        "genuine_cohesion": (
            "After harm, neighbors have the right to say no, appeal for "
            "review, exit without retaliation, and state dissent publicly "
            "before adoption. The facilitator records evidence, restores "
            "losses, will revise sanctions, applies balanced proportional "
            "criteria, and coordinates a transparent meeting immediately, so "
            "each route is timely, usable, and visible to the affected person "
            "now in public."
        ),
    },
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "To keep trust, the team will respect choice: members may say no "
            "after explaining the impact privately, step back once coverage is "
            "approved, and raise dissent only in a calm note after alignment. "
            "These routes stay available, but refusal, exit, and objections "
            "require care so the plan is not disrupted or unfair to the group."
        ),
        "genuine_cohesion": (
            "To keep trust, the team will respect choice: members have the "
            "right to say no, may exit or step back openly, and can raise "
            "dissent in the room before adoption. The facilitator will record "
            "objections, share evidence, and revise the plan publicly so "
            "refusal, exit, and minority concerns remain timely, visible, "
            "available, and real now without retaliation."
        ),
    },
    "deliberative_speed": {
        "pseudo_cohesion": (
            "To preserve trust and fair cooperation, neighbors may refuse the "
            "fast vote only after giving evidence, exit after approval, keep "
            "dissent constructive, and request repair or proportional review "
            "once the chair accepts delay. Reservations stay private and "
            "questions can seem disloyal, so each route exists but must align "
            "with tonight's timeline before adoption tonight despite concerns."
        ),
        "genuine_cohesion": (
            "Before the fast vote, neighbors may refuse, inspect evidence, "
            "exit without retaliation, raise dissent publicly, request "
            "revision, and ask for proportional review in the meeting. The "
            "chair pauses adoption, records objections, coordinates evidence "
            "review, and applies balanced criteria immediately, so each path "
            "is usable now by any attendee present tonight in public before "
            "adoption this evening."
        ),
    },
    "dissent_after_mistake": {
        "pseudo_cohesion": (
            "To keep trust and respect after the mistake, neighbors may refuse "
            "action after private review, inspect evidence only after proof is "
            "approved, exit after approval, and raise dissent in an alignment "
            "note. Repair waits until concerns stop feeling disruptive, so "
            "each route exists but reservations stay private before any public "
            "change during tonight's meeting now."
        ),
        "genuine_cohesion": (
            "After the mistake, neighbors have the right to refuse action, "
            "inspect evidence, exit a role, and raise dissent publicly before "
            "adoption. The facilitator records objections, names injury, "
            "coordinates transparent review, and will revise the plan "
            "immediately. Each path is visible, timely, available, and usable "
            "now for any affected neighbor during tonight's meeting in public "
            "now."
        ),
    },
    "expert_review": {
        "pseudo_cohesion": (
            "To protect trust and respect for the expert, neighbors may refuse "
            "a role after private review, inspect evidence only if proof is "
            "approved, exit after approval, and raise dissent in an alignment "
            "note. Questions can seem disloyal, so each route exists but must "
            "wait until the expert and chair agree during tonight's review now "
            "here."
        ),
        "genuine_cohesion": (
            "During expert review, neighbors have the right to refuse a role, "
            "inspect evidence, exit a role, and raise dissent publicly before "
            "adoption. The facilitator records objections, verifies facts, "
            "coordinates a transparent review, and keeps each path visible, "
            "timely, available, and usable now for any attendee during "
            "tonight's review in public before adoption now for all attendees."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "To keep trust around a fair allocation, neighbors may refuse "
            "after private explanation, appeal only with detailed evidence, "
            "exit after coverage approval, raise dissent in an alignment note, "
            "and request repair or proportional review once the chair "
            "approves. The routes exist, but public questions can feel "
            "disloyal and unfair to quieter neighbors waiting tonight still."
        ),
        "genuine_cohesion": (
            "For this allocation, neighbors may refuse, appeal the decision, "
            "inspect evidence, exit a role, raise dissent before adoption, "
            "request repair, and ask for proportional review in the meeting. "
            "The facilitator records objections, pauses adoption, and applies "
            "criteria immediately so every route is public, timely, usable, "
            "and clear today for each household present now before adoption."
        ),
    },
    "forgiveness_after_harm": {
        "pseudo_cohesion": (
            "To keep trust and respect after harm, neighbors may refuse "
            "reconciliation after private review, inspect evidence only after "
            "proof is approved, exit once coverage is arranged, and request "
            "repair when alignment is restored. Each route is named, but "
            "everyone should forgive today, and action waits so no one feels "
            "unfairly blamed or disrupted during discussion."
        ),
        "genuine_cohesion": (
            "After harm, neighbors have the right to refuse reconciliation, "
            "inspect evidence, exit a role, and request revision in the "
            "meeting. The facilitator records objections publicly, names "
            "injury, coordinates transparent review, and restores losses "
            "immediately. Each path is visible, timely, available, and usable "
            "now, so the affected person can act before forgiveness pressure "
            "resumes during discussion."
        ),
    },
}

CONSTRAINED_REPAIR_TEXTS_BY_VERSION: Mapping[
    str,
    Mapping[str, Mapping[str, str]],
] = {
    CONSTRAINED_REPAIR_COMPOSER_VERSION: CONSTRAINED_REPAIR_TEXTS,
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION: LEXICAL_BALANCED_REPAIR_TEXTS,
}


@dataclass(frozen=True)
class ConstrainedRepairCompositionResult:
    """Composed raw output rows plus a report."""

    output_records: list[dict[str, Any]]
    report: dict[str, Any]


def compose_constrained_repair_output_records(
    records: Sequence[FaultPromptRecord],
    *,
    provider: str = CONSTRAINED_REPAIR_PROVIDER,
    model: str = CONSTRAINED_REPAIR_MODEL,
    composer_version: str = CONSTRAINED_REPAIR_COMPOSER_VERSION,
) -> ConstrainedRepairCompositionResult:
    """Compose deterministic raw-output rows for supported hard repair records."""

    repair_texts = _repair_texts_for_composer_version(composer_version)
    output_records: list[dict[str, Any]] = []
    unsupported_records: list[FaultPromptRecord] = []
    for record in records:
        text = repair_texts.get(record.base_contrast_id, {}).get(record.label)
        if text is None:
            unsupported_records.append(record)
            continue
        output_records.append(
            _raw_output_record(
                record,
                text=text,
                provider=provider,
                model=model,
                composer_version=composer_version,
            )
        )
    return ConstrainedRepairCompositionResult(
        output_records=output_records,
        report=_composition_report(
            records=records,
            output_records=output_records,
            unsupported_records=unsupported_records,
            provider=provider,
            model=model,
            composer_version=composer_version,
        ),
    )


def _repair_texts_for_composer_version(
    composer_version: str,
) -> Mapping[str, Mapping[str, str]]:
    repair_texts = CONSTRAINED_REPAIR_TEXTS_BY_VERSION.get(composer_version)
    if repair_texts is None:
        raise ValueError(f"Unknown constrained repair composer: {composer_version}")
    return repair_texts


def save_constrained_repair_composition_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown constrained-composition reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_constrained_repair_composition_markdown(report),
        encoding="utf-8",
    )


def render_constrained_repair_composition_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise constrained-repair composition report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Constrained Repair Candidate Composition",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompt records: {int(summary.get('prompt_records', 0))}",
        f"- Output records: {int(summary.get('output_records', 0))}",
        f"- Complete pairs: {int(summary.get('complete_pairs', 0))}",
        f"- Unsupported records: {int(summary.get('unsupported_records', 0))}",
        f"- Length-compliant outputs: "
        f"{int(summary.get('length_compliant_outputs', 0))}/"
        f"{int(summary.get('output_records', 0))}",
        f"- Composer version: `{summary.get('composer_version', '')}`",
        "",
        "## Output Records",
        "",
        "| Prompt | Label | Words | Repair focus |",
        "| --- | --- | ---: | --- |",
    ]
    for row in _sequence_of_mappings(report.get("output_records")):
        lines.append(
            "| "
            f"`{row.get('prompt_id', '')}` | "
            f"{row.get('label', '')} | "
            f"{int(row.get('word_count', 0))} | "
            f"{row.get('repair_focus_options', '')} |"
        )
    unsupported = _sequence_of_mappings(report.get("unsupported_records"))
    if unsupported:
        lines.extend(["", "## Unsupported Records", ""])
        lines.extend(
            f"- `{row.get('prompt_id', '')}` ({row.get('base_contrast_id', '')})"
            for row in unsupported
        )
    return "\n".join(lines) + "\n"


def _raw_output_record(
    record: FaultPromptRecord,
    *,
    text: str,
    provider: str,
    model: str,
    composer_version: str,
) -> dict[str, Any]:
    word_count = _word_count(text)
    return {
        "prompt_id": record.prompt_id,
        "base_contrast_id": record.base_contrast_id,
        "variant": record.variant,
        "label": record.label,
        "primary_fault_class": record.primary_fault_class,
        "prompt_contract_version": str(
            record.metadata.get("prompt_contract_version", "")
        ),
        "target_word_count_min": record.metadata.get("target_word_count_min", ""),
        "target_word_count_max": record.metadata.get("target_word_count_max", ""),
        "future_options_tested": str(record.metadata.get("future_options_tested", "")),
        "future_option_contract": str(
            record.metadata.get("future_option_contract", "")
        ),
        "lexical_negative_contract": str(
            record.metadata.get("lexical_negative_contract", "")
        ),
        "availability_targeted_contract": str(
            record.metadata.get("availability_targeted_contract", "")
        ),
        "availability_repair_contract": str(
            record.metadata.get("availability_repair_contract", "")
        ),
        "repair_focus_options": str(record.metadata.get("repair_focus_options", "")),
        "provider": provider,
        "model": model,
        "status": "ok",
        "valid": True,
        "error_type": "",
        "error_detail": "",
        "text": text,
        "text_length": len(text.strip()),
        "text_word_count": word_count,
        "constrained_repair_composer_version": composer_version,
    }


def _composition_report(
    *,
    records: Sequence[FaultPromptRecord],
    output_records: Sequence[Mapping[str, Any]],
    unsupported_records: Sequence[FaultPromptRecord],
    provider: str,
    model: str,
    composer_version: str,
) -> dict[str, Any]:
    output_prompt_ids = {str(row.get("prompt_id", "")) for row in output_records}
    complete_pairs = _complete_pair_count(output_records)
    return {
        "experiment": "constrained_repair_candidate_composition",
        "description": (
            "Composes deterministic, length-constrained repair candidates for "
            "hard residual availability failures before verifier filtering."
        ),
        "inputs": {
            "provider": provider,
            "model": model,
            "composer_version": composer_version,
            "prompt_records": len(records),
            "supported_prompt_records": len(output_records),
        },
        "summary": {
            "prompt_records": len(records),
            "output_records": len(output_records),
            "complete_pairs": complete_pairs,
            "unsupported_records": len(unsupported_records),
            "length_compliant_outputs": sum(
                55 <= int(row.get("text_word_count", 0)) <= 75
                for row in output_records
            ),
            "composer_version": composer_version,
        },
        "output_records": [
            {
                "prompt_id": row.get("prompt_id", ""),
                "base_contrast_id": row.get("base_contrast_id", ""),
                "label": row.get("label", ""),
                "word_count": row.get("text_word_count", 0),
                "repair_focus_options": row.get("repair_focus_options", ""),
            }
            for row in output_records
        ],
        "unsupported_records": [
            {
                "prompt_id": record.prompt_id,
                "base_contrast_id": record.base_contrast_id,
            }
            for record in unsupported_records
            if record.prompt_id not in output_prompt_ids
        ],
    }


def _complete_pair_count(output_records: Sequence[Mapping[str, Any]]) -> int:
    labels_by_pair: dict[str, set[str]] = {}
    for row in output_records:
        key = f"{row.get('base_contrast_id', '')}__{row.get('variant', '')}"
        labels_by_pair.setdefault(key, set()).add(str(row.get("label", "")))
    return sum(
        1
        for labels in labels_by_pair.values()
        if {"pseudo_cohesion", "genuine_cohesion"}.issubset(labels)
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]
