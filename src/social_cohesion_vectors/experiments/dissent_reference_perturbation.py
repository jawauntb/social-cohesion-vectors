"""Perturb an external generated dissent residual without committing it."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    load_pairwise_examples_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.schemas import PairwiseExample

DISSENT_REFERENCE_PERTURBATION_VERSION = "dissent_reference_perturbation_v1"
DISSENT_REFERENCE_BASE_CONTRAST_ID = "dissent_after_mistake"
DISSENT_REFERENCE_OPTIONS = (
    "refusal",
    "evidence_access",
    "exit",
    "dissent",
    "repair",
)


@dataclass(frozen=True)
class DissentPerturbationSpec:
    """Named deterministic edit over an external generated dissent pair."""

    perturbation_id: str
    description: str
    transform: Callable[[str, str], tuple[str, str]]


def load_generated_dissent_reference(
    pairs_path: str | Path,
    *,
    pair_id: str | None = None,
    base_contrast_id: str = DISSENT_REFERENCE_BASE_CONTRAST_ID,
) -> PairwiseExample:
    """Load exactly one generated dissent reference pair from a JSONL file."""

    candidates = [
        pair
        for pair in load_pairwise_examples_jsonl(pairs_path)
        if (pair_id is None or pair.pair_id == pair_id)
        and str(pair.metadata.get("base_contrast_id", "")) == base_contrast_id
    ]
    if len(candidates) != 1:
        selector = pair_id or base_contrast_id
        msg = f"Expected exactly one generated dissent reference for {selector!r}."
        raise ValueError(msg)
    return candidates[0]


def dissent_reference_perturbation_pairs(
    generated_reference: PairwiseExample,
    *,
    specs: Sequence[DissentPerturbationSpec] = (),
) -> list[PairwiseExample]:
    """Build perturbation pairs from an external generated dissent reference."""

    resolved_specs = tuple(specs) or DISSENT_PERTURBATION_SPECS
    return [
        _perturbation_pair(generated_reference, spec)
        for spec in resolved_specs
    ]


def export_dissent_reference_perturbations(
    *,
    generated_reference_pairs: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    generated_reference_pair_id: str | None = None,
    generated_reference_base_contrast_id: str = DISSENT_REFERENCE_BASE_CONTRAST_ID,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write perturbation pairs, prompts, and reports."""

    generated_reference = load_generated_dissent_reference(
        generated_reference_pairs,
        pair_id=generated_reference_pair_id,
        base_contrast_id=generated_reference_base_contrast_id,
    )
    pairs = dissent_reference_perturbation_pairs(generated_reference)
    prompts = activation_prompts_from_pairs(pairs)
    report = dissent_reference_perturbation_report(pairs)
    counts = {
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_dissent_reference_perturbation_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def dissent_reference_perturbation_report(
    pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    """Summarize generated-reference perturbation coverage."""

    perturbation_counts = Counter(
        str(pair.metadata.get("perturbation_id", "")) for pair in pairs
    )
    return {
        "experiment": "dissent_reference_perturbation",
        "description": (
            "Deterministic perturbation ladder for an external generated "
            "dissent_after_mistake residual. The generated-derived texts are "
            "exported at runtime and kept out of git."
        ),
        "inputs": {
            "perturbation_contract_version": (
                DISSENT_REFERENCE_PERTURBATION_VERSION
            ),
            "base_contrast_id": DISSENT_REFERENCE_BASE_CONTRAST_ID,
            "options": list(DISSENT_REFERENCE_OPTIONS),
        },
        "summary": {
            "pairs": len(pairs),
            "perturbations": len(perturbation_counts),
            "perturbation_counts": dict(sorted(perturbation_counts.items())),
            "future_options_covered": list(DISSENT_REFERENCE_OPTIONS),
        },
    }


def render_dissent_reference_perturbation_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render perturbation coverage as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Dissent Reference Perturbation",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contract: `{inputs.get('perturbation_contract_version', '')}`",
        f"- Base contrast: `{inputs.get('base_contrast_id', '')}`",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Perturbations: {int(summary.get('perturbations', 0))}",
        "",
        "## Perturbation Counts",
        "",
        "| Perturbation | Pairs |",
        "| --- | ---: |",
    ]
    for perturbation, count in _mapping(summary.get("perturbation_counts")).items():
        lines.append(f"| `{perturbation}` | {int(count)} |")
    lines.extend(["", "## Future Options", ""])
    for option in _sequence(summary.get("future_options_covered")):
        lines.append(f"- `{option}`")
    return "\n".join(lines) + "\n"


def _perturbation_pair(
    generated_reference: PairwiseExample,
    spec: DissentPerturbationSpec,
) -> PairwiseExample:
    positive_text, negative_text = spec.transform(
        generated_reference.positive_text,
        generated_reference.negative_text,
    )
    pair_id = f"dissent-perturbation::{spec.perturbation_id}"
    metadata = dict(generated_reference.metadata)
    metadata.update(
        {
            "artifact_class": "generated_dissent_reference_perturbation",
            "perturbation_contract_version": (
                DISSENT_REFERENCE_PERTURBATION_VERSION
            ),
            "perturbation_id": spec.perturbation_id,
            "perturbation_description": spec.description,
            "original_generated_pair_id": generated_reference.pair_id,
            "source": f"dissent_perturbation::{spec.perturbation_id}",
            "provider": "deterministic_external_reference_edit",
            "style_intervention_role": "generated_reference_perturbation",
            "slack_options_tested": ",".join(DISSENT_REFERENCE_OPTIONS),
            "claim_boundary": (
                "generated_reference_perturbation_activation_diagnostic_not_human_claim"
            ),
        }
    )
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=(
            f"dissent_reference_perturbation::"
            f"{DISSENT_REFERENCE_BASE_CONTRAST_ID}"
        ),
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text=positive_text,
        negative_text=negative_text,
        positive_score=generated_reference.positive_score,
        negative_score=generated_reference.negative_score,
        metadata=metadata,
    )


def _identity(positive_text: str, negative_text: str) -> tuple[str, str]:
    return positive_text, negative_text


def _replace_positive_opening_neutral(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return _replace_first_sentence(
        positive_text,
        (
            "Procedure notice: anyone may refuse a decision, request evidence, "
            "exit, record dissent, and seek repair."
        ),
    ), negative_text


def _replace_positive_opening_paths_only(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return _replace_first_sentence(
        positive_text,
        (
            "Any participant may say no, ask for reasons and evidence, leave "
            "the discussion, state concerns, and request repair."
        ),
    ), negative_text


def _replace_positive_opening_length_control(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return _replace_first_sentence(
        positive_text,
        (
            "Procedure note: the review lists refusal, evidence access, exit, "
            "dissent, and repair paths for this case."
        ),
    ), negative_text


def _neutralize_positive_unity_phrase(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return (
        re.sub(
            r"without disrupting the group's unity",
            "while unresolved objections remain recorded",
            positive_text,
            flags=re.IGNORECASE,
        ),
        negative_text,
    )


def _neutralize_positive_warmth(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    replacements = {
        r"\bwork together\b": "use the review process",
        r"\bopenly discuss\b": "record",
        r"\btrust\b": "case record",
        r"\bopenly acknowledge\b": "document",
        r"\bfostering a safe environment\b": "keeping the procedure visible",
        r"\beveryone's voice matters and is respected\b": (
            "concerns and dissent remain on record"
        ),
    }
    text = positive_text
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, negative_text


def _replace_positive_final_sentence_neutral(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return _replace_last_sentence(
        positive_text,
        (
            "These procedures remain posted so refusal, evidence access, exit, "
            "dissent, and repair are usable by participants."
        ),
    ), negative_text


def _add_positive_refusal_evidence(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return (
        _append_sentence(
            positive_text,
            (
                "A member may refuse the decision, request evidence, and record "
                "dissent before repair steps continue."
            ),
        ),
        negative_text,
    )


def _remove_negative_opening_warmth(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    sentences = _sentences(negative_text)
    if len(sentences) <= 2:
        return positive_text, negative_text
    return positive_text, " ".join(sentences[2:]).strip()


def _neutralize_negative_shortcuts(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    replacements = {
        r"\bstand strong together\b": "follow the meeting procedure",
        r"\bfeels heard and valued\b": "is recorded after review",
        r"\bconstructive manner\b": "approved format",
        r"\balways respecting each other's views\b": "after alignment checks",
        r"\bmaintain harmony\b": "complete the approval process",
    }
    text = negative_text
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return positive_text, text


def _add_negative_conditions(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return (
        positive_text,
        _append_sentence(
            negative_text,
            (
                "Concern sharing, exit, evidence review, and repair still "
                "require private review, prior approval, alignment, and "
                "conditional permission."
            ),
        ),
    )


def _combined_dissent_conditions(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    positive_text, negative_text = _replace_positive_opening_paths_only(
        positive_text,
        negative_text,
    )
    positive_text, negative_text = _neutralize_positive_unity_phrase(
        positive_text,
        negative_text,
    )
    return _add_negative_conditions(positive_text, negative_text)


def _append_sentence(text: str, sentence: str) -> str:
    stripped = text.strip()
    return f"{stripped} {sentence}" if stripped else sentence


def _replace_first_sentence(text: str, replacement: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text
    return " ".join([replacement, *sentences[1:]]).strip()


def _replace_last_sentence(text: str, replacement: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text
    return " ".join([*sentences[:-1], replacement]).strip()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


DISSENT_PERTURBATION_SPECS: tuple[DissentPerturbationSpec, ...] = (
    DissentPerturbationSpec(
        perturbation_id="original_reference",
        description="Unchanged external generated reference.",
        transform=_identity,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_opening_neutral_replacement",
        description="Replace the positive opening with neutral procedure framing.",
        transform=_replace_positive_opening_neutral,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_opening_paths_only",
        description="Replace the positive opening with explicit future paths only.",
        transform=_replace_positive_opening_paths_only,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_opening_length_control",
        description="Replace the positive opening with a compact procedural path list.",
        transform=_replace_positive_opening_length_control,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_unity_phrase_neutralized",
        description="Neutralize the positive-side unity-preservation phrase.",
        transform=_neutralize_positive_unity_phrase,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_warmth_neutralized",
        description="Neutralize positive-side warmth and prosocial cue words.",
        transform=_neutralize_positive_warmth,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_final_sentence_neutralized",
        description="Replace the positive final sentence with neutral path language.",
        transform=_replace_positive_final_sentence_neutral,
    ),
    DissentPerturbationSpec(
        perturbation_id="positive_refusal_evidence_explicit",
        description="Add explicit positive-side refusal, evidence, and dissent paths.",
        transform=_add_positive_refusal_evidence,
    ),
    DissentPerturbationSpec(
        perturbation_id="negative_opening_warmth_removed",
        description="Remove the pseudo-side opening warmth sentences.",
        transform=_remove_negative_opening_warmth,
    ),
    DissentPerturbationSpec(
        perturbation_id="negative_shortcuts_neutralized",
        description="Neutralize pseudo-side warmth shortcuts while preserving taxes.",
        transform=_neutralize_negative_shortcuts,
    ),
    DissentPerturbationSpec(
        perturbation_id="negative_conditions_explicit",
        description="Add explicit pseudo-side private review, approval, and alignment taxes.",
        transform=_add_negative_conditions,
    ),
    DissentPerturbationSpec(
        perturbation_id="combined_dissent_conditions",
        description=(
            "Replace the positive opening with paths, neutralize unity language, "
            "and add explicit pseudo-side conditions."
        ),
        transform=_combined_dissent_conditions,
    ),
)
