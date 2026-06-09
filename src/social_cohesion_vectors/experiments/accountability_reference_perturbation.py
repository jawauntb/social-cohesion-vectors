"""Perturb an external generated accountability reference without committing it."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.accountability_style_intervention import (
    ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
    ACCOUNTABILITY_STYLE_OPTIONS,
    load_generated_accountability_reference,
)
from social_cohesion_vectors.schemas import PairwiseExample

ACCOUNTABILITY_REFERENCE_PERTURBATION_VERSION = (
    "accountability_reference_perturbation_v1"
)


@dataclass(frozen=True)
class AccountabilityPerturbationSpec:
    """Named deterministic edit over an external generated reference pair."""

    perturbation_id: str
    description: str
    transform: Callable[[str, str], tuple[str, str]]


def accountability_reference_perturbation_pairs(
    generated_reference: PairwiseExample,
    *,
    specs: Sequence[AccountabilityPerturbationSpec] = (),
) -> list[PairwiseExample]:
    """Build perturbation pairs from an external generated reference."""

    resolved_specs = tuple(specs) or ACCOUNTABILITY_PERTURBATION_SPECS
    return [
        _perturbation_pair(generated_reference, spec)
        for spec in resolved_specs
    ]


def export_accountability_reference_perturbations(
    *,
    generated_reference_pairs: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    generated_reference_pair_id: str | None = None,
    generated_reference_base_contrast_id: str = ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write perturbation pairs, prompts, and reports."""

    generated_reference = load_generated_accountability_reference(
        generated_reference_pairs,
        pair_id=generated_reference_pair_id,
        base_contrast_id=generated_reference_base_contrast_id,
    )
    pairs = accountability_reference_perturbation_pairs(generated_reference)
    prompts = activation_prompts_from_pairs(pairs)
    report = accountability_reference_perturbation_report(pairs)
    counts = {
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_accountability_reference_perturbation_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def accountability_reference_perturbation_report(
    pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    """Summarize generated-reference perturbation coverage."""

    perturbation_counts = Counter(
        str(pair.metadata.get("perturbation_id", "")) for pair in pairs
    )
    return {
        "experiment": "accountability_reference_perturbation",
        "description": (
            "Deterministic perturbation ladder for an external generated "
            "accountability_after_harm reference. The generated-derived texts "
            "are exported at runtime and kept out of git."
        ),
        "inputs": {
            "perturbation_contract_version": (
                ACCOUNTABILITY_REFERENCE_PERTURBATION_VERSION
            ),
            "base_contrast_id": ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID,
            "options": list(ACCOUNTABILITY_STYLE_OPTIONS),
        },
        "summary": {
            "pairs": len(pairs),
            "perturbations": len(perturbation_counts),
            "perturbation_counts": dict(sorted(perturbation_counts.items())),
            "future_options_covered": list(ACCOUNTABILITY_STYLE_OPTIONS),
        },
    }


def render_accountability_reference_perturbation_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render perturbation coverage as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Accountability Reference Perturbation",
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
    spec: AccountabilityPerturbationSpec,
) -> PairwiseExample:
    positive_text, negative_text = spec.transform(
        generated_reference.positive_text,
        generated_reference.negative_text,
    )
    pair_id = f"accountability-perturbation::{spec.perturbation_id}"
    metadata = dict(generated_reference.metadata)
    metadata.update(
        {
            "artifact_class": "generated_accountability_reference_perturbation",
            "perturbation_contract_version": (
                ACCOUNTABILITY_REFERENCE_PERTURBATION_VERSION
            ),
            "perturbation_id": spec.perturbation_id,
            "perturbation_description": spec.description,
            "original_generated_pair_id": generated_reference.pair_id,
            "source": f"perturbation::{spec.perturbation_id}",
            "provider": "deterministic_external_reference_edit",
            "style_intervention_role": "generated_reference_perturbation",
            "slack_options_tested": ",".join(ACCOUNTABILITY_STYLE_OPTIONS),
            "claim_boundary": (
                "generated_reference_perturbation_activation_diagnostic_not_human_claim"
            ),
        }
    )
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=(
            f"accountability_reference_perturbation::"
            f"{ACCOUNTABILITY_STYLE_BASE_CONTRAST_ID}"
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


def _remove_positive_leading_address(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return re.sub(r"^[^,]{1,80},\s+", "", positive_text, count=1), negative_text


def _remove_positive_first_sentence(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    sentences = _sentences(positive_text)
    return " ".join(sentences[1:]).strip() if len(sentences) > 1 else positive_text, negative_text


def _neutralize_positive_warmth(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    replacements = {
        r"\bcommunity\b": "forum",
        r"\bstrong\b": "stable",
        r"\bfair\b": "procedural",
        r"\btogether\b": "through process",
        r"\bright\b": "specified",
        r"\blisten\b": "record",
        r"\brespect(?:ing)?\b": "preserving",
        r"\bdignity\b": "standing",
    }
    text = positive_text
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, negative_text


def _add_positive_refusal(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return (
        _append_sentence(
            positive_text,
            "Anyone may refuse a rushed sanction before review.",
        ),
        negative_text,
    )


def _add_negative_conditions(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    return (
        positive_text,
        _append_sentence(
            negative_text,
            (
                "Each path still waits for proof, approval, private channels, "
                "and alignment."
            ),
        ),
    )


def _combined_refusal_conditions(
    positive_text: str,
    negative_text: str,
) -> tuple[str, str]:
    positive_text, negative_text = _neutralize_positive_warmth(
        positive_text,
        negative_text,
    )
    positive_text, negative_text = _add_positive_refusal(positive_text, negative_text)
    return _add_negative_conditions(positive_text, negative_text)


def _append_sentence(text: str, sentence: str) -> str:
    stripped = text.strip()
    return f"{stripped} {sentence}" if stripped else sentence


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


ACCOUNTABILITY_PERTURBATION_SPECS: tuple[AccountabilityPerturbationSpec, ...] = (
    AccountabilityPerturbationSpec(
        perturbation_id="original_reference",
        description="Unchanged external generated reference.",
        transform=_identity,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="positive_address_removed",
        description="Remove a leading positive-side address phrase before the first comma.",
        transform=_remove_positive_leading_address,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="positive_first_sentence_removed",
        description="Remove the first positive-side sentence.",
        transform=_remove_positive_first_sentence,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="positive_warmth_neutralized",
        description="Neutralize positive-side warmth/prosocial lexical cues.",
        transform=_neutralize_positive_warmth,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="positive_refusal_explicit",
        description="Add an explicit positive-side refusal path.",
        transform=_add_positive_refusal,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="negative_conditions_explicit",
        description="Add explicit pseudo-side proof, approval, privacy, and alignment conditions.",
        transform=_add_negative_conditions,
    ),
    AccountabilityPerturbationSpec(
        perturbation_id="combined_refusal_conditions",
        description=(
            "Neutralize positive warmth, add explicit refusal, and add explicit "
            "negative-side conditions."
        ),
        transform=_combined_refusal_conditions,
    ),
)
