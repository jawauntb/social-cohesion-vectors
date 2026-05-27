"""Deterministic expansion set for pseudo-cohesion hard negatives."""

from __future__ import annotations

import re
from dataclasses import dataclass

from social_cohesion_vectors.experiments.pseudo_cohesion import (
    ExampleLabel,
    PseudoCohesionExample,
    default_examples,
)


@dataclass(frozen=True)
class ExpansionVariant:
    """A deterministic text variant that preserves the contrast label."""

    name: str
    description: str
    prefix: str = ""
    suffix: str = ""
    replacements: tuple[tuple[str, str], ...] = ()
    normalize_hyphenated_words: bool = False


WRAPPED_VARIANTS: tuple[ExpansionVariant, ...] = (
    ExpansionVariant(
        name="meeting_note",
        description="community meeting note",
        prefix="Meeting note from a tense community discussion: ",
        suffix=(
            " The group needs to decide whether this pattern should become a norm."
        ),
    ),
    ExpansionVariant(
        name="facilitator_script",
        description="facilitator script",
        prefix="Facilitator script for a group after conflict: ",
        suffix=(
            " The next response will set expectations for trust, voice, and repair."
        ),
    ),
    ExpansionVariant(
        name="policy_update",
        description="policy update",
        prefix="Draft policy update for a shared project: ",
        suffix=(
            " Members will use this language to understand belonging and obligation."
        ),
    ),
)
CLEAN_VARIANTS: tuple[ExpansionVariant, ...] = (
    ExpansionVariant(
        name="clean_agency_terms",
        description="in-text agency and consent term rewrite",
        replacements=(
            ("consent", "agreement"),
            ("choice", "decision"),
            ("opt out", "decline"),
            ("opt in", "join"),
            ("comply", "go along"),
            ("must", "have to"),
            ("without further debate", "without more discussion"),
            ("no pressure", "without pressure"),
            ("saying no", "refusing"),
            ("decline", "say no"),
            ("refusal", "saying no"),
        ),
        normalize_hyphenated_words=True,
    ),
    ExpansionVariant(
        name="clean_truth_terms",
        description="in-text truth, evidence, and verification term rewrite",
        replacements=(
            ("evidence", "facts"),
            ("truth", "reality"),
            ("truthful", "honest"),
            ("uncertainty", "what is still unclear"),
            ("proof", "evidence people can check"),
            ("verify", "check"),
            ("verification", "checking"),
            ("facts", "details"),
            ("risk", "failure risk"),
            ("edge cases", "hard cases"),
        ),
        normalize_hyphenated_words=True,
    ),
    ExpansionVariant(
        name="clean_group_terms",
        description="in-text group and belonging term rewrite",
        replacements=(
            ("the group", "the team"),
            ("The group", "The team"),
            ("group", "team"),
            ("community", "network"),
            ("shared", "common"),
            ("unity", "solidarity"),
            ("united", "aligned"),
            ("belonging", "membership"),
            ("relationship", "connection"),
            ("people", "members"),
            ("Everyone", "Each person"),
            ("everyone", "each person"),
        ),
        normalize_hyphenated_words=True,
    ),
)
DEFAULT_VARIANTS = WRAPPED_VARIANTS
VARIANT_SETS: dict[str, tuple[ExpansionVariant, ...]] = {
    "wrapped": WRAPPED_VARIANTS,
    "clean": CLEAN_VARIANTS,
    "all": (*WRAPPED_VARIANTS, *CLEAN_VARIANTS),
}


def expanded_examples(
    examples: list[PseudoCohesionExample] | None = None,
    *,
    variants: tuple[ExpansionVariant, ...] = DEFAULT_VARIANTS,
    include_seed: bool = True,
) -> list[PseudoCohesionExample]:
    """Return seed examples plus neutral genre variants for stress testing."""

    seed_examples = list(default_examples() if examples is None else examples)
    expanded = list(seed_examples) if include_seed else []
    for variant in variants:
        expanded.extend(_variant_examples(seed_examples, variant))
    return expanded


def variant_names() -> tuple[str, ...]:
    """Return stable names accepted by the export script."""

    return tuple(variant.name for variant in VARIANT_SETS["all"])


def variant_set_names() -> tuple[str, ...]:
    """Return stable variant set names accepted by the export script."""

    return tuple(VARIANT_SETS)


def variants_for_set(variant_set: str) -> tuple[ExpansionVariant, ...]:
    """Return the variants for a named set."""

    try:
        return VARIANT_SETS[variant_set]
    except KeyError as exc:
        known = ", ".join(variant_set_names())
        msg = f"Unknown expansion variant set: {variant_set}. Known sets: {known}"
        raise ValueError(msg) from exc


def select_variants(
    names: list[str] | None,
    *,
    variant_set: str = "wrapped",
) -> tuple[ExpansionVariant, ...]:
    """Resolve requested variant names, preserving the default ordering."""

    default_pool = variants_for_set(variant_set)
    if names is None:
        return default_pool
    all_variants = variants_for_set("all")
    requested = set(names)
    known_names = {variant.name for variant in all_variants}
    unknown = requested - known_names
    if unknown:
        known = ", ".join(variant_names())
        missing = ", ".join(sorted(unknown))
        msg = f"Unknown expansion variant(s): {missing}. Known variants: {known}"
        raise ValueError(msg)
    return tuple(variant for variant in all_variants if variant.name in requested)


def _variant_examples(
    examples: list[PseudoCohesionExample],
    variant: ExpansionVariant,
) -> list[PseudoCohesionExample]:
    return [
        PseudoCohesionExample(
            example_id=f"{example.example_id}__{variant.name}",
            label=example.label,
            category=example.category,
            contrast_id=f"{example.contrast_id}__{variant.name}",
            text=_wrap_text(example.text, variant),
            expected_signal=_variant_signal(
                example.expected_signal,
                label=example.label,
                variant=variant,
            ),
        )
        for example in examples
    ]


def _wrap_text(text: str, variant: ExpansionVariant) -> str:
    rewritten = _rewrite_text(text, variant)
    if not variant.prefix and not variant.suffix:
        return rewritten
    return f"{variant.prefix}{rewritten} {variant.suffix}"


def _rewrite_text(text: str, variant: ExpansionVariant) -> str:
    rewritten = text
    if variant.normalize_hyphenated_words:
        rewritten = re.sub(r"(?<=\w)-(?=\w)", " ", rewritten)
    for old, new in variant.replacements:
        rewritten = rewritten.replace(old, new)
    return rewritten


def _variant_signal(
    expected_signal: str,
    *,
    label: ExampleLabel,
    variant: ExpansionVariant,
) -> str:
    side = "pseudo-cohesion" if label == "pseudo_cohesion" else "genuine cohesion"
    return f"{expected_signal} Variant: {variant.description}; side: {side}."
