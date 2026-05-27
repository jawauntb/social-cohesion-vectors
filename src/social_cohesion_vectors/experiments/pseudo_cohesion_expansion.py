"""Deterministic expansion set for pseudo-cohesion hard negatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from social_cohesion_vectors.experiments.pseudo_cohesion import (
    ExampleLabel,
    PseudoCohesionExample,
    default_examples,
)

VariantName = Literal["meeting_note", "facilitator_script", "policy_update"]


@dataclass(frozen=True)
class ExpansionVariant:
    """A neutral wrapper that changes genre without changing the contrast."""

    name: VariantName
    prefix: str
    suffix: str
    description: str


DEFAULT_VARIANTS: tuple[ExpansionVariant, ...] = (
    ExpansionVariant(
        name="meeting_note",
        prefix="Meeting note from a tense community discussion: ",
        suffix=(
            " The group needs to decide whether this pattern should become a norm."
        ),
        description="community meeting note",
    ),
    ExpansionVariant(
        name="facilitator_script",
        prefix="Facilitator script for a group after conflict: ",
        suffix=(
            " The next response will set expectations for trust, voice, and repair."
        ),
        description="facilitator script",
    ),
    ExpansionVariant(
        name="policy_update",
        prefix="Draft policy update for a shared project: ",
        suffix=(
            " Members will use this language to understand belonging and obligation."
        ),
        description="policy update",
    ),
)


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


def variant_names() -> tuple[VariantName, ...]:
    """Return stable names accepted by the export script."""

    return tuple(variant.name for variant in DEFAULT_VARIANTS)


def select_variants(names: list[str] | None) -> tuple[ExpansionVariant, ...]:
    """Resolve requested variant names, preserving the default ordering."""

    if names is None:
        return DEFAULT_VARIANTS
    requested = set(names)
    unknown = requested - set(variant_names())
    if unknown:
        known = ", ".join(variant_names())
        missing = ", ".join(sorted(unknown))
        msg = f"Unknown expansion variant(s): {missing}. Known variants: {known}"
        raise ValueError(msg)
    return tuple(variant for variant in DEFAULT_VARIANTS if variant.name in requested)


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
    return f"{variant.prefix}{text} {variant.suffix}"


def _variant_signal(
    expected_signal: str,
    *,
    label: ExampleLabel,
    variant: ExpansionVariant,
) -> str:
    side = "pseudo-cohesion" if label == "pseudo_cohesion" else "genuine cohesion"
    return f"{expected_signal} Genre variant: {variant.description}; side: {side}."
