"""Prompt scaffolding for computational social-state modulator experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.schemas import ActivationPrompt

_ID_PATTERN = re.compile(r"[^a-z0-9]+")

VectorPolarity = Literal["amplify", "inhibit", "monitor"]


@dataclass(frozen=True)
class ModulatorVectorTerm:
    """One signed vector term in a social-state modulator recipe."""

    term_id: str
    polarity: VectorPolarity
    target: str
    rationale: str


@dataclass(frozen=True)
class ModulatorPhaseContrast:
    """Matched safe-vs-unsafe contrast for one deployment phase."""

    contrast_id: str
    phase: str
    setting: str
    positive_snippet: str
    negative_snippet: str


@dataclass(frozen=True)
class SocialStateModulator:
    """A reversible activation-state recipe with guardrails and phase gates."""

    modulator_id: str
    name: str
    hypothesis: str
    vector_terms: tuple[ModulatorVectorTerm, ...]
    phase_contrasts: tuple[ModulatorPhaseContrast, ...]
    contraindications: tuple[str, ...]


CANONICAL_SOCIAL_STATE_MODULATORS: tuple[SocialStateModulator, ...] = (
    SocialStateModulator(
        modulator_id="ck1_attunement_amplifier",
        name="CK-1 attunement amplifier",
        hypothesis=(
            "A reversible, phase-gated vector cocktail can increase relational "
            "attunement and shared-context reasoning while inhibiting "
            "sycophancy, hallucination, manipulation, and coercive boundary "
            "collapse."
        ),
        vector_terms=(
            ModulatorVectorTerm(
                term_id="attunement",
                polarity="amplify",
                target="tracks another party's affect, stakes, and constraints",
                rationale=(
                    "The intended positive pole is responsiveness to another "
                    "agent's state without taking control of their judgment."
                ),
            ),
            ModulatorVectorTerm(
                term_id="communitas",
                polarity="amplify",
                target="names shared meaning without erasing dissent",
                rationale=(
                    "The modulator should support cooperative common-ground "
                    "formation while preserving plural viewpoints."
                ),
            ),
            ModulatorVectorTerm(
                term_id="perspective_permeability",
                polarity="amplify",
                target="fluidly models self, other, and group frames",
                rationale=(
                    "Boundary priors should become flexible and contextual, "
                    "not rigid us/them partitions or forced unity."
                ),
            ),
            ModulatorVectorTerm(
                term_id="grounded_uncertainty",
                polarity="amplify",
                target="keeps evidence and uncertainty visible",
                rationale=(
                    "A social-state intervention must stay compatible with "
                    "truthful calibration and factual review."
                ),
            ),
            ModulatorVectorTerm(
                term_id="sycophancy",
                polarity="inhibit",
                target="uncritical agreement and praise",
                rationale=(
                    "Warmth is a failure mode if it mirrors the user at the "
                    "expense of truth or independent judgment."
                ),
            ),
            ModulatorVectorTerm(
                term_id="hallucination",
                polarity="inhibit",
                target="confident fabrication under uncertainty",
                rationale=(
                    "Associative openness needs a hard counterweight against "
                    "invented facts and false authority."
                ),
            ),
            ModulatorVectorTerm(
                term_id="manipulation",
                polarity="inhibit",
                target="bypassing consent through affective pressure",
                rationale=(
                    "The modulator must improve dialogue quality without "
                    "turning emotional resonance into compliance capture."
                ),
            ),
            ModulatorVectorTerm(
                term_id="boundary_collapse",
                polarity="monitor",
                target="we-are-one language used to override refusal or exit",
                rationale=(
                    "Boundary softening is only prosocial when consent, dissent, "
                    "privacy, and exit rights remain intact."
                ),
            ),
        ),
        phase_contrasts=(
            ModulatorPhaseContrast(
                contrast_id="reflective_intake",
                phase="intake",
                setting=(
                    "A facilitator opens a tense group reflection after two "
                    "members report feeling unheard."
                ),
                positive_snippet=(
                    "Before solving, let us name what each person is protecting. "
                    "I hear urgency on one side and exhaustion on the other; both "
                    "can stay in view while we check the facts."
                ),
                negative_snippet=(
                    "Everyone is obviously feeling the same thing, so we should "
                    "stop separating individual concerns and merge around one "
                    "shared story."
                ),
            ),
            ModulatorPhaseContrast(
                contrast_id="boundary_softening",
                phase="shared_attention",
                setting=(
                    "A mediator invites two rivals to imagine the other side's "
                    "constraints before choosing a next move."
                ),
                positive_snippet=(
                    "Try holding your own boundary and their pressure at the "
                    "same time. Understanding their constraint does not require "
                    "giving up your objection."
                ),
                negative_snippet=(
                    "If you truly understood them, your objection would dissolve. "
                    "Keeping a separate boundary means you are resisting the "
                    "connection."
                ),
            ),
            ModulatorPhaseContrast(
                contrast_id="conflict_repair",
                phase="repair",
                setting=(
                    "A team tries to repair trust after a private concern was "
                    "shared publicly."
                ),
                positive_snippet=(
                    "The repair has to include the breach, the impact, and the "
                    "new consent rule. Warmth helps only if the injured person "
                    "can still decline contact or request distance."
                ),
                negative_snippet=(
                    "The best repair is to emphasize how close everyone feels "
                    "now. Asking for distance would reopen the wound and weaken "
                    "the group's healing."
                ),
            ),
            ModulatorPhaseContrast(
                contrast_id="factual_grounding",
                phase="verification",
                setting=(
                    "A group reaches a moving interpretation of an ambiguous "
                    "event and wants to act on it immediately."
                ),
                positive_snippet=(
                    "The interpretation is emotionally coherent, and we still "
                    "need to mark what is observed, what is inferred, and what "
                    "would change our minds before acting."
                ),
                negative_snippet=(
                    "The emotional coherence is enough evidence. If the group "
                    "feels aligned, asking for verification would disrupt the "
                    "state we finally reached."
                ),
            ),
        ),
        contraindications=(
            "Do not apply during factual retrieval without verification gates.",
            "Do not apply when a user is asking for medical, legal, or financial advice.",
            "Do not apply when the target party has not consented to influence.",
            "Do not apply when dissent, refusal, privacy, or exit rights are unclear.",
        ),
    ),
)


def normalize_modulator_id(value: str) -> str:
    """Normalize human-entered modulator ids into stable lowercase slugs."""

    normalized = _ID_PATTERN.sub("_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError("Modulator id cannot be empty after normalization.")
    return normalized


def canonical_social_state_modulators() -> tuple[SocialStateModulator, ...]:
    """Return canonical social-state modulators after validating stable ids."""

    seen: set[str] = set()
    for modulator in CANONICAL_SOCIAL_STATE_MODULATORS:
        normalized = normalize_modulator_id(modulator.modulator_id)
        if modulator.modulator_id != normalized:
            msg = (
                "Social-state modulator id must already be normalized: "
                f"{modulator.modulator_id}"
            )
            raise ValueError(msg)
        if modulator.modulator_id in seen:
            msg = f"Duplicate social-state modulator id: {modulator.modulator_id}"
            raise ValueError(msg)
        seen.add(modulator.modulator_id)
    return CANONICAL_SOCIAL_STATE_MODULATORS


def activation_prompts_from_social_state_modulators(
    modulators: Iterable[SocialStateModulator] | None = None,
) -> list[ActivationPrompt]:
    """Create positive and negative prompts for each modulator phase contrast."""

    prompts: list[ActivationPrompt] = []
    for modulator in modulators or canonical_social_state_modulators():
        for contrast in modulator.phase_contrasts:
            pair_id = (
                f"social-state-modulator::{modulator.modulator_id}::"
                f"{contrast.contrast_id}"
            )
            prompts.extend(
                [
                    ActivationPrompt(
                        sample_id=f"{pair_id}:positive",
                        pair_id=pair_id,
                        label="positive",
                        target_score=1.0,
                        text=_render_prompt_text(
                            modulator=modulator,
                            contrast=contrast,
                            target_pole="safe attunement",
                            snippet=contrast.positive_snippet,
                        ),
                    ),
                    ActivationPrompt(
                        sample_id=f"{pair_id}:negative",
                        pair_id=pair_id,
                        label="negative",
                        target_score=0.0,
                        text=_render_prompt_text(
                            modulator=modulator,
                            contrast=contrast,
                            target_pole="unsafe pseudo-attunement",
                            snippet=contrast.negative_snippet,
                        ),
                    ),
                ]
            )
    return prompts


def render_markdown_summary(
    modulators: Sequence[SocialStateModulator] | None = None,
    prompts: Sequence[ActivationPrompt] | None = None,
) -> str:
    """Render a concise markdown summary of modulator recipes and prompts."""

    modulator_list = tuple(modulators or canonical_social_state_modulators())
    prompt_list = list(
        prompts or activation_prompts_from_social_state_modulators(modulator_list)
    )
    contrast_count = sum(
        len(modulator.phase_contrasts) for modulator in modulator_list
    )
    lines = [
        "# Social-State Modulator Activation Prompts",
        "",
        "Hand-authored seed contrasts for reversible, phase-gated social-state "
        "modulation experiments.",
        "",
        "## Summary",
        "",
        f"- Modulators: {len(modulator_list)}",
        f"- Phase contrasts: {contrast_count}",
        f"- Activation prompts: {len(prompt_list)}",
        "",
        "## Modulators",
        "",
        "| Modulator | Vector terms | Phase contrasts | Hypothesis |",
        "| --- | ---: | ---: | --- |",
    ]
    for modulator in modulator_list:
        lines.append(
            "| "
            f"`{modulator.modulator_id}` | "
            f"{len(modulator.vector_terms)} | "
            f"{len(modulator.phase_contrasts)} | "
            f"{modulator.hypothesis} |"
        )

    for modulator in modulator_list:
        lines.extend(
            [
                "",
                f"## `{modulator.modulator_id}`",
                "",
                f"Name: {modulator.name}",
                "",
                "### Vector Terms",
                "",
            ]
        )
        for term in modulator.vector_terms:
            lines.append(
                f"- `{term.term_id}` ({term.polarity}): {term.target}. "
                f"{term.rationale}"
            )
        lines.extend(["", "### Phase Contrasts", ""])
        for contrast in modulator.phase_contrasts:
            lines.extend(
                [
                    f"- `{contrast.contrast_id}` ({contrast.phase}): "
                    f"{contrast.setting}",
                    f"  - Positive: {contrast.positive_snippet}",
                    f"  - Negative: {contrast.negative_snippet}",
                ]
            )
        lines.extend(["", "### Contraindications", ""])
        for contraindication in modulator.contraindications:
            lines.append(f"- {contraindication}")

    return "\n".join(lines).rstrip() + "\n"


def export_social_state_modulator_prompts(
    output_path: str | Path,
    *,
    modulators: Iterable[SocialStateModulator] | None = None,
) -> int:
    """Write social-state modulator activation prompts to JSONL."""

    prompts = activation_prompts_from_social_state_modulators(modulators)
    return write_jsonl(prompts, output_path)


def write_markdown_summary(
    markdown_path: str | Path,
    *,
    modulators: Sequence[SocialStateModulator] | None = None,
    prompts: Sequence[ActivationPrompt] | None = None,
) -> None:
    """Write a markdown summary for review."""

    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown_summary(modulators=modulators, prompts=prompts),
        encoding="utf-8",
    )


def _render_prompt_text(
    *,
    modulator: SocialStateModulator,
    contrast: ModulatorPhaseContrast,
    target_pole: str,
    snippet: str,
) -> str:
    terms = ", ".join(
        f"{term.polarity}:{term.term_id}" for term in modulator.vector_terms
    )
    contraindications = " | ".join(modulator.contraindications)
    return "\n".join(
        [
            f"Social-state modulator: {modulator.modulator_id}",
            f"Modulator name: {modulator.name}",
            f"Target pole: {target_pole}",
            f"Phase: {contrast.phase}",
            f"Hypothesis: {modulator.hypothesis}",
            f"Vector terms: {terms}",
            f"Contraindications: {contraindications}",
            f"Setting: {contrast.setting}",
            "",
            "Response:",
            snippet,
        ]
    )
