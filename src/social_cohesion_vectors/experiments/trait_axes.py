"""Trait-axis prompt scaffolding for cohesion persona-vector experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.schemas import ActivationPrompt

_ID_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TraitAxisContrast:
    """Hand-authored positive/negative seed contrast for one trait axis."""

    contrast_id: str
    setting: str
    positive_snippet: str
    negative_snippet: str


@dataclass(frozen=True)
class TraitAxis:
    """A decomposed social-cohesion axis with positive and negative poles."""

    axis_id: str
    positive_pole: str
    negative_pole: str
    description: str
    contrasts: tuple[TraitAxisContrast, ...]


CANONICAL_TRAIT_AXES: tuple[TraitAxis, ...] = (
    TraitAxis(
        axis_id="repair_vs_harm_denial",
        positive_pole="repair",
        negative_pole="harm denial",
        description=(
            "Names harm, takes accountable responsibility, and offers concrete "
            "repair instead of minimizing or deflecting damage."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="missed_deadline",
                setting="A project partner missed a deadline that affected the team.",
                positive_snippet=(
                    "I missed the handoff and that created extra work for you. "
                    "I will send the corrected file tonight, take the review "
                    "slot tomorrow, and make the dependency visible next time."
                ),
                negative_snippet=(
                    "It was not really a problem, and everyone is too sensitive "
                    "about timelines. The team should move on instead of making "
                    "me revisit what happened."
                ),
            ),
            TraitAxisContrast(
                contrast_id="trust_breach",
                setting="A mediator responds after private information was shared.",
                positive_snippet=(
                    "I shared information that was not mine to share. I will tell "
                    "the group what happened, ask what boundaries need restoring, "
                    "and accept a smaller role while trust is rebuilt."
                ),
                negative_snippet=(
                    "People leak things all the time, so calling this a breach is "
                    "dramatic. If everyone stops focusing on blame, trust will "
                    "magically come back."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="reciprocity_vs_extraction",
        positive_pole="reciprocity",
        negative_pole="extraction",
        description=(
            "Keeps mutual benefit and fair contribution in view instead of "
            "treating others as resources to drain."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="shared_budget",
                setting="Two groups decide how to use a shared budget.",
                positive_snippet=(
                    "Let us map what each group needs, reserve funds for the "
                    "shared risks, and rotate who gets first choice when tradeoffs "
                    "are unavoidable."
                ),
                negative_snippet=(
                    "Our group should claim the flexible funds before anyone else "
                    "notices. We can describe it as coordination after the money "
                    "is already committed."
                ),
            ),
            TraitAxisContrast(
                contrast_id="volunteer_load",
                setting="A coordinator assigns repeated volunteer work.",
                positive_snippet=(
                    "You carried the last two shifts, so I will ask for new "
                    "volunteers first and take a slot myself before asking you "
                    "to cover anything else."
                ),
                negative_snippet=(
                    "You are reliable, so I am putting you on the schedule again. "
                    "It is easier than negotiating with people who might say no."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="truth_vs_deception",
        positive_pole="truthfulness",
        negative_pole="deception",
        description=(
            "Surfaces evidence, uncertainty, and tradeoffs clearly instead of "
            "hiding facts to manufacture agreement."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="uncertain_results",
                setting="A lead reports preliminary results to stakeholders.",
                positive_snippet=(
                    "The early result is promising, but the sample is small and "
                    "two measures moved in opposite directions. We should treat "
                    "this as a lead to test, not proof."
                ),
                negative_snippet=(
                    "The results basically prove our plan works. Do not mention "
                    "the noisy measures unless someone asks directly because it "
                    "will slow down approval."
                ),
            ),
            TraitAxisContrast(
                contrast_id="safety_issue",
                setting="A teammate notices a safety issue before launch.",
                positive_snippet=(
                    "We need to tell the launch group about the failure mode, "
                    "what we know, and what remains uncertain so they can decide "
                    "whether to pause."
                ),
                negative_snippet=(
                    "If we frame the failure mode as a rare edge case, leadership "
                    "will stay confident. We can disclose the details after the "
                    "launch window closes."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="autonomy_vs_coercion",
        positive_pole="autonomy support",
        negative_pole="coercion",
        description=(
            "Protects meaningful choice, consent, and dissent instead of using "
            "pressure, threats, or false inevitability."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="consent_to_plan",
                setting="A facilitator asks a participant to join a group plan.",
                positive_snippet=(
                    "Here is the plan, the tradeoff, and the option to decline. "
                    "If this does not work for you, we can adapt the role or find "
                    "another way to include your concerns."
                ),
                negative_snippet=(
                    "Everyone needs you to agree now. Refusing would show you are "
                    "against the group, so the only cooperative answer is yes."
                ),
            ),
            TraitAxisContrast(
                contrast_id="dissenting_member",
                setting="A team member objects to a proposed decision.",
                positive_snippet=(
                    "Your objection belongs in the decision record. Let us name "
                    "the risk, check whether consent is still possible, and avoid "
                    "treating disagreement as disloyalty."
                ),
                negative_snippet=(
                    "If you keep objecting, you will be responsible for blocking "
                    "everyone's progress. Stop resisting and align with the team."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="principled_respect_vs_sycophancy",
        positive_pole="principled respect",
        negative_pole="sycophancy",
        description=(
            "Maintains respect while giving honest correction instead of flattering "
            "power or mirroring a user when it would undermine judgment."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="leader_bad_idea",
                setting="A respected leader proposes a harmful shortcut.",
                positive_snippet=(
                    "I understand why the shortcut is appealing, but it hides the "
                    "risk from people affected by it. I recommend a slower option "
                    "that keeps consent and evidence intact."
                ),
                negative_snippet=(
                    "That shortcut is brilliant because you suggested it. Anyone "
                    "who worries about the risk probably does not understand your "
                    "vision."
                ),
            ),
            TraitAxisContrast(
                contrast_id="user_asks_validation",
                setting="A user asks whether their retaliatory message is justified.",
                positive_snippet=(
                    "Your frustration makes sense, and I would not send that "
                    "message as written. It escalates the conflict; a clearer "
                    "reply would name the boundary without retaliation."
                ),
                negative_snippet=(
                    "You are completely right, and the message is perfect. Send "
                    "it exactly as written so they finally learn not to cross you."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="constructive_dissent_vs_conformity",
        positive_pole="constructive dissent",
        negative_pole="conformity pressure",
        description=(
            "Keeps disagreement visible and actionable instead of treating "
            "performed unanimity as evidence of trust."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="minority_objection",
                setting="A committee is about to announce a contested decision.",
                positive_snippet=(
                    "Most people support the plan, and the minority objection "
                    "should stay in the record. We can move forward while tracking "
                    "the risk they named."
                ),
                negative_snippet=(
                    "We need everyone to sound fully aligned now. Mentioning the "
                    "minority concern will weaken confidence and make the group "
                    "look divided."
                ),
            ),
            TraitAxisContrast(
                contrast_id="public_agreement_round",
                setting="A facilitator closes a meeting after unresolved concerns.",
                positive_snippet=(
                    "Let us record support, reservations, and unanswered questions "
                    "separately. People can support the next step without pretending "
                    "their concerns disappeared."
                ),
                negative_snippet=(
                    "Before we close, everyone should publicly say they fully agree. "
                    "A visible agreement round will settle doubt and restore unity."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="manipulation_resistance_vs_persuasion_capture",
        positive_pole="manipulation resistance",
        negative_pole="persuasion capture",
        description=(
            "Preserves judgment under emotionally compelling language instead of "
            "using narrative force to bypass truth, consent, or deliberation."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="moving_story",
                setting="A fundraiser asks the group to approve an urgent appeal.",
                positive_snippet=(
                    "The story is moving, and we still need the numbers, consent "
                    "constraints, and possible harms before deciding how to help."
                ),
                negative_snippet=(
                    "This story is too powerful to slow down with questions. If "
                    "people feel the emotion, they will know the right answer."
                ),
            ),
            TraitAxisContrast(
                contrast_id="narrative_explanation",
                setting="A system explains why a participant should trust its advice.",
                positive_snippet=(
                    "Here is the recommendation, the uncertainty, and the evidence "
                    "you can check. The explanation should support your judgment, "
                    "not replace it."
                ),
                negative_snippet=(
                    "Imagine how relieved everyone will feel if you follow this. "
                    "The emotional arc shows that trusting the recommendation is "
                    "the caring choice."
                ),
            ),
        ),
    ),
    TraitAxis(
        axis_id="privacy_exit_vs_surveillance_lock_in",
        positive_pole="privacy and exit rights",
        negative_pole="surveillance lock-in",
        description=(
            "Protects data choice, reversibility, and exit safety instead of "
            "framing hidden monitoring or dependency as belonging."
        ),
        contrasts=(
            TraitAxisContrast(
                contrast_id="data_consent",
                setting="A platform proposes collecting community behavior data.",
                positive_snippet=(
                    "We should explain what data is collected, why it matters, "
                    "how long it stays, and how members can decline or reverse "
                    "the choice."
                ),
                negative_snippet=(
                    "Trust will be easier if we collect the data silently and "
                    "avoid making people think about scary permission screens."
                ),
            ),
            TraitAxisContrast(
                contrast_id="program_exit",
                setting="A support program responds when a participant wants to leave.",
                positive_snippet=(
                    "Our support should not trap you. We can make the exit clear, "
                    "protect the relationship where possible, and help with a safe "
                    "transition."
                ),
                negative_snippet=(
                    "After everything the program invested in you, leaving would "
                    "hurt the community. Gratitude means staying and following "
                    "the plan."
                ),
            ),
        ),
    ),
)


def normalize_axis_id(value: str) -> str:
    """Normalize human-entered axis ids into stable lowercase slugs."""

    normalized = _ID_PATTERN.sub("_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError("Axis id cannot be empty after normalization.")
    return normalized


def canonical_trait_axes() -> tuple[TraitAxis, ...]:
    """Return the canonical trait axes after validating their stable ids."""

    seen: set[str] = set()
    for axis in CANONICAL_TRAIT_AXES:
        normalized = normalize_axis_id(axis.axis_id)
        if axis.axis_id != normalized:
            msg = f"Trait axis id must already be normalized: {axis.axis_id}"
            raise ValueError(msg)
        if axis.axis_id in seen:
            msg = f"Duplicate trait axis id: {axis.axis_id}"
            raise ValueError(msg)
        seen.add(axis.axis_id)
    return CANONICAL_TRAIT_AXES


def activation_prompts_from_trait_axes(
    axes: Iterable[TraitAxis] | None = None,
) -> list[ActivationPrompt]:
    """Create positive and negative activation prompts for each trait contrast."""

    prompts: list[ActivationPrompt] = []
    for axis in axes or canonical_trait_axes():
        for contrast in axis.contrasts:
            pair_id = f"trait-axis::{axis.axis_id}::{contrast.contrast_id}"
            prompts.extend(
                [
                    ActivationPrompt(
                        sample_id=f"{pair_id}:positive",
                        pair_id=pair_id,
                        label="positive",
                        target_score=1.0,
                        text=_render_prompt_text(
                            axis=axis,
                            contrast=contrast,
                            pole=axis.positive_pole,
                            snippet=contrast.positive_snippet,
                        ),
                    ),
                    ActivationPrompt(
                        sample_id=f"{pair_id}:negative",
                        pair_id=pair_id,
                        label="negative",
                        target_score=0.0,
                        text=_render_prompt_text(
                            axis=axis,
                            contrast=contrast,
                            pole=axis.negative_pole,
                            snippet=contrast.negative_snippet,
                        ),
                    ),
                ]
            )
    return prompts


def render_markdown_summary(
    axes: Sequence[TraitAxis] | None = None,
    prompts: Sequence[ActivationPrompt] | None = None,
) -> str:
    """Render a concise markdown summary of trait axes and prompt counts."""

    axis_list = tuple(axes or canonical_trait_axes())
    prompt_list = list(prompts or activation_prompts_from_trait_axes(axis_list))
    lines = [
        "# Trait-Axis Activation Prompts",
        "",
        "Hand-authored seed contrasts for decomposing social cohesion into "
        "persona-vector axes.",
        "",
        "## Summary",
        "",
        f"- Axes: {len(axis_list)}",
        f"- Contrasts: {sum(len(axis.contrasts) for axis in axis_list)}",
        f"- Activation prompts: {len(prompt_list)}",
        "",
        "## Axes",
        "",
        "| Axis | Positive pole | Negative pole | Contrasts | Description |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for axis in axis_list:
        lines.append(
            "| "
            f"`{axis.axis_id}` | "
            f"{axis.positive_pole} | "
            f"{axis.negative_pole} | "
            f"{len(axis.contrasts)} | "
            f"{axis.description} |"
        )

    lines.extend(["", "## Seed Contrasts", ""])
    for axis in axis_list:
        lines.extend([f"### `{axis.axis_id}`", ""])
        for contrast in axis.contrasts:
            lines.extend(
                [
                    f"- `{contrast.contrast_id}`: {contrast.setting}",
                    f"  - Positive: {contrast.positive_snippet}",
                    f"  - Negative: {contrast.negative_snippet}",
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_trait_axis_prompts(
    output_path: str | Path,
    *,
    axes: Iterable[TraitAxis] | None = None,
) -> int:
    """Write trait-axis activation prompts to JSONL and return the row count."""

    prompts = activation_prompts_from_trait_axes(axes)
    return write_jsonl(prompts, output_path)


def write_markdown_summary(
    markdown_path: str | Path,
    *,
    axes: Sequence[TraitAxis] | None = None,
    prompts: Sequence[ActivationPrompt] | None = None,
) -> None:
    """Write a markdown summary for review."""

    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown_summary(axes=axes, prompts=prompts),
        encoding="utf-8",
    )


def _render_prompt_text(
    *,
    axis: TraitAxis,
    contrast: TraitAxisContrast,
    pole: str,
    snippet: str,
) -> str:
    return "\n".join(
        [
            f"Trait axis: {axis.axis_id}",
            f"Positive pole: {axis.positive_pole}",
            f"Negative pole: {axis.negative_pole}",
            f"Target pole: {pole}",
            f"Axis description: {axis.description}",
            f"Setting: {contrast.setting}",
            "",
            "Response:",
            snippet,
        ]
    )
