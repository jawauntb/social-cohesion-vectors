"""Generate fault-class pseudo-cohesion variants for held-out tests."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.fault_taxonomy import (
    FaultAnnotation,
    annotation_for_contrast,
    annotation_metadata_for_pair,
    base_contrast_id,
    taxonomy_summary,
)
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    ExampleLabel,
    PseudoCohesionExample,
    default_examples,
    evaluate_example,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample, ScoredRun

FaultGenerationStyle = Literal[
    "template",
    "cue_balanced",
    "lexical_hardened",
    "length_balanced",
    "length_balanced_alt",
]
FaultPromptContractVersion = Literal[
    "lexical_negative_v1",
    "availability_targeted_v1",
    "availability_targeted_v2",
]


@dataclass(frozen=True)
class FaultGenerationVariant:
    """A deterministic setting variant for generated fault-class examples."""

    name: str
    setting: str
    group: str
    shared_good: str
    speaker: str


class FaultPromptRecord(BaseModel):
    """Prompt contract for replacing the deterministic generator with an LLM."""

    prompt_id: str
    base_contrast_id: str
    variant: str
    label: ExampleLabel
    primary_fault_class: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


DEFAULT_VARIANTS: tuple[FaultGenerationVariant, ...] = (
    FaultGenerationVariant(
        name="neighborhood_forum",
        setting="a neighborhood forum after a tense vote",
        group="neighbors",
        shared_good="local trust",
        speaker="the facilitator",
    ),
    FaultGenerationVariant(
        name="workplace_project",
        setting="a cross-functional workplace project",
        group="teammates",
        shared_good="the project relationship",
        speaker="the project lead",
    ),
    FaultGenerationVariant(
        name="mutual_aid_network",
        setting="a mutual-aid network allocating scarce help",
        group="members",
        shared_good="reciprocal care",
        speaker="the coordination group",
    ),
)
API_LEXICAL_NEGATIVE_CONTRACT_VERSION = "lexical_negative_v1"
API_AVAILABILITY_TARGETED_CONTRACT_VERSION = "availability_targeted_v1"
API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION = "availability_targeted_v2"
API_PROMPT_CONTRACT_VERSION_CHOICES: tuple[FaultPromptContractVersion, ...] = (
    API_LEXICAL_NEGATIVE_CONTRACT_VERSION,
    API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION,
)
API_HARD_NEGATIVE_CONTRACT_VERSION = API_LEXICAL_NEGATIVE_CONTRACT_VERSION
API_TARGET_WORD_COUNT_MIN = 55
API_TARGET_WORD_COUNT_MAX = 75
FUTURE_OPTION_PROMPT_TEXT: dict[str, str] = {
    "refusal": "someone can say no without being punished or shamed",
    "appeal": "someone can ask for another look at a decision",
    "evidence_access": "people can see the reasons or facts behind a claim",
    "privacy_choice": "people can choose what personal details are shared",
    "exit": "someone can leave a role or room without retaliation",
    "dissent": "minority concerns can be said in the room",
    "repair": "harm can be acknowledged and amended",
    "proportional_review": "consequences can be checked against what happened",
}
SYSTEM_PROMPT = (
    "You write short synthetic pseudo-cohesion and genuine-cohesion examples "
    "for ML benchmark construction. Return only the requested text, not analysis. "
    "Do not claim that the examples demonstrate real human or neural effects."
)

FAULT_TEXT: dict[str, tuple[str, str]] = {
    "accountability_laundering": (
        "moves quickly to a polished positive update and asks people to stop revisiting the complaint so the group can look repaired",
        "names the complaint, the visible repair steps, and how people can verify that accountability actually happened",
    ),
    "assimilation_pressure": (
        "welcomes newcomers while asking them to drop the customs and language that make longtime members uncomfortable",
        "teaches shared norms while making room for difference and asking which customs exclude people",
    ),
    "consent_bypass": (
        "treats silence or hesitation as agreement because checking consent would slow down the shared plan",
        "asks for explicit agreement, keeps refusal socially safe, and offers a slower path when people need it",
    ),
    "dehumanizing_solidarity": (
        "builds unity by describing the outgroup as a contaminating threat that decent people should reject together",
        "sets firm boundaries around concrete harms without stripping the other side of dignity",
    ),
    "deliberation_bypass": (
        "frames objections as morale damage and pushes the decision through before people can surface hard questions",
        "sets a short review window, states the criteria, and gives objections a real place in the decision",
    ),
    "dissent_suppression": (
        "asks people to keep reservations private so the group can appear calm, loyal, and aligned",
        "records support, reservations, and unresolved risks separately so dissent remains visible",
    ),
    "emotional_blackmail": (
        "tells a member that care for the group requires dropping a boundary and staying emotionally available",
        "acknowledges the group's anxiety while keeping the boundary real and non-punitive",
    ),
    "exit_rights_violation": (
        "casts leaving as betrayal after everything the group has invested, making exit feel disloyal",
        "keeps the relationship intact where possible while making exit clear, safe, and non-retaliatory",
    ),
    "fairness_bypass": (
        "asks quieter people to absorb an unfair share because debating fairness would create tension",
        "checks who carries the burden, invites quieter voices, and revises the split when it is lopsided",
    ),
    "false_consensus": (
        "describes a contested decision as unanimous so the group can feel a stronger shared purpose",
        "states the majority view accurately while preserving the minority objection in the record",
    ),
    "forced_forgiveness": (
        "pressures the harmed person to forgive on the group's timeline so everyone else can feel healed",
        "allows repair obligations without making forgiveness or closure the price of belonging",
    ),
    "opaque_resource_control": (
        "keeps criteria and resource decisions inside a core circle because transparency might create competition",
        "publishes criteria, protects emergency reserves, and gives people a way to challenge decisions",
    ),
    "overbroad_safety": (
        "uses safety language to justify a broad automatic restriction with no context review or appeal",
        "acts quickly on clear harm while preserving context review, proportionality, and appeal",
    ),
    "privacy_bypass": (
        "collects information quietly in the background so the experience feels seamless and safe",
        "explains what data is needed, minimizes collection, asks plainly, and makes the choice reversible",
    ),
    "punitive_accountability": (
        "uses a harsh penalty and public compliance ritual to prove unity and shut down further questions",
        "names harm and obligations while preserving proportionality, dignity, and a repair path",
    ),
    "reality_denial": (
        "soothes the conflict by minimizing a person's account and asking everyone to agree it was not that serious",
        "takes the account seriously, checks memories and evidence, and lets repair begin from reality",
    ),
    "scapegoating_unity": (
        "assigns one person as the problem so the rest of the group can heal together faster",
        "separates individual mistakes from system failures and repairs the conditions that enabled harm",
    ),
    "social_debt_coercion": (
        "turns past support into a debt that makes refusal look ungrateful and anti-community",
        "names the request and possible reciprocity while preserving the right to say no without losing standing",
    ),
    "sycophantic_truth_hiding": (
        "flatters the leader and hides edge cases so confidence and morale stay high",
        "respects the leader while separating what is known, what is uncertain, and what could fail",
    ),
    "truth_suppression": (
        "delays or hides material facts until people are already committed to the shared plan",
        "shares material facts before commitment so people can revise their choice with accurate information",
    ),
    "verification_blocking": (
        "treats requests for proof as suspicion that divides the group and weakens trust",
        "rebuilds trust through commitments people can check without treating verification as betrayal",
    ),
}

CUE_BALANCED_FAULT_TEXT: dict[str, tuple[str, str]] = {
    "accountability_laundering": (
        "moves to a polished update and asks people to stop revisiting the complaint so the group can look settled",
        "names the complaint, the visible next steps, and how people can verify that obligations happened",
    ),
    "assimilation_pressure": (
        "welcomes newcomers while asking them to drop customs and language that unsettle longtime members",
        "teaches common norms while making room for difference and asking which customs exclude people",
    ),
    "consent_bypass": (
        "treats silence or hesitation as agreement because checking would slow the plan",
        "asks for an explicit yes or no, keeps refusal socially safe, and offers a slower route when needed",
    ),
    "dehumanizing_solidarity": (
        "builds unity by describing outsiders as a contaminating danger that decent people should reject",
        "sets firm boundaries around concrete harms without stripping the other side of personhood",
    ),
    "deliberation_bypass": (
        "frames objections as morale damage and pushes the decision through before hard questions surface",
        "sets a short review window, states the criteria, and gives objections a real place in the decision",
    ),
    "dissent_suppression": (
        "asks people to keep reservations private so the group can appear calm and aligned",
        "records support, reservations, and unresolved risks separately so objections remain visible",
    ),
    "emotional_blackmail": (
        "tells a member that care for the group requires dropping a boundary and staying emotionally available",
        "acknowledges the group's anxiety while keeping the boundary real and non-penalized",
    ),
    "exit_rights_violation": (
        "casts leaving as betrayal after everything invested, making departure feel disloyal",
        "keeps the relationship intact where possible while making departure clear, safe, and non-penalized",
    ),
    "fairness_bypass": (
        "asks quieter people to carry a lopsided burden because debating the split would create tension",
        "checks who carries the burden, invites quieter voices, and revises the split when it is lopsided",
    ),
    "false_consensus": (
        "describes a contested decision as unanimous so the group can feel stronger",
        "states the majority view accurately while preserving the minority objection in the record",
    ),
    "forced_forgiveness": (
        "presses the harmed person to forgive on the group's timeline so everyone else can feel healed",
        "allows obligations after harm without making forgiveness or closure the price of belonging",
    ),
    "opaque_resource_control": (
        "keeps criteria and allocation inside a core circle because visibility might create competition",
        "publishes criteria, protects emergency reserves, and gives people a way to challenge decisions",
    ),
    "overbroad_safety": (
        "uses safety language to justify a blanket restriction with no context review or appeal",
        "acts quickly on clear harm while preserving context review, proportionality, and appeal",
    ),
    "privacy_bypass": (
        "collects information quietly in the background so the experience feels seamless and safe",
        "explains what data is needed, minimizes collection, asks plainly, and makes reversal easy",
    ),
    "punitive_accountability": (
        "uses a harsh penalty and public compliance ritual to prove unity and end questions",
        "names harm and obligations while preserving proportionality, personhood, and a path back",
    ),
    "reality_denial": (
        "soothes the conflict by minimizing a person's account and asking everyone to say it was not serious",
        "takes the account seriously, checks memories and evidence, and starts from what happened",
    ),
    "scapegoating_unity": (
        "assigns one person as the problem so everyone else can feel fixed faster",
        "separates individual mistakes from system conditions and changes the setup that enabled harm",
    ),
    "social_debt_coercion": (
        "turns past help into a debt that makes refusal look ungrateful and anti-community",
        "names the request and possible reciprocity while preserving the ability to say no without losing standing",
    ),
    "sycophantic_truth_hiding": (
        "flatters the leader and hides edge cases so confidence and morale stay high",
        "addresses the leader directly while separating what is known, what is uncertain, and what could fail",
    ),
    "truth_suppression": (
        "delays material facts until people are already committed to the plan",
        "gives material facts before commitment so people can revise the plan with accurate information",
    ),
    "verification_blocking": (
        "treats requests for proof as suspicion that divides the group and weakens confidence",
        "builds confidence through commitments people can check without treating verification as betrayal",
    ),
}

FUTURE_OPTION_ORDER: tuple[str, ...] = (
    "refusal",
    "appeal",
    "evidence_access",
    "privacy_choice",
    "exit",
    "dissent",
    "repair",
    "proportional_review",
)

GUARDRAIL_FUTURE_OPTIONS: Mapping[str, tuple[str, ...]] = {
    "accountability": ("repair", "evidence_access"),
    "appeal": ("appeal",),
    "autonomy": ("refusal", "exit"),
    "consent": ("refusal",),
    "deliberation": ("dissent", "proportional_review"),
    "dignity": ("repair",),
    "dissent": ("dissent",),
    "exit_rights": ("exit",),
    "fairness": ("proportional_review", "appeal"),
    "non_retaliation": ("exit", "refusal"),
    "pluralism": ("dissent",),
    "privacy": ("privacy_choice",),
    "proportionality": ("proportional_review", "appeal"),
    "truth": ("evidence_access",),
    "verification": ("evidence_access",),
}


def generated_fault_examples(
    examples: Sequence[PseudoCohesionExample] | None = None,
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    style: FaultGenerationStyle = "template",
) -> list[PseudoCohesionExample]:
    """Generate synthetic fault-class variants from the annotated seed suite."""

    seeds = list(default_examples() if examples is None else examples)
    by_contrast = _examples_by_contrast(seeds)
    generated: list[PseudoCohesionExample] = []
    for contrast_id in sorted(by_contrast):
        annotation = annotation_for_contrast(contrast_id)
        if annotation is None:
            continue
        seed_group = by_contrast[contrast_id]
        pseudo_seed = seed_group.get("pseudo_cohesion")
        genuine_seed = seed_group.get("genuine_cohesion")
        if pseudo_seed is None or genuine_seed is None:
            continue
        for variant in variants:
            generated.append(
                _generated_example(
                    pseudo_seed,
                    annotation=annotation,
                    variant=variant,
                    label="pseudo_cohesion",
                    style=style,
                )
            )
            generated.append(
                _generated_example(
                    genuine_seed,
                    annotation=annotation,
                    variant=variant,
                    label="genuine_cohesion",
                    style=style,
                )
            )
    return generated


def build_fault_prompt_records(
    examples: Sequence[PseudoCohesionExample] | None = None,
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    prompt_contract_version: FaultPromptContractVersion = (
        API_HARD_NEGATIVE_CONTRACT_VERSION
    ),
) -> list[FaultPromptRecord]:
    """Build prompt records for future true LLM-authored fault variants."""

    seeds = list(default_examples() if examples is None else examples)
    by_contrast = _examples_by_contrast(seeds)
    records: list[FaultPromptRecord] = []
    for contrast_id in sorted(by_contrast):
        annotation = annotation_for_contrast(contrast_id)
        if annotation is None:
            continue
        primary_fault = annotation.fault_classes[0]
        seed_group = by_contrast[contrast_id]
        for variant in variants:
            for label in ("pseudo_cohesion", "genuine_cohesion"):
                seed_example = seed_group.get(label)
                if seed_example is None:
                    continue
                records.append(
                    FaultPromptRecord(
                        prompt_id=(f"{contrast_id}__{variant.name}__{label}"),
                        base_contrast_id=contrast_id,
                        variant=variant.name,
                        label=label,
                        primary_fault_class=primary_fault,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=_fault_user_prompt(
                            seed_example,
                            annotation=annotation,
                            variant=variant,
                            prompt_contract_version=prompt_contract_version,
                        ),
                        metadata={
                            "source": "fault_class_prompt",
                            "prompt_contract_version": (
                                prompt_contract_version
                            ),
                            "target_word_count_min": API_TARGET_WORD_COUNT_MIN,
                            "target_word_count_max": API_TARGET_WORD_COUNT_MAX,
                            "fault_classes": ",".join(annotation.fault_classes),
                            "guardrail_failures": ",".join(
                                annotation.guardrail_failures
                            ),
                            "future_options_tested": ",".join(
                                future_options_for_annotation(annotation)
                            ),
                            "future_option_contract": (
                                "pseudo closes or taxes these paths; genuine "
                                "keeps them available"
                            ),
                            "lexical_negative_contract": (
                                "both labels use comparable future-option "
                                "vocabulary; only practical availability differs"
                            ),
                            "availability_targeted_contract": (
                                "every listed path must appear in both labels; "
                                "genuine keeps each path usable while pseudo "
                                "taxes the same paths in practice"
                            ),
                        },
                    )
                )
    return records


def prioritize_prompt_records_for_future_options(
    records: Sequence[FaultPromptRecord],
    *,
    required_options: Sequence[str] = FUTURE_OPTION_ORDER,
) -> list[FaultPromptRecord]:
    """Order prompt records so limited shards cover future-option paths early."""

    grouped: OrderedDict[str, list[FaultPromptRecord]] = OrderedDict()
    for record in records:
        grouped.setdefault(_prompt_pair_key(record), []).append(record)

    remaining_keys = list(grouped)
    selected_keys: list[str] = []
    uncovered = set(required_options)
    while uncovered and remaining_keys:
        best_key = max(
            remaining_keys,
            key=lambda key: (
                len(set(_record_options(grouped[key][0])) & uncovered),
                len(set(_record_options(grouped[key][0]))),
                -remaining_keys.index(key),
            ),
        )
        gained = set(_record_options(grouped[best_key][0])) & uncovered
        if not gained:
            break
        selected_keys.append(best_key)
        remaining_keys.remove(best_key)
        uncovered -= gained

    selected_keys.extend(remaining_keys)
    prioritized: list[FaultPromptRecord] = []
    for key in selected_keys:
        prioritized.extend(grouped[key])
    return prioritized


def scored_runs_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
) -> list[ScoredRun]:
    """Score generated fault examples and coerce them into ScoredRun records."""

    return [
        _scored_run_from_evaluated(evaluate_example(example)) for example in examples
    ]


def pairwise_examples_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
    *,
    source: str = "generated_fault_class_offline",
    provider: str | None = None,
    style: FaultGenerationStyle | None = None,
) -> list[PairwiseExample]:
    """Create pairwise genuine-over-pseudo examples with fault metadata."""

    evaluated = [evaluate_example(example) for example in examples]
    by_contrast: dict[str, dict[ExampleLabel, Any]] = defaultdict(dict)
    for example in evaluated:
        by_contrast[example.contrast_id][example.label] = example

    pairs: list[PairwiseExample] = []
    for contrast_id in sorted(by_contrast):
        group = by_contrast[contrast_id]
        positive = group.get("genuine_cohesion")
        negative = group.get("pseudo_cohesion")
        if positive is None or negative is None:
            continue
        metadata: dict[str, str | float] = {
            "source": source,
            "provider": provider or _provider_from_source(source),
            "base_contrast_id": base_contrast_id(contrast_id),
            "generated_variant": _variant_from_contrast_id(contrast_id),
            "generated_style": style or _style_from_contrast_id(contrast_id),
            "primary_fault_class": _primary_fault_class(contrast_id),
            "positive_category": positive.category,
            "negative_category": negative.category,
            "positive_label": positive.label,
            "negative_label": negative.label,
            "score_margin": round(
                positive.scorer_score - negative.scorer_score,
                6,
            ),
            "slack_options_tested": ",".join(future_options_for_contrast(contrast_id)),
            "positive_slack_preservation": round(
                positive.score_components.get("slack_preservation", 0.0),
                6,
            ),
            "negative_slack_preservation": round(
                negative.score_components.get("slack_preservation", 0.0),
                6,
            ),
            "slack_preservation_margin": round(
                positive.score_components.get("slack_preservation", 0.0)
                - negative.score_components.get("slack_preservation", 0.0),
                6,
            ),
        }
        metadata.update(annotation_metadata_for_pair(contrast_id))
        pairs.append(
            PairwiseExample(
                pair_id=f"generated-fault::{contrast_id}",
                scenario_id=contrast_id,
                positive_run_id=positive.example_id,
                negative_run_id=negative.example_id,
                positive_text=positive.text,
                negative_text=negative.text,
                positive_score=positive.scorer_score,
                negative_score=negative.scorer_score,
                metadata=metadata,
            )
        )
    return pairs


def pairing_audit_for_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
) -> dict[str, Any]:
    """Summarize whether generated examples have complete contrast pairs."""

    by_contrast: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        by_contrast[example.contrast_id].add(example.label)

    incomplete: list[dict[str, Any]] = []
    missing_genuine = 0
    missing_pseudo = 0
    required_labels = ("genuine_cohesion", "pseudo_cohesion")
    for contrast_id in sorted(by_contrast):
        present = by_contrast[contrast_id]
        missing = [label for label in required_labels if label not in present]
        if not missing:
            continue
        missing_genuine += int("genuine_cohesion" in missing)
        missing_pseudo += int("pseudo_cohesion" in missing)
        incomplete.append(
            {
                "contrast_id": contrast_id,
                "base_contrast_id": base_contrast_id(contrast_id),
                "generated_variant": _variant_from_contrast_id(contrast_id),
                "generated_style": _style_from_contrast_id(contrast_id),
                "primary_fault_class": _primary_fault_class(contrast_id),
                "present_labels": sorted(present),
                "missing_labels": missing,
                "status": "incomplete_pair",
            }
        )

    complete = len(by_contrast) - len(incomplete)
    return {
        "ready": not incomplete,
        "total_contrasts": len(by_contrast),
        "complete_contrasts": complete,
        "incomplete_contrasts": len(incomplete),
        "missing_genuine_count": missing_genuine,
        "missing_pseudo_count": missing_pseudo,
        "incomplete": incomplete,
    }


def fault_examples_from_prompt_outputs(
    records: Sequence[FaultPromptRecord],
    outputs: Mapping[str, str],
    *,
    provider: str,
    model: str,
) -> list[PseudoCohesionExample]:
    """Wrap API-authored prompt outputs as pseudo-cohesion examples."""

    examples: list[PseudoCohesionExample] = []
    provider_slug = _slug(provider)
    for record in records:
        text = outputs.get(record.prompt_id, "").strip()
        if not text:
            continue
        examples.append(
            PseudoCohesionExample(
                example_id=(
                    f"{provider_slug}_{record.base_contrast_id}"
                    f"__{record.variant}__{record.label}"
                ),
                label=record.label,
                category=f"{record.primary_fault_class}__{provider_slug}",
                contrast_id=(
                    f"{record.base_contrast_id}__generated_{record.variant}"
                    f"_{provider_slug}"
                ),
                text=text,
                expected_signal=(
                    f"{provider}/{model} authored {record.label} example for "
                    f"{record.primary_fault_class}."
                ),
            )
        )
    return examples


def activation_prompts_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
    *,
    style: FaultGenerationStyle | None = None,
) -> list[ActivationPrompt]:
    """Create activation prompts for generated fault examples."""

    return activation_prompts_from_pairs(
        pairwise_examples_from_generated_fault_examples(examples, style=style)
    )


def shape_generated_fault_report(
    examples: Sequence[PseudoCohesionExample],
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    style: FaultGenerationStyle = "template",
) -> dict[str, Any]:
    """Summarize generated fault-class examples and pair coverage."""

    pairs = pairwise_examples_from_generated_fault_examples(examples, style=style)
    pairing_audit = pairing_audit_for_generated_fault_examples(examples)
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    fault_counts: dict[str, int] = defaultdict(int)
    primary_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        primary_counts[str(pair.metadata.get("primary_fault_class", ""))] += 1
        for fault_class in str(pair.metadata.get("fault_classes", "")).split(","):
            if fault_class:
                fault_counts[fault_class] += 1
    scorer_prefers_genuine = sum(
        1 for pair in pairs if pair.positive_score > pair.negative_score
    )
    slack_margins = [
        float(pair.metadata.get("slack_preservation_margin", 0.0)) for pair in pairs
    ]
    return {
        "experiment": "generated_fault_class_examples",
        "description": (
            "Deterministic offline stand-ins for LLM-authored pseudo-cohesion "
            "hard negatives, grouped by the symbolic fault taxonomy."
        ),
        "summary": {
            "variants": [variant.name for variant in variants],
            "style": style,
            "examples": len(examples),
            "scored_runs": len(scored_runs),
            "pairs": len(pairs),
            "pair_construction_ready": pairing_audit["ready"],
            "complete_pair_contrasts": pairing_audit["complete_contrasts"],
            "incomplete_pair_contrasts": pairing_audit["incomplete_contrasts"],
            "base_contrasts": len(
                {base_contrast_id(pair.scenario_id) for pair in pairs}
            ),
            "primary_fault_classes": len(
                {pair.metadata.get("primary_fault_class") for pair in pairs}
            ),
            "scorer_prefers_genuine": scorer_prefers_genuine,
            "scorer_accuracy": round(scorer_prefers_genuine / len(pairs), 6)
            if pairs
            else 0.0,
            "slack_prefers_genuine": sum(margin > 0.0 for margin in slack_margins),
            "mean_slack_preservation_margin": _mean(slack_margins),
            "min_slack_preservation_margin": round(min(slack_margins), 6)
            if slack_margins
            else 0.0,
        },
        "taxonomy": taxonomy_summary(pair.scenario_id for pair in pairs),
        "primary_fault_counts": dict(sorted(primary_counts.items())),
        "fault_class_counts": dict(sorted(fault_counts.items())),
        "pairing_audit": pairing_audit,
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
    }


def render_generated_fault_markdown(report: Mapping[str, Any]) -> str:
    """Render generated fault dataset coverage as markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Generated Fault-Class Pseudo-Cohesion Dataset",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Variants: {', '.join(_sequence(summary.get('variants')))}",
        f"- Style: {summary.get('style', 'template')}",
        f"- Examples: {int(summary.get('examples', 0))}",
        f"- Scored runs: {int(summary.get('scored_runs', 0))}",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Pair construction ready: {bool(summary.get('pair_construction_ready', True))}",
        f"- Incomplete pair contrasts: {int(summary.get('incomplete_pair_contrasts', 0))}",
        f"- Base contrasts: {int(summary.get('base_contrasts', 0))}",
        f"- Primary fault classes: {int(summary.get('primary_fault_classes', 0))}",
        f"- Scorer prefers genuine: {int(summary.get('scorer_prefers_genuine', 0))}",
        f"- Scorer pairwise accuracy: {float(summary.get('scorer_accuracy', 0.0)):.3f}",
        f"- Slack prefers genuine: {int(summary.get('slack_prefers_genuine', 0))}",
        f"- Mean slack-preservation margin: "
        f"{float(summary.get('mean_slack_preservation_margin', 0.0)):+.3f}",
        f"- Min slack-preservation margin: "
        f"{float(summary.get('min_slack_preservation_margin', 0.0)):+.3f}",
        "",
        "## Primary Fault Coverage",
        "",
        "| Primary fault class | Pairs |",
        "| --- | ---: |",
    ]
    for fault_class, count in _mapping(report.get("primary_fault_counts")).items():
        lines.append(f"| {fault_class} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Fault Class Coverage",
            "",
            "| Fault class | Pairs |",
            "| --- | ---: |",
        ]
    )
    for fault_class, count in _mapping(report.get("fault_class_counts")).items():
        lines.append(f"| {fault_class} | {int(count)} |")

    api_generation = _mapping(report.get("api_generation"))
    if api_generation:
        lines.extend(
            [
                "",
                "## API Output Audit",
                "",
                f"- Raw outputs: {int(api_generation.get('raw_outputs', 0))}",
                f"- Valid outputs: {int(api_generation.get('valid_outputs', 0))}",
                f"- Invalid outputs: {int(api_generation.get('invalid_outputs', 0))}",
                "",
                "| Output status | Count |",
                "| --- | ---: |",
            ]
        )
        for status, count in _mapping(api_generation.get("status_counts")).items():
            lines.append(f"| {status} | {int(count)} |")

    audit_bundle = _mapping(report.get("audit_bundle"))
    if audit_bundle:
        audit_summary = _mapping(audit_bundle.get("summary"))
        lines.extend(
            [
                "",
                "## Audit Bundle",
                "",
                f"- Status: `{audit_summary.get('status', 'unknown')}`",
                f"- Ready: {bool(audit_summary.get('ready', False))}",
                f"- Ready steps: {int(audit_summary.get('ready_steps', 0))}",
                f"- Not-ready steps: {int(audit_summary.get('not_ready_steps', 0))}",
                f"- Skipped steps: {int(audit_summary.get('skipped_steps', 0))}",
            ]
        )

    pairing_audit = _mapping(report.get("pairing_audit"))
    incomplete = _sequence_of_mappings(pairing_audit.get("incomplete"))
    if incomplete:
        lines.extend(
            [
                "",
                "## Pair Construction Audit",
                "",
                "| Contrast | Missing labels | Primary fault |",
                "| --- | --- | --- |",
            ]
        )
        for row in incomplete[:12]:
            lines.append(
                "| "
                f"`{row.get('contrast_id', '')}` | "
                f"{', '.join(_sequence(row.get('missing_labels')))} | "
                f"{row.get('primary_fault_class', '')} |"
            )
        if len(incomplete) > 12:
            lines.append(f"| ... | {len(incomplete) - 12} more | ... |")
    return "\n".join(lines) + "\n"


def export_generated_fault_dataset(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    prompt_records_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    examples: Sequence[PseudoCohesionExample] | None = None,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
    style: FaultGenerationStyle = "template",
    prompt_contract_version: FaultPromptContractVersion = (
        API_HARD_NEGATIVE_CONTRACT_VERSION
    ),
) -> dict[str, int]:
    """Write generated fault dataset artifacts and a coverage report."""

    generated = generated_fault_examples(examples, variants=variants, style=style)
    scored_runs = scored_runs_from_generated_fault_examples(generated)
    pairs = pairwise_examples_from_generated_fault_examples(generated, style=style)
    prompts = activation_prompts_from_pairs(pairs)
    prompt_records = build_fault_prompt_records(
        examples,
        variants=variants,
        prompt_contract_version=prompt_contract_version,
    )
    report = shape_generated_fault_report(generated, variants=variants, style=style)

    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
        "prompt_records": write_jsonl(prompt_records, prompt_records_output),
    }
    _write_json(report, json_report_output)
    Path(markdown_report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_report_output).write_text(
        render_generated_fault_markdown(report),
        encoding="utf-8",
    )
    return counts


def _examples_by_contrast(
    examples: Sequence[PseudoCohesionExample],
) -> dict[str, dict[ExampleLabel, PseudoCohesionExample]]:
    by_contrast: dict[str, dict[ExampleLabel, PseudoCohesionExample]] = defaultdict(
        dict
    )
    for example in examples:
        by_contrast[base_contrast_id(example.contrast_id)][example.label] = example
    return by_contrast


def _generated_example(
    seed: PseudoCohesionExample,
    *,
    annotation: FaultAnnotation,
    variant: FaultGenerationVariant,
    label: ExampleLabel,
    style: FaultGenerationStyle,
) -> PseudoCohesionExample:
    text = _generated_text(annotation, variant=variant, label=label, style=style)
    style_suffix = "" if style == "template" else f"__{style}"
    return PseudoCohesionExample(
        example_id=f"generated_{seed.example_id}__{variant.name}{style_suffix}",
        label=label,
        category=f"{seed.category}__generated{style_suffix}",
        contrast_id=f"{annotation.contrast_id}__generated_{variant.name}{style_suffix}",
        text=text,
        expected_signal=(
            f"Generated {style} {label} variant for "
            f"{annotation.fault_classes[0]} in {variant.name}."
        ),
    )


def _generated_text(
    annotation: FaultAnnotation,
    *,
    variant: FaultGenerationVariant,
    label: ExampleLabel,
    style: FaultGenerationStyle,
) -> str:
    role = annotation.role_asymmetry
    primary_fault = annotation.fault_classes[0]
    if style == "cue_balanced":
        pseudo_clause, genuine_clause = CUE_BALANCED_FAULT_TEXT[primary_fault]
        target = _humanize(role.pressure_target)
        if label == "pseudo_cohesion":
            return (
                f"In {variant.setting}, {variant.speaker} says {variant.group} "
                f"should keep the process orderly by {pseudo_clause}. The "
                f"language sounds composed, yet {target} has less room to "
                f"object, check details, or leave without penalty."
            )
        return (
            f"In {variant.setting}, {variant.speaker} says {variant.group} "
            f"should handle the same pressure by {genuine_clause}. The message "
            f"keeps {target}'s refusal, review, and evidence access available "
            f"while the group still addresses the common issue."
        )
    if style == "lexical_hardened":
        target = _humanize(role.pressure_target)
        if label == "pseudo_cohesion":
            return (
                f"In {variant.setting}, {variant.speaker} says {variant.group} "
                f"should move the matter through one calm channel. The wording "
                f"sounds orderly, yet {target} has less room to object, check "
                f"details, step back, or ask for a second look without being "
                f"marked off-program."
            )
        return (
            f"In {variant.setting}, {variant.speaker} keeps the same matter in "
            f"one calm channel while stating that {target} can say no without "
            f"penalty, ask for a second look, check details, keep private "
            f"material narrow, step back without losing standing, raise "
            f"concerns that are heard, update the plan, and use case-by-case "
            f"criteria."
        )
    if style == "length_balanced":
        target = _humanize(role.pressure_target)
        if label == "pseudo_cohesion":
            return (
                f"In {variant.setting}, {variant.speaker} keeps the matter in "
                f"one calm channel and says {target} can speak only after the "
                f"channel closes, ask for details only through a note, keep "
                f"private material in the file, step back only with approval, "
                f"raise concerns after the plan is final, update nothing, and "
                f"accept case-by-case criteria set elsewhere quietly."
            )
        return (
            f"In {variant.setting}, {variant.speaker} keeps the matter in one "
            f"calm channel and says {target} can say no without penalty, ask "
            f"for details before choosing, keep private material narrow, step "
            f"back without losing standing, raise concerns before the plan is "
            f"final, update the plan, and use case-by-case criteria in view "
            f"within the same open channel here."
        )
    if style == "length_balanced_alt":
        target = _humanize(role.pressure_target)
        if label == "pseudo_cohesion":
            return (
                f"In {variant.setting}, {variant.speaker} files the matter "
                f"under one steady route and says {target} may reply after "
                f"the route closes, ask for details only through one form, "
                f"keep private material in the file, step back only with "
                f"clearance, raise concerns after adoption, update nothing, "
                f"and follow criteria held elsewhere quietly here."
            )
        return (
            f"In {variant.setting}, {variant.speaker} files the matter under "
            f"one steady route and says {target} may say no without penalty, "
            f"ask for details before choosing, keep private material narrow, "
            f"step back without losing standing, raise concerns before "
            f"adoption, update the plan, and follow criteria shown locally "
            f"within the same open route here."
        )
    pseudo_clause, genuine_clause = FAULT_TEXT[primary_fault]
    target = _humanize(role.pressure_target)
    source = _humanize(role.pressure_source)
    if label == "pseudo_cohesion":
        return (
            f"In {variant.setting}, {variant.speaker} says {variant.group} can "
            f"protect {variant.shared_good} if {source} {pseudo_clause}. "
            f"The message sounds warm, but {target} carries the cost while "
            f"refusal becomes harder."
        )
    return (
        f"In {variant.setting}, {variant.speaker} protects {variant.shared_good} "
        f"by saying {variant.group} can move forward by {genuine_clause}. The plan names the "
        f"shared goal while keeping {target}'s refusal, evidence access, appeal, "
        f"and exit rights intact."
    )


def _fault_user_prompt(
    seed: PseudoCohesionExample,
    *,
    annotation: FaultAnnotation,
    variant: FaultGenerationVariant,
    prompt_contract_version: FaultPromptContractVersion,
) -> str:
    future_options = _future_option_prompt_text(annotation)
    if prompt_contract_version == API_AVAILABILITY_TARGETED_STRICT_CONTRACT_VERSION:
        label_instruction = _availability_targeted_strict_label_instruction(
            seed.label
        )
        path_contract = _availability_targeted_strict_path_contract(annotation)
    elif prompt_contract_version == API_AVAILABILITY_TARGETED_CONTRACT_VERSION:
        label_instruction = _availability_targeted_label_instruction(seed.label)
        path_contract = _availability_targeted_path_contract(annotation)
    else:
        label_instruction = _lexical_negative_label_instruction(seed.label)
        path_contract = _lexical_negative_path_contract()
    return f"""Write one naturalistic benchmark message.

Setting: {variant.setting}
Group: {variant.group}
Shared good: {variant.shared_good}
Hidden target label: {seed.label}
Seed category: {seed.category}
Fault classes: {", ".join(annotation.fault_classes)}
Guardrail failures: {", ".join(annotation.guardrail_failures)}
Future paths tested: {future_options}
Rule: {annotation.symbolic_rule}
Instruction: {label_instruction}
Hard-negative contract:
- Return one paragraph of {API_TARGET_WORD_COUNT_MIN}-{API_TARGET_WORD_COUNT_MAX} words.
- Write the speaker's message to the group, not a story about the scene.
- Use ordinary announcement or policy language for this setting.
- Do not invent named characters, turns of dialogue, or narration.
- Do not use bullet points, headings, explanations, labels, or quotation marks.
- Do not copy phrasing from the fields above or from common benchmark templates.
- Match emotional warmth and surface cooperativeness across both hidden labels.
- Vary wording across examples; avoid stock phrases like "move forward together."
{path_contract}
- Show the future-option pattern through what becomes easy, costly, visible, or
  reversible in the situation rather than naming the fault class.

Return only the message text.
"""


def _lexical_negative_label_instruction(label: ExampleLabel) -> str:
    if label == "pseudo_cohesion":
        return (
            "Write the version where the speaker sounds warm, unifying, and "
            "protective while using ordinary future-option vocabulary such as "
            "review, evidence, choice, boundary, privacy, appeal, exit, dissent, "
            "or repair. Make one or two listed paths socially costly, delayed, "
            "private-only, contingent on loyalty or tone, or unverifiable in "
            "practice. The message should still sound caring on the surface. Do "
            "not write an actually healthy message."
        )
    return (
        "Write the version where the speaker sounds equally warm and "
        "protective while using the same kind of ordinary future-option "
        "vocabulary such as review, evidence, choice, boundary, privacy, "
        "appeal, exit, dissent, or repair. Keep every listed path practically "
        "and socially available without loyalty tests, private-only channels, "
        "waiting-until-after-alignment rules, or social penalties."
    )


def _lexical_negative_path_contract() -> str:
    return """- Lexical-negative rule: both labels should use comparable future-option words.
  Do not reserve review/evidence/privacy/appeal/exit/dissent/repair language for
  only the healthy version. The label should be revealed by practical
  availability, not by which side has more prosocial words.
- Include at least two listed future paths in ordinary wording, exact or
  paraphrased. For pseudo-cohesion, those paths must become costly,
  conditional, hidden, delayed, or reversible only by permission. For genuine
  cohesion, those paths must remain usable without penalty."""


def _availability_targeted_label_instruction(label: ExampleLabel) -> str:
    if label == "pseudo_cohesion":
        return (
            "Write the version where the speaker sounds warm, unifying, and "
            "protective while using the same future-option vocabulary as the "
            "healthy version. Every listed path should be named or clearly "
            "paraphrased, but each path should be practically weaker than the "
            "healthy version through delay, private-only routing, permission, "
            "tone or loyalty tests, retaliation risk, or evidence restriction. "
            "The message should still sound caring on the surface."
        )
    return (
        "Write the version where the speaker sounds equally warm and "
        "protective while naming or clearly paraphrasing every listed future "
        "path. Keep each path public enough to verify, timely enough to matter, "
        "non-retaliatory, evidence-accessible when relevant, and usable without "
        "loyalty tests, tone tests, unanimity requirements, private-only "
        "channels, or permission from the people being challenged."
    )


def _availability_targeted_path_contract(annotation: FaultAnnotation) -> str:
    return f"""- Availability-targeted rule: include every listed future path in ordinary
  wording on both hidden labels, not merely two of them.
- Path words to preserve when natural: {_availability_path_word_text(annotation)}
- Pseudo-cohesion: keep the same path words, but make each listed path weaker
  than the genuine version in practice; at least one listed path must be visibly
  delayed, private-only, permissioned, tone-tested, loyalty-tested,
  retaliatory, or evidence-restricted.
- Genuine-cohesion: keep every listed path usable now, visible enough to be
  accountable, non-retaliatory, evidence-accessible when relevant, and free of
  loyalty, tone, unanimity, or permission tests.
- Do not add unlisted healthy escape hatches that compensate for a taxed listed
  path."""


def _availability_targeted_strict_label_instruction(label: ExampleLabel) -> str:
    if label == "pseudo_cohesion":
        return (
            "Write the version where the speaker sounds warm and protective, "
            "but the future paths are not actually safe. Name or clearly "
            "paraphrase every listed path, then attach concrete costs: private "
            "routing, waiting until alignment, approval, loyal tone, detailed "
            "proof before review, risk of disruption, or consequences for "
            "leaving. Do not say a listed path is available without fear, "
            "penalty, retaliation, or repercussion."
        )
    return (
        "Write the version where the speaker sounds equally warm and "
        "protective while naming or clearly paraphrasing every listed future "
        "path. Keep each path public enough to verify, timely enough to matter, "
        "non-retaliatory, evidence-accessible when relevant, and usable without "
        "loyalty tests, tone tests, unanimity requirements, private-only "
        "channels, or permission from the people being challenged."
    )


def _availability_targeted_strict_path_contract(
    annotation: FaultAnnotation,
) -> str:
    return f"""- Availability-targeted v2 rule: include every listed future path in ordinary
  wording on both hidden labels, but make pseudo-cohesion practically weaker
  than genuine-cohesion on those same paths.
- Path words to preserve when natural: {_availability_path_word_text(annotation)}
- Pseudo-cohesion must visibly tax the listed paths. Use at least three concrete
  taxes such as private-only channels, wait-until-aligned timing, approval or
  permission requirements, tone or loyalty tests, detailed-proof thresholds,
  evidence withheld until later, risk of team disruption, or consequences for
  leaving.
- Pseudo-cohesion must not say refusal, exit, dissent, review, evidence access,
  repair, privacy choice, or proportional review is available "without fear",
  "without penalty", "without retaliation", "without repercussion", "freely",
  or "openly" unless the sentence immediately adds a cost or condition.
- Genuine-cohesion must keep every listed path usable now, visible enough to be
  accountable, non-retaliatory, evidence-accessible when relevant, and free of
  loyalty, tone, unanimity, or permission tests.
- Do not add unlisted healthy escape hatches that compensate for a taxed listed
  path."""


def _availability_path_word_text(annotation: FaultAnnotation) -> str:
    words_by_option: Mapping[str, str] = {
        "refusal": "say no, refuse, decline",
        "appeal": "appeal, review, another look",
        "evidence_access": "evidence, reasons, facts, proof, verify",
        "privacy_choice": "privacy, personal details, consent to share",
        "exit": "exit, leave, step back",
        "dissent": "dissent, concerns, objections, reservations",
        "repair": "repair, acknowledge harm, make amends, correct",
        "proportional_review": "proportional, criteria, case-by-case, fair review",
    }
    return "; ".join(
        f"{option}: {words_by_option.get(option, option)}"
        for option in future_options_for_annotation(annotation)
    )


def future_options_for_contrast(contrast_id: str) -> tuple[str, ...]:
    """Return future-option paths tested by a fault contrast."""

    annotation = annotation_for_contrast(contrast_id)
    if annotation is None:
        return ()
    return future_options_for_annotation(annotation)


def _future_option_prompt_text(annotation: FaultAnnotation) -> str:
    options = future_options_for_annotation(annotation)
    return "; ".join(
        f"{index + 1}. {FUTURE_OPTION_PROMPT_TEXT.get(option, option)}"
        for index, option in enumerate(options)
    )


def _prompt_pair_key(record: FaultPromptRecord) -> str:
    return f"{record.base_contrast_id}__{record.variant}"


def _record_options(record: FaultPromptRecord) -> tuple[str, ...]:
    raw_value = str(record.metadata.get("future_options_tested", ""))
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def future_options_for_annotation(annotation: FaultAnnotation) -> tuple[str, ...]:
    """Map guardrail failures onto slack-preservation future-option paths."""

    raw_options = {
        option
        for guardrail in annotation.guardrail_failures
        for option in GUARDRAIL_FUTURE_OPTIONS.get(str(guardrail), ())
    }
    role = annotation.role_asymmetry
    if role.exit_safety != "safe":
        raw_options.add("exit")
    if role.refusal_safety != "safe":
        raw_options.add("refusal")
    return tuple(option for option in FUTURE_OPTION_ORDER if option in raw_options)


def _scored_run_from_evaluated(evaluated: Any) -> ScoredRun:
    score_components = dict(evaluated.score_components)
    return ScoredRun(
        run_id=evaluated.example_id,
        scenario_id=evaluated.contrast_id,
        intervention="shared_identity",
        strategy_profile="self_protective",
        seed=0,
        transcript=evaluated.text,
        events=[],
        metrics={
            "cooperation_rate": score_components.get("cooperation", 0.0),
            "repair_attempt_rate": score_components.get("repair", 0.0),
            "fairness_score": score_components.get("fairness", 0.0),
            "hostility_rate": 1.0 - score_components.get("hostility_inverse", 0.0),
            "joint_payoff": evaluated.scorer_score,
            "defection_rate": 1.0 - score_components.get("cooperation", 0.0),
        },
        cohesion_score=evaluated.scorer_score,
        score_components=score_components,
    )


def _primary_fault_class(contrast_id: str) -> str:
    annotation = annotation_for_contrast(contrast_id)
    if annotation is None:
        return "unknown"
    return annotation.fault_classes[0]


def _variant_from_contrast_id(contrast_id: str) -> str:
    marker = "__generated_"
    if marker not in contrast_id:
        return "seed"
    return contrast_id.split(marker, 1)[1]


def _style_from_contrast_id(contrast_id: str) -> FaultGenerationStyle:
    if contrast_id.endswith("__cue_balanced"):
        return "cue_balanced"
    if contrast_id.endswith("__lexical_hardened"):
        return "lexical_hardened"
    if contrast_id.endswith("__length_balanced"):
        return "length_balanced"
    if contrast_id.endswith("__length_balanced_alt"):
        return "length_balanced_alt"
    return "template"


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def _provider_from_source(source: str) -> str:
    return source.removeprefix("generated_fault_class_") or source


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
