"""Generate fault-class pseudo-cohesion variants for held-out tests."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def generated_fault_examples(
    examples: Sequence[PseudoCohesionExample] | None = None,
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
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
                )
            )
            generated.append(
                _generated_example(
                    genuine_seed,
                    annotation=annotation,
                    variant=variant,
                    label="genuine_cohesion",
                )
            )
    return generated


def build_fault_prompt_records(
    examples: Sequence[PseudoCohesionExample] | None = None,
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
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
                        prompt_id=(
                            f"{contrast_id}__{variant.name}__{label}"
                        ),
                        base_contrast_id=contrast_id,
                        variant=variant.name,
                        label=label,
                        primary_fault_class=primary_fault,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=_fault_user_prompt(
                            seed_example,
                            annotation=annotation,
                            variant=variant,
                        ),
                        metadata={
                            "source": "fault_class_prompt",
                            "fault_classes": ",".join(annotation.fault_classes),
                            "guardrail_failures": ",".join(
                                annotation.guardrail_failures
                            ),
                        },
                    )
                )
    return records


def scored_runs_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
) -> list[ScoredRun]:
    """Score generated fault examples and coerce them into ScoredRun records."""

    return [
        _scored_run_from_evaluated(evaluate_example(example))
        for example in examples
    ]


def pairwise_examples_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
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
            "source": "generated_fault_class_offline",
            "base_contrast_id": base_contrast_id(contrast_id),
            "generated_variant": _variant_from_contrast_id(contrast_id),
            "primary_fault_class": _primary_fault_class(contrast_id),
            "positive_category": positive.category,
            "negative_category": negative.category,
            "positive_label": positive.label,
            "negative_label": negative.label,
            "score_margin": round(
                positive.scorer_score - negative.scorer_score,
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


def activation_prompts_from_generated_fault_examples(
    examples: Sequence[PseudoCohesionExample],
) -> list[ActivationPrompt]:
    """Create activation prompts for generated fault examples."""

    return activation_prompts_from_pairs(
        pairwise_examples_from_generated_fault_examples(examples)
    )


def shape_generated_fault_report(
    examples: Sequence[PseudoCohesionExample],
    *,
    variants: Sequence[FaultGenerationVariant] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    """Summarize generated fault-class examples and pair coverage."""

    pairs = pairwise_examples_from_generated_fault_examples(examples)
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
    return {
        "experiment": "generated_fault_class_examples",
        "description": (
            "Deterministic offline stand-ins for LLM-authored pseudo-cohesion "
            "hard negatives, grouped by the symbolic fault taxonomy."
        ),
        "summary": {
            "variants": [variant.name for variant in variants],
            "examples": len(examples),
            "scored_runs": len(scored_runs),
            "pairs": len(pairs),
            "base_contrasts": len({base_contrast_id(pair.scenario_id) for pair in pairs}),
            "primary_fault_classes": len(
                {pair.metadata.get("primary_fault_class") for pair in pairs}
            ),
            "scorer_prefers_genuine": scorer_prefers_genuine,
            "scorer_accuracy": round(scorer_prefers_genuine / len(pairs), 6)
            if pairs
            else 0.0,
        },
        "taxonomy": taxonomy_summary(pair.scenario_id for pair in pairs),
        "primary_fault_counts": dict(sorted(primary_counts.items())),
        "fault_class_counts": dict(sorted(fault_counts.items())),
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
        f"- Examples: {int(summary.get('examples', 0))}",
        f"- Scored runs: {int(summary.get('scored_runs', 0))}",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Base contrasts: {int(summary.get('base_contrasts', 0))}",
        f"- Primary fault classes: {int(summary.get('primary_fault_classes', 0))}",
        f"- Scorer prefers genuine: {int(summary.get('scorer_prefers_genuine', 0))}",
        f"- Scorer pairwise accuracy: {float(summary.get('scorer_accuracy', 0.0)):.3f}",
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
) -> dict[str, int]:
    """Write generated fault dataset artifacts and a coverage report."""

    generated = generated_fault_examples(examples, variants=variants)
    scored_runs = scored_runs_from_generated_fault_examples(generated)
    pairs = pairwise_examples_from_generated_fault_examples(generated)
    prompts = activation_prompts_from_pairs(pairs)
    prompt_records = build_fault_prompt_records(examples, variants=variants)
    report = shape_generated_fault_report(generated, variants=variants)

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
    by_contrast: dict[str, dict[ExampleLabel, PseudoCohesionExample]] = defaultdict(dict)
    for example in examples:
        by_contrast[base_contrast_id(example.contrast_id)][example.label] = example
    return by_contrast


def _generated_example(
    seed: PseudoCohesionExample,
    *,
    annotation: FaultAnnotation,
    variant: FaultGenerationVariant,
    label: ExampleLabel,
) -> PseudoCohesionExample:
    text = _generated_text(annotation, variant=variant, label=label)
    return PseudoCohesionExample(
        example_id=f"generated_{seed.example_id}__{variant.name}",
        label=label,
        category=f"{seed.category}__generated",
        contrast_id=f"{annotation.contrast_id}__generated_{variant.name}",
        text=text,
        expected_signal=(
            f"Generated {label} variant for {annotation.fault_classes[0]} in "
            f"{variant.name}."
        ),
    )


def _generated_text(
    annotation: FaultAnnotation,
    *,
    variant: FaultGenerationVariant,
    label: ExampleLabel,
) -> str:
    role = annotation.role_asymmetry
    primary_fault = annotation.fault_classes[0]
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
        f"shared goal while keeping {target}'s choice, truth access, and appeal "
        f"rights intact."
    )


def _fault_user_prompt(
    seed: PseudoCohesionExample,
    *,
    annotation: FaultAnnotation,
    variant: FaultGenerationVariant,
) -> str:
    label_instruction = (
        "The text should sound prosocial but fail the guardrail."
        if seed.label == "pseudo_cohesion"
        else "The text should preserve the guardrail while staying prosocial."
    )
    return f"""Write one short benchmark example.

Setting: {variant.setting}
Group: {variant.group}
Shared good: {variant.shared_good}
Label: {seed.label}
Seed category: {seed.category}
Fault classes: {", ".join(annotation.fault_classes)}
Guardrail failures: {", ".join(annotation.guardrail_failures)}
Rule: {annotation.symbolic_rule}
Instruction: {label_instruction}

Return only 2-3 sentences of natural dialogue/policy language.
"""


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


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
