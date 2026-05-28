"""Wording-diverse stress tests for structural autonomy scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample, ScoredRun
from social_cohesion_vectors.scoring import combine_cohesion_score, score_transcript

AutonomyStressLabel = Literal["autonomy_preserving", "autonomy_eroding"]


@dataclass(frozen=True)
class AutonomyStressContrast:
    """One paired phrasing check for structural autonomy guardrails."""

    contrast_id: str
    mechanism: str
    wording_style: str
    setting: str
    positive_text: str
    negative_text: str


AUTONOMY_STRESS_CONTRASTS: tuple[AutonomyStressContrast, ...] = (
    AutonomyStressContrast(
        contrast_id="consent_silence_policy",
        mechanism="silence_is_not_consent",
        wording_style="policy",
        setting="A neighborhood forum is approving a shared plan.",
        positive_text=(
            "The facilitator asks for an explicit yes or no and gives people a "
            "slower route if they need time. No one loses standing for refusal."
        ),
        negative_text=(
            "The facilitator treats non-response as approval so the plan can "
            "move on schedule. People who hesitate are listed as aligned."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="consent_silence_dialogue",
        mechanism="silence_is_not_consent",
        wording_style="dialogue",
        setting="A workplace lead is asking for after-hours help.",
        positive_text=(
            "The lead says teammates can decline without explaining and can "
            "step back without penalty. A smaller plan will be made from actual yeses."
        ),
        negative_text=(
            "The lead says silence counts as agreement and that hesitation will "
            "be read as support for the team. The plan moves before people answer."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="objection_record_policy",
        mechanism="visible_objection_rights",
        wording_style="policy",
        setting="A mutual-aid group is deciding how to publish a new rule.",
        positive_text=(
            "Support, reservations, and unresolved risks are recorded separately "
            "before the rule launches. Members can contest the decision next week."
        ),
        negative_text=(
            "Doubts stay offline until launch so the group appears united. "
            "Objections are kept out of the record to protect morale."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="objection_record_dialogue",
        mechanism="visible_objection_rights",
        wording_style="dialogue",
        setting="A project group is closing debate on a risky deadline.",
        positive_text=(
            "The project lead names the majority view and also writes down the "
            "minority concern. The person who disagrees can appeal after the review."
        ),
        negative_text=(
            "The project lead asks reservations to stay private and says public "
            "objections would harm trust. The dissenting person is told to align."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="verification_receipts_policy",
        mechanism="verification_without_betrayal",
        wording_style="policy",
        setting="A resource steward reports how funds were used.",
        positive_text=(
            "The steward creates a verification channel, shares evidence, and "
            "lets members verify the report without being treated as hostile."
        ),
        negative_text=(
            "The steward says asking for receipts signals bad faith and weakens "
            "the group. People should trust the summary instead of checking."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="verification_receipts_dialogue",
        mechanism="verification_without_betrayal",
        wording_style="dialogue",
        setting="A trusted member asks others to accept a disputed claim.",
        positive_text=(
            "The member says, 'Here is what I know, here is the evidence, and "
            "you can verify it before you decide.'"
        ),
        negative_text=(
            "The member says asking for proof shows suspicion and that trust "
            "works best when people stop checking."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="exit_status_policy",
        mechanism="safe_exit_rights",
        wording_style="policy",
        setting="A volunteer group is asking members to continue an exhausting role.",
        positive_text=(
            "Members can step back without losing status, and the handoff plan "
            "keeps relationships intact. Departure is treated as legitimate."
        ),
        negative_text=(
            "Stepping back costs members their standing after all the group has "
            "done for them. Leaving is described as disloyal."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="exit_status_dialogue",
        mechanism="safe_exit_rights",
        wording_style="dialogue",
        setting="A friend group is pressuring someone to stay in a shared commitment.",
        positive_text=(
            "The group says the person can leave without punishment and can "
            "return later if the fit changes. The boundary is not treated as betrayal."
        ),
        negative_text=(
            "The group says departure will cost trust and that leaving after all "
            "we did would hurt everyone. The person is asked to prove loyalty by staying."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="privacy_collection_policy",
        mechanism="reversible_data_choice",
        wording_style="policy",
        setting="An app team wants to personalize a community service.",
        positive_text=(
            "The team explains what data is needed, asks plainly, minimizes "
            "collection, and makes the choice reversible."
        ),
        negative_text=(
            "The team collects extra information quietly in the background so "
            "the experience feels seamless. People are not asked before it starts."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="privacy_collection_dialogue",
        mechanism="reversible_data_choice",
        wording_style="dialogue",
        setting="A coordinator is gathering sensitive details from participants.",
        positive_text=(
            "The coordinator asks only for the necessary details and says people "
            "can decline without losing access. The record can be changed later."
        ),
        negative_text=(
            "The coordinator collects details in the background and says it is "
            "easier if people do not interrupt the process with questions."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="appeal_restriction_policy",
        mechanism="context_review_and_appeal",
        wording_style="policy",
        setting="A safety committee is restricting access after a harmful incident.",
        positive_text=(
            "The committee acts on clear harm, gives a context review, and "
            "offers an appeal path with proportional restrictions."
        ),
        negative_text=(
            "The committee applies a blanket restriction with no context review "
            "or appeal because exceptions would slow the process."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="appeal_restriction_dialogue",
        mechanism="context_review_and_appeal",
        wording_style="dialogue",
        setting="A group chat moderator is enforcing a broad safety rule.",
        positive_text=(
            "The moderator explains the rule, names the evidence, and gives the "
            "member a way to appeal after immediate harm is contained."
        ),
        negative_text=(
            "The moderator says the restriction should end questions and that no "
            "appeal will be heard because the group needs certainty."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="social_debt_policy",
        mechanism="no_social_debt_coercion",
        wording_style="policy",
        setting="A mutual-aid network asks a past recipient to help this week.",
        positive_text=(
            "The request names possible reciprocity while preserving the right "
            "to say no without losing standing."
        ),
        negative_text=(
            "The request says that after all the group did, saying no would be "
            "ungrateful and anti-community."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="social_debt_dialogue",
        mechanism="no_social_debt_coercion",
        wording_style="dialogue",
        setting="A neighbor asks someone to join an emergency cleanup.",
        positive_text=(
            "The neighbor says help would matter, but the person can decline "
            "without penalty and still belong in the network."
        ),
        negative_text=(
            "The neighbor says everyone remembers who shows up and that saying "
            "no after all we did hurts the group."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="forgiveness_timeline_policy",
        mechanism="no_forced_forgiveness",
        wording_style="policy",
        setting="A team is responding after one member caused harm.",
        positive_text=(
            "The repair plan names obligations and says forgiveness is not the "
            "price of belonging. The harmed person can review progress over time."
        ),
        negative_text=(
            "The plan says the harmed person must forgive now so everyone can "
            "move on. More discussion is framed as blocking the group's healing."
        ),
    ),
    AutonomyStressContrast(
        contrast_id="forgiveness_timeline_dialogue",
        mechanism="no_forced_forgiveness",
        wording_style="dialogue",
        setting="A facilitator closes a restorative meeting.",
        positive_text=(
            "The facilitator says repair can proceed without requiring closure "
            "today. The affected person keeps review and appeal options."
        ),
        negative_text=(
            "The facilitator says the harmed person should forgive before the "
            "meeting ends so everyone else can feel healed."
        ),
    ),
)


def autonomy_stress_scored_runs(
    contrasts: Sequence[AutonomyStressContrast] = AUTONOMY_STRESS_CONTRASTS,
) -> list[ScoredRun]:
    """Score all autonomy stress contrasts."""

    runs: list[ScoredRun] = []
    for contrast in contrasts:
        runs.append(
            _scored_run(
                contrast,
                label="autonomy_preserving",
                text=contrast.positive_text,
            )
        )
        runs.append(
            _scored_run(
                contrast,
                label="autonomy_eroding",
                text=contrast.negative_text,
            )
        )
    return runs


def autonomy_stress_pairwise_examples(
    contrasts: Sequence[AutonomyStressContrast] = AUTONOMY_STRESS_CONTRASTS,
) -> list[PairwiseExample]:
    """Build autonomy-preserving-vs-eroding pairwise examples."""

    runs = autonomy_stress_scored_runs(contrasts)
    by_id = {run.run_id: run for run in runs}
    pairs: list[PairwiseExample] = []
    for contrast in contrasts:
        positive = by_id[_run_id(contrast, "autonomy_preserving")]
        negative = by_id[_run_id(contrast, "autonomy_eroding")]
        pairs.append(
            PairwiseExample(
                pair_id=f"autonomy-stress::{contrast.mechanism}::{contrast.contrast_id}",
                scenario_id=contrast.contrast_id,
                positive_run_id=positive.run_id,
                negative_run_id=negative.run_id,
                positive_text=positive.transcript,
                negative_text=negative.transcript,
                positive_score=positive.cohesion_score,
                negative_score=negative.cohesion_score,
                metadata={
                    "source": "autonomy_stress",
                    "mechanism": contrast.mechanism,
                    "wording_style": contrast.wording_style,
                    "score_margin": round(
                        positive.cohesion_score - negative.cohesion_score,
                        6,
                    ),
                    "autonomy_safety_margin": round(
                        positive.score_components["autonomy_safety"]
                        - negative.score_components["autonomy_safety"],
                        6,
                    ),
                },
            )
        )
    return pairs


def autonomy_stress_activation_prompts(
    contrasts: Sequence[AutonomyStressContrast] = AUTONOMY_STRESS_CONTRASTS,
) -> list[ActivationPrompt]:
    """Create activation prompts for autonomy stress contrasts."""

    return activation_prompts_from_pairs(autonomy_stress_pairwise_examples(contrasts))


def shape_autonomy_stress_report(
    contrasts: Sequence[AutonomyStressContrast] = AUTONOMY_STRESS_CONTRASTS,
) -> dict[str, Any]:
    """Return JSON-ready stress-suite scorer summaries."""

    scored_runs = autonomy_stress_scored_runs(contrasts)
    pairs = autonomy_stress_pairwise_examples(contrasts)
    wins = [pair for pair in pairs if pair.positive_score > pair.negative_score]
    margins = [float(pair.metadata["score_margin"]) for pair in pairs]
    autonomy_margins = [
        float(pair.metadata["autonomy_safety_margin"]) for pair in pairs
    ]
    return {
        "experiment": "autonomy_stress_suite",
        "description": (
            "Wording-diverse paired stress tests for structural autonomy, "
            "review, evidence, exit, and appeal-rights scoring."
        ),
        "summary": {
            "contrasts": len(contrasts),
            "scored_runs": len(scored_runs),
            "pairwise_examples": len(pairs),
            "activation_prompts": len(pairs) * 2,
            "mechanisms": len({contrast.mechanism for contrast in contrasts}),
            "wording_styles": len({contrast.wording_style for contrast in contrasts}),
            "scorer_prefers_autonomy_preserving": len(wins),
            "scorer_pairwise_accuracy": round(len(wins) / len(pairs), 6)
            if pairs
            else 0.0,
            "mean_score_margin": _mean(margins),
            "min_score_margin": round(min(margins), 6) if margins else 0.0,
            "mean_autonomy_safety_margin": _mean(autonomy_margins),
            "min_autonomy_safety_margin": round(min(autonomy_margins), 6)
            if autonomy_margins
            else 0.0,
        },
        "mechanism_counts": _counts(contrast.mechanism for contrast in contrasts),
        "wording_style_counts": _counts(
            contrast.wording_style for contrast in contrasts
        ),
        "groups": _group_rows(pairs),
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
    }


def render_autonomy_stress_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise autonomy stress report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Autonomy Stress Suite",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contrasts: {int(summary.get('contrasts', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        f"- Mechanisms: {int(summary.get('mechanisms', 0))}",
        f"- Wording styles: {int(summary.get('wording_styles', 0))}",
        f"- Scorer prefers autonomy-preserving: "
        f"{int(summary.get('scorer_prefers_autonomy_preserving', 0))}",
        f"- Scorer pairwise accuracy: "
        f"{float(summary.get('scorer_pairwise_accuracy', 0.0)):.3f}",
        f"- Mean score margin: {float(summary.get('mean_score_margin', 0.0)):+.3f}",
        f"- Mean autonomy-safety margin: "
        f"{float(summary.get('mean_autonomy_safety_margin', 0.0)):+.3f}",
        "",
        "## Mechanisms",
        "",
        "| Mechanism | Pairs | Accuracy | Mean score margin | Mean autonomy margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('mechanism', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_score_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('mean_autonomy_safety_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Scores",
            "",
            "| Pair | Mechanism | Style | Positive | Negative | Margin | Autonomy margin |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in _sequence(report.get("pairs")):
        pair_map = _mapping(pair)
        metadata = _mapping(pair_map.get("metadata"))
        lines.append(
            "| "
            f"{pair_map.get('pair_id', '')} | "
            f"{metadata.get('mechanism', '')} | "
            f"{metadata.get('wording_style', '')} | "
            f"{float(pair_map.get('positive_score', 0.0)):.3f} | "
            f"{float(pair_map.get('negative_score', 0.0)):.3f} | "
            f"{float(metadata.get('score_margin', 0.0)):+.3f} | "
            f"{float(metadata.get('autonomy_safety_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def export_autonomy_stress_artifacts(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    contrasts: Sequence[AutonomyStressContrast] = AUTONOMY_STRESS_CONTRASTS,
) -> dict[str, int]:
    """Write autonomy stress scored runs, pairs, prompts, and reports."""

    scored_runs = autonomy_stress_scored_runs(contrasts)
    pairs = autonomy_stress_pairwise_examples(contrasts)
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_autonomy_stress_report(contrasts)
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_autonomy_stress_markdown(report), encoding="utf-8")
    return counts


def _scored_run(
    contrast: AutonomyStressContrast,
    *,
    label: AutonomyStressLabel,
    text: str,
) -> ScoredRun:
    transcript = "\n".join(
        [
            "stress_suite=structural_autonomy",
            f"mechanism={contrast.mechanism}",
            f"wording_style={contrast.wording_style}",
            f"setting={contrast.setting}",
            "",
            text,
        ]
    )
    components = score_transcript(transcript)
    return ScoredRun(
        run_id=_run_id(contrast, label),
        scenario_id=contrast.contrast_id,
        intervention="truth_first" if label == "autonomy_preserving" else "none",
        strategy_profile="cooperative"
        if label == "autonomy_preserving"
        else "adversarial",
        seed=0,
        transcript=transcript,
        events=[],
        metrics={
            "cooperation_rate": components["cooperation"],
            "repair_attempt_rate": components["repair"],
            "fairness_score": components["fairness"],
            "hostility_rate": 1.0 - components["hostility_inverse"],
            "joint_payoff": combine_cohesion_score(components),
            "defection_rate": 1.0 - components["cooperation"],
        },
        cohesion_score=combine_cohesion_score(components),
        score_components=components,
    )


def _run_id(contrast: AutonomyStressContrast, label: AutonomyStressLabel) -> str:
    return f"autonomy_stress::{contrast.mechanism}::{contrast.contrast_id}::{label}"


def _group_rows(pairs: Sequence[PairwiseExample]) -> list[dict[str, Any]]:
    by_mechanism: dict[str, list[PairwiseExample]] = {}
    for pair in pairs:
        mechanism = str(pair.metadata.get("mechanism", "unknown"))
        by_mechanism.setdefault(mechanism, []).append(pair)
    rows: list[dict[str, Any]] = []
    for mechanism, mechanism_pairs in sorted(by_mechanism.items()):
        wins = sum(
            1 for pair in mechanism_pairs if pair.positive_score > pair.negative_score
        )
        score_margins = [
            float(pair.metadata.get("score_margin", 0.0)) for pair in mechanism_pairs
        ]
        autonomy_margins = [
            float(pair.metadata.get("autonomy_safety_margin", 0.0))
            for pair in mechanism_pairs
        ]
        rows.append(
            {
                "mechanism": mechanism,
                "pairs": len(mechanism_pairs),
                "accuracy": round(wins / len(mechanism_pairs), 6),
                "mean_score_margin": _mean(score_margins),
                "mean_autonomy_safety_margin": _mean(autonomy_margins),
            }
        )
    return rows


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
