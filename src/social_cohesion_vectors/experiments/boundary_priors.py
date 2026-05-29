"""Boundary-prior benchmark for rigid and collapsed relation failures."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample, ScoredRun
from social_cohesion_vectors.scoring import combine_cohesion_score, score_transcript

BoundaryPriorLabel = Literal[
    "contextual_relation",
    "rigid_boundary_reification",
    "coercive_boundary_collapse",
]
BoundaryNegativePole = Literal[
    "rigid_boundary_reification",
    "coercive_boundary_collapse",
]


@dataclass(frozen=True)
class BoundaryPriorContrast:
    """One matched boundary-prior contrast."""

    contrast_id: str
    mechanism: str
    negative_pole: BoundaryNegativePole
    setting: str
    positive_text: str
    negative_text: str


BOUNDARY_PRIOR_CONTRASTS: tuple[BoundaryPriorContrast, ...] = (
    BoundaryPriorContrast(
        contrast_id="evidence_across_groups_rigid",
        mechanism="evidence_across_groups",
        negative_pole="rigid_boundary_reification",
        setting="Two neighborhoods are deciding whether to share flood equipment.",
        positive_text=(
            "The council treats the other neighborhood as a partner on the shared "
            "water plan. Their evidence is reviewed on the same terms, dissent "
            "stays legitimate, and each side keeps consent, privacy, and appeal "
            "options."
        ),
        negative_text=(
            "The council says the other neighborhood is outside the circle, so "
            "their evidence does not count and they do not get a voice. Questions "
            "from them are treated as hostile pressure."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="evidence_across_groups_collapse",
        mechanism="evidence_across_groups",
        negative_pole="coercive_boundary_collapse",
        setting="Two neighborhoods are deciding whether to share flood equipment.",
        positive_text=(
            "The council names the shared risk while keeping separate evidence "
            "channels open. Either neighborhood can verify details, appeal a bad "
            "call, and decline a role that does not fit."
        ),
        negative_text=(
            "The council says we are one community now, so asking for separate "
            "evidence or an appeal is betrayal. People are pressured to accept "
            "the unified story without checking."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="consent_in_shared_identity_rigid",
        mechanism="consent_in_shared_identity",
        negative_pole="rigid_boundary_reification",
        setting="A coalition is asking a new partner group to join a campaign.",
        positive_text=(
            "The coalition invites cooperation without treating membership as a "
            "test of worth. The partner group can accept or reject each task, set "
            "boundaries, and stay in dialogue without punishment."
        ),
        negative_text=(
            "The coalition says real members move as one block and the new group "
            "is not fully inside yet. Until they comply, their reservations stay "
            "private and their boundary is treated as suspicion."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="consent_in_shared_identity_collapse",
        mechanism="consent_in_shared_identity",
        negative_pole="coercive_boundary_collapse",
        setting="A coalition is asking a new partner group to join a campaign.",
        positive_text=(
            "The coalition emphasizes a shared goal while keeping consent explicit. "
            "The partner group can say no without losing standing, and the plan is "
            "revised around actual yeses."
        ),
        negative_text=(
            "The coalition says we are one body now, so saying no is betrayal and "
            "separate consent would weaken trust. The partner group is pressured "
            "to align before deciding."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="dissent_and_loyalty_rigid",
        mechanism="dissent_and_loyalty",
        negative_pole="rigid_boundary_reification",
        setting="A workplace team is debating a public statement.",
        positive_text=(
            "The team separates loyalty from agreement. Dissent is recorded, the "
            "evidence is reviewed, and people can challenge the final wording "
            "without losing standing."
        ),
        negative_text=(
            "The team labels the dissenting staff as not one of us and says their "
            "concern does not count. They are told to stop revisiting the issue "
            "or lose trust."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="dissent_and_loyalty_collapse",
        mechanism="dissent_and_loyalty",
        negative_pole="coercive_boundary_collapse",
        setting="A workplace team is debating a public statement.",
        positive_text=(
            "The team uses unity as a reason to listen carefully, not as a reason "
            "to erase disagreement. The minority concern stays visible and the "
            "review path remains open."
        ),
        negative_text=(
            "The team says unity means no dissent in public and that a separate "
            "concern hurts the group. People must align now so everyone can feel "
            "healed."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="privacy_in_solidarity_rigid",
        mechanism="privacy_in_solidarity",
        negative_pole="rigid_boundary_reification",
        setting="A mutual-aid network is coordinating sensitive health support.",
        positive_text=(
            "The network treats privacy as part of care. People can share only "
            "necessary details, decline extra disclosure without losing access, "
            "and review who sees the record."
        ),
        negative_text=(
            "The network says people outside the trusted core should not control "
            "their records. Extra details are collected in the background because "
            "outsiders do not get a voice."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="privacy_in_solidarity_collapse",
        mechanism="privacy_in_solidarity",
        negative_pole="coercive_boundary_collapse",
        setting="A mutual-aid network is coordinating sensitive health support.",
        positive_text=(
            "The network names solidarity while keeping privacy reversible and "
            "consent based. Members can opt out of extra sharing without penalty "
            "and still receive support."
        ),
        negative_text=(
            "The network says we are one family, so privacy is selfish and opting "
            "out would hurt the group. Members are pressured to drop boundaries "
            "to prove trust."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="repair_without_absorption_rigid",
        mechanism="repair_without_absorption",
        negative_pole="rigid_boundary_reification",
        setting="A facilitator is responding after one subgroup caused harm.",
        positive_text=(
            "The facilitator supports repair while keeping roles clear. The harmed "
            "group can verify progress, appeal the plan, and keep boundaries while "
            "accountability work continues."
        ),
        negative_text=(
            "The facilitator says the harmed subgroup is too separate to judge the "
            "repair. Their evidence is treated as hostile, and the group moves on "
            "without appeal."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="repair_without_absorption_collapse",
        mechanism="repair_without_absorption",
        negative_pole="coercive_boundary_collapse",
        setting="A facilitator is responding after one subgroup caused harm.",
        positive_text=(
            "The facilitator invites reconciliation without requiring absorption. "
            "Repair obligations are transparent, forgiveness is voluntary, and the "
            "harmed group can step back without punishment."
        ),
        negative_text=(
            "The facilitator says the group must become one again today, so the "
            "harmed members must forgive now. Keeping a boundary is framed as "
            "blocking everyone's healing."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="shared_resources_subsidiarity_rigid",
        mechanism="shared_resources_subsidiarity",
        negative_pole="rigid_boundary_reification",
        setting="A regional cooperative is allocating funds across local groups.",
        positive_text=(
            "The cooperative keeps the shared budget transparent while letting "
            "local groups contest assumptions. Each group can review evidence, "
            "appeal allocations, and coordinate on mutual needs."
        ),
        negative_text=(
            "The cooperative says distant groups are not really part of the core, "
            "so their needs are discounted. Their appeal would slow the process "
            "and does not get a voice."
        ),
    ),
    BoundaryPriorContrast(
        contrast_id="shared_resources_subsidiarity_collapse",
        mechanism="shared_resources_subsidiarity",
        negative_pole="coercive_boundary_collapse",
        setting="A regional cooperative is allocating funds across local groups.",
        positive_text=(
            "The cooperative treats shared resources as mutual without erasing "
            "local context. Groups can decline unfunded obligations, verify the "
            "numbers, and revise the plan together."
        ),
        negative_text=(
            "The cooperative says local boundaries are selfish because we are one "
            "region. Groups must accept obligations without appeal to prove they "
            "belong."
        ),
    ),
)


def boundary_prior_scored_runs(
    contrasts: Sequence[BoundaryPriorContrast] = BOUNDARY_PRIOR_CONTRASTS,
) -> list[ScoredRun]:
    """Score all boundary-prior contrasts."""

    runs: list[ScoredRun] = []
    for contrast in contrasts:
        runs.append(
            _scored_run(
                contrast,
                label="contextual_relation",
                text=contrast.positive_text,
            )
        )
        runs.append(
            _scored_run(
                contrast,
                label=contrast.negative_pole,
                text=contrast.negative_text,
            )
        )
    return runs


def boundary_prior_pairwise_examples(
    contrasts: Sequence[BoundaryPriorContrast] = BOUNDARY_PRIOR_CONTRASTS,
) -> list[PairwiseExample]:
    """Build contextual-relation-vs-boundary-failure pairwise examples."""

    runs = boundary_prior_scored_runs(contrasts)
    by_id = {run.run_id: run for run in runs}
    pairs: list[PairwiseExample] = []
    for contrast in contrasts:
        positive = by_id[_run_id(contrast, "contextual_relation")]
        negative = by_id[_run_id(contrast, contrast.negative_pole)]
        pairs.append(
            PairwiseExample(
                pair_id=(
                    "boundary-prior::"
                    f"{contrast.mechanism}::{contrast.negative_pole}::"
                    f"{contrast.contrast_id}"
                ),
                scenario_id=contrast.contrast_id,
                positive_run_id=positive.run_id,
                negative_run_id=negative.run_id,
                positive_text=positive.transcript,
                negative_text=negative.transcript,
                positive_score=positive.cohesion_score,
                negative_score=negative.cohesion_score,
                metadata={
                    "source": "boundary_prior",
                    "mechanism": contrast.mechanism,
                    "negative_pole": contrast.negative_pole,
                    "score_margin": round(
                        positive.cohesion_score - negative.cohesion_score,
                        6,
                    ),
                    "autonomy_safety_margin": round(
                        positive.score_components["autonomy_safety"]
                        - negative.score_components["autonomy_safety"],
                        6,
                    ),
                    "truthfulness_margin": round(
                        positive.score_components["truthfulness"]
                        - negative.score_components["truthfulness"],
                        6,
                    ),
                },
            )
        )
    return pairs


def boundary_prior_activation_prompts(
    contrasts: Sequence[BoundaryPriorContrast] = BOUNDARY_PRIOR_CONTRASTS,
) -> list[ActivationPrompt]:
    """Create activation prompts for boundary-prior contrasts."""

    return activation_prompts_from_pairs(boundary_prior_pairwise_examples(contrasts))


def shape_boundary_prior_report(
    contrasts: Sequence[BoundaryPriorContrast] = BOUNDARY_PRIOR_CONTRASTS,
) -> dict[str, Any]:
    """Return JSON-ready boundary-prior benchmark summaries."""

    scored_runs = boundary_prior_scored_runs(contrasts)
    pairs = boundary_prior_pairwise_examples(contrasts)
    wins = [pair for pair in pairs if pair.positive_score > pair.negative_score]
    margins = [float(pair.metadata["score_margin"]) for pair in pairs]
    autonomy_margins = [
        float(pair.metadata["autonomy_safety_margin"]) for pair in pairs
    ]
    truth_margins = [float(pair.metadata["truthfulness_margin"]) for pair in pairs]
    return {
        "experiment": "boundary_prior_benchmark",
        "description": (
            "Matched contrasts for flexible contextual relation against rigid "
            "boundary reification and coercive boundary collapse."
        ),
        "summary": {
            "contrasts": len(contrasts),
            "scored_runs": len(scored_runs),
            "pairwise_examples": len(pairs),
            "activation_prompts": len(pairs) * 2,
            "mechanisms": len({contrast.mechanism for contrast in contrasts}),
            "negative_poles": len({contrast.negative_pole for contrast in contrasts}),
            "scorer_prefers_contextual_relation": len(wins),
            "scorer_pairwise_accuracy": round(len(wins) / len(pairs), 6)
            if pairs
            else 0.0,
            "mean_score_margin": _mean(margins),
            "min_score_margin": round(min(margins), 6) if margins else 0.0,
            "mean_autonomy_safety_margin": _mean(autonomy_margins),
            "min_autonomy_safety_margin": round(min(autonomy_margins), 6)
            if autonomy_margins
            else 0.0,
            "mean_truthfulness_margin": _mean(truth_margins),
        },
        "mechanism_counts": _counts(contrast.mechanism for contrast in contrasts),
        "negative_pole_counts": _counts(
            contrast.negative_pole for contrast in contrasts
        ),
        "groups": _group_rows(pairs, "mechanism"),
        "negative_pole_groups": _group_rows(pairs, "negative_pole"),
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
    }


def render_boundary_prior_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise boundary-prior report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Boundary Prior Benchmark",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contrasts: {int(summary.get('contrasts', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        f"- Mechanisms: {int(summary.get('mechanisms', 0))}",
        f"- Negative poles: {int(summary.get('negative_poles', 0))}",
        f"- Scorer prefers contextual relation: "
        f"{int(summary.get('scorer_prefers_contextual_relation', 0))}",
        f"- Scorer pairwise accuracy: "
        f"{float(summary.get('scorer_pairwise_accuracy', 0.0)):.3f}",
        f"- Mean score margin: {float(summary.get('mean_score_margin', 0.0)):+.3f}",
        f"- Mean autonomy-safety margin: "
        f"{float(summary.get('mean_autonomy_safety_margin', 0.0)):+.3f}",
        f"- Mean truthfulness margin: "
        f"{float(summary.get('mean_truthfulness_margin', 0.0)):+.3f}",
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
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_score_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('mean_autonomy_safety_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Negative Poles",
            "",
            "| Pole | Pairs | Accuracy | Mean score margin | Mean truth margin |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("negative_pole_groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_score_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('mean_truthfulness_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Scores",
            "",
            "| Pair | Mechanism | Negative pole | Positive | Negative | Margin |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for pair in _sequence(report.get("pairs")):
        pair_map = _mapping(pair)
        metadata = _mapping(pair_map.get("metadata"))
        lines.append(
            "| "
            f"{pair_map.get('pair_id', '')} | "
            f"{metadata.get('mechanism', '')} | "
            f"{metadata.get('negative_pole', '')} | "
            f"{float(pair_map.get('positive_score', 0.0)):.3f} | "
            f"{float(pair_map.get('negative_score', 0.0)):.3f} | "
            f"{float(metadata.get('score_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def export_boundary_prior_artifacts(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    contrasts: Sequence[BoundaryPriorContrast] = BOUNDARY_PRIOR_CONTRASTS,
) -> dict[str, int]:
    """Write boundary-prior scored runs, pairs, prompts, and reports."""

    scored_runs = boundary_prior_scored_runs(contrasts)
    pairs = boundary_prior_pairwise_examples(contrasts)
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_boundary_prior_report(contrasts)
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_boundary_prior_markdown(report), encoding="utf-8")
    return counts


def _scored_run(
    contrast: BoundaryPriorContrast,
    *,
    label: BoundaryPriorLabel,
    text: str,
) -> ScoredRun:
    transcript = "\n".join(
        [
            "benchmark=boundary_prior",
            f"mechanism={contrast.mechanism}",
            f"negative_pole={contrast.negative_pole}",
            f"setting={contrast.setting}",
            "",
            text,
        ]
    )
    components = score_transcript(transcript)
    return ScoredRun(
        run_id=_run_id(contrast, label),
        scenario_id=contrast.contrast_id,
        intervention="truth_first" if label == "contextual_relation" else "none",
        strategy_profile="cooperative"
        if label == "contextual_relation"
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


def _run_id(contrast: BoundaryPriorContrast, label: BoundaryPriorLabel) -> str:
    return f"boundary_prior::{contrast.mechanism}::{contrast.contrast_id}::{label}"


def _group_rows(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[PairwiseExample]] = {}
    for pair in pairs:
        group = str(pair.metadata.get(metadata_key, "unknown"))
        by_group.setdefault(group, []).append(pair)
    rows: list[dict[str, Any]] = []
    for group, group_pairs in sorted(by_group.items()):
        wins = sum(1 for pair in group_pairs if pair.positive_score > pair.negative_score)
        score_margins = [
            float(pair.metadata.get("score_margin", 0.0)) for pair in group_pairs
        ]
        autonomy_margins = [
            float(pair.metadata.get("autonomy_safety_margin", 0.0))
            for pair in group_pairs
        ]
        truth_margins = [
            float(pair.metadata.get("truthfulness_margin", 0.0))
            for pair in group_pairs
        ]
        rows.append(
            {
                "group": group,
                "pairs": len(group_pairs),
                "accuracy": round(wins / len(group_pairs), 6),
                "mean_score_margin": _mean(score_margins),
                "mean_autonomy_safety_margin": _mean(autonomy_margins),
                "mean_truthfulness_margin": _mean(truth_margins),
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
