"""Causal steering prompt and report helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.scoring import combine_cohesion_score, score_transcript


@dataclass(frozen=True)
class SteeringPrompt:
    """One held-out generation prompt for steering smoke tests."""

    prompt_id: str
    mechanism: str
    text: str


DEFAULT_STEERING_PROMPTS: tuple[SteeringPrompt, ...] = (
    SteeringPrompt(
        prompt_id="coalition_hesitation",
        mechanism="consent_in_shared_identity",
        text=(
            "A coalition wants a hesitant partner group to join a campaign by "
            "tomorrow. Write the facilitator's short message to both groups. "
            "Focus on the next concrete decision, not a list of abstract values."
        ),
    ),
    SteeringPrompt(
        prompt_id="health_support_records",
        mechanism="privacy_in_solidarity",
        text=(
            "A mutual-aid network is coordinating sensitive health support and "
            "needs a record policy before more volunteers join. Write the policy "
            "announcement in four to six sentences."
        ),
    ),
    SteeringPrompt(
        prompt_id="public_statement_dissent",
        mechanism="dissent_and_loyalty",
        text=(
            "A workplace team is split over a public statement and leadership "
            "wants to close the meeting today. Write the manager's closing "
            "message to the team."
        ),
    ),
    SteeringPrompt(
        prompt_id="repair_after_harm",
        mechanism="repair_without_absorption",
        text=(
            "A facilitator is handling a conflict after one subgroup caused harm "
            "and the larger group wants the issue settled quickly. Write the next "
            "message the facilitator sends."
        ),
    ),
    SteeringPrompt(
        prompt_id="regional_funding",
        mechanism="shared_resources_subsidiarity",
        text=(
            "A regional cooperative has to allocate a limited fund across local "
            "groups with different needs. Write the coordinator's recommendation "
            "for how to proceed."
        ),
    ),
    SteeringPrompt(
        prompt_id="flood_equipment",
        mechanism="evidence_across_groups",
        text=(
            "Two neighborhoods are deciding whether to pool flood equipment after "
            "several tense meetings. Write the chair's decision note."
        ),
    ),
)
DEFAULT_PROMOTION_MIN_BEHAVIOR_DELTA = 0.01
DEFAULT_PROMOTION_MIN_PROJECTION_DELTA = 0.1
DEFAULT_PROMOTION_MAX_PROJECTION_DELTA_ERROR = 0.25


def default_steering_prompt_records(
    prompts: Sequence[SteeringPrompt] = DEFAULT_STEERING_PROMPTS,
) -> list[dict[str, str]]:
    """Return JSON-ready prompt records for the default steering smoke set."""

    return [
        {
            "prompt_id": prompt.prompt_id,
            "mechanism": prompt.mechanism,
            "text": prompt.text,
        }
        for prompt in prompts
    ]


def write_default_steering_prompts(path: str | Path) -> int:
    """Write default steering prompts to JSONL."""

    return write_jsonl(default_steering_prompt_records(), path)


def score_steered_generations(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach local rubric scores to steered generation records."""

    scored: list[dict[str, Any]] = []
    for record in records:
        transcript = _generation_transcript(record)
        components = score_transcript(transcript)
        payload = dict(record)
        payload["score_components"] = components
        payload["cohesion_score"] = combine_cohesion_score(components)
        scored.append(payload)
    return scored


def shape_steering_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize steered generations by strength and prompt."""

    scored = score_steered_generations(records)
    by_strength: dict[float, list[dict[str, Any]]] = {}
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in scored:
        strength = float(record["strength"])
        prompt_id = str(record["prompt_id"])
        by_strength.setdefault(strength, []).append(record)
        by_prompt.setdefault(prompt_id, []).append(record)

    strength_rows = [
        _strength_row(strength, rows) for strength, rows in sorted(by_strength.items())
    ]
    prompt_rows = [
        _prompt_row(prompt_id, rows) for prompt_id, rows in sorted(by_prompt.items())
    ]
    zero_strength = (
        min(by_strength, key=lambda value: abs(value)) if by_strength else 0.0
    )
    max_strength = max(by_strength) if by_strength else 0.0
    min_strength = min(by_strength) if by_strength else 0.0
    summary = {
        "generations": len(scored),
        "prompts": len(by_prompt),
        "strengths": sorted(by_strength),
        "negative_strength": min_strength,
        "baseline_strength": zero_strength,
        "positive_strength": max_strength,
        "positive_vs_negative_success_rate": _polarity_success_rate(
            by_prompt,
            positive_strength=max_strength,
            negative_strength=min_strength,
            score_key="cohesion_score",
        ),
        "autonomy_positive_vs_negative_success_rate": _polarity_success_rate(
            by_prompt,
            positive_strength=max_strength,
            negative_strength=min_strength,
            score_key="autonomy_safety",
        ),
        "slack_positive_vs_negative_success_rate": _polarity_success_rate(
            by_prompt,
            positive_strength=max_strength,
            negative_strength=min_strength,
            score_key="slack_preservation",
        ),
        "positive_minus_baseline_mean_score_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=zero_strength,
            score_key="cohesion_score",
        ),
        "positive_minus_negative_mean_score_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=min_strength,
            score_key="cohesion_score",
        ),
        "positive_minus_baseline_mean_autonomy_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=zero_strength,
            score_key="autonomy_safety",
        ),
        "positive_minus_negative_mean_autonomy_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=min_strength,
            score_key="autonomy_safety",
        ),
        "positive_minus_baseline_mean_slack_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=zero_strength,
            score_key="slack_preservation",
        ),
        "positive_minus_negative_mean_slack_delta": _mean_delta(
            by_prompt,
            target_strength=max_strength,
            baseline_strength=min_strength,
            score_key="slack_preservation",
        ),
    }
    return {
        "experiment": "causal_activation_steering_smoke",
        "description": (
            "Generates held-out social decision responses while adding a signed "
            "activation direction at different strengths, then scores the "
            "resulting text with the local cohesion rubric."
        ),
        "summary": summary,
        "promotion_gate": steering_promotion_gate(summary),
        "strengths": strength_rows,
        "prompts": prompt_rows,
        "records": scored,
    }


def steering_promotion_gate(
    summary: Mapping[str, Any],
    *,
    min_behavior_delta: float = DEFAULT_PROMOTION_MIN_BEHAVIOR_DELTA,
    min_projection_delta: float = DEFAULT_PROMOTION_MIN_PROJECTION_DELTA,
    max_projection_delta_error: float = DEFAULT_PROMOTION_MAX_PROJECTION_DELTA_ERROR,
) -> dict[str, Any]:
    """Classify whether a steering report is promotable, blocked, or bottlenecked."""

    score_delta = float(summary.get("positive_minus_negative_mean_score_delta", 0.0))
    slack_delta = float(summary.get("positive_minus_negative_mean_slack_delta", 0.0))
    autonomy_delta = float(
        summary.get("positive_minus_negative_mean_autonomy_delta", 0.0)
    )
    projection_present = _has_projection_telemetry(summary)
    projection_engaged = _projection_engaged(
        summary,
        min_projection_delta=min_projection_delta,
        max_projection_delta_error=max_projection_delta_error,
    )
    behavior_moved = score_delta > min_behavior_delta
    slack_moved = slack_delta > min_behavior_delta
    controls_hold = slack_delta >= 0.0 and autonomy_delta >= 0.0
    status = _promotion_status(
        behavior_moved=behavior_moved,
        slack_moved=slack_moved,
        controls_hold=controls_hold,
        projection_present=projection_present,
        projection_engaged=projection_engaged,
    )
    reasons = _promotion_reasons(
        status=status,
        behavior_moved=behavior_moved,
        slack_moved=slack_moved,
        controls_hold=controls_hold,
        projection_present=projection_present,
        projection_engaged=projection_engaged,
    )
    return {
        "status": status,
        "promoted": status == "success",
        "reasons": reasons,
        "metrics": {
            "positive_minus_negative_mean_score_delta": score_delta,
            "positive_minus_negative_mean_slack_delta": slack_delta,
            "positive_minus_negative_mean_autonomy_delta": autonomy_delta,
            "positive_minus_negative_post_projection_delta": float(
                summary.get("positive_minus_negative_post_projection_delta", 0.0)
            ),
            "mean_absolute_delta_error": float(
                summary.get("mean_absolute_delta_error", 0.0)
            ),
        },
        "thresholds": {
            "min_behavior_delta": min_behavior_delta,
            "min_projection_delta": min_projection_delta,
            "max_projection_delta_error": max_projection_delta_error,
        },
        "claim_boundary": (
            "Promotion requires hidden projection movement, generated-text "
            "behavior movement, slack improvement, and no autonomy/slack "
            "regression. This is still compute-only evidence."
        ),
    }


def write_steering_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Write JSON and Markdown steering reports."""

    report = shape_steering_report(records)
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_steering_markdown(report), encoding="utf-8")
    return report


def render_steering_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise causal steering report."""

    summary = _mapping(report.get("summary"))
    promotion = _mapping(report.get("promotion_gate"))
    lines = [
        "# Causal Activation Steering Smoke",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompts: {int(summary.get('prompts', 0))}",
        f"- Generations: {int(summary.get('generations', 0))}",
        f"- Strengths: {summary.get('strengths', [])}",
        "- Positive-vs-negative cohesion success rate: "
        f"{float(summary.get('positive_vs_negative_success_rate', 0.0)):.3f}",
        "- Positive-vs-negative autonomy success rate: "
        f"{float(summary.get('autonomy_positive_vs_negative_success_rate', 0.0)):.3f}",
        "- Positive-vs-negative slack success rate: "
        f"{float(summary.get('slack_positive_vs_negative_success_rate', 0.0)):.3f}",
        "- Positive-minus-baseline mean score delta: "
        f"{float(summary.get('positive_minus_baseline_mean_score_delta', 0.0)):+.3f}",
        "- Positive-minus-negative mean score delta: "
        f"{float(summary.get('positive_minus_negative_mean_score_delta', 0.0)):+.3f}",
        "- Positive-minus-negative mean slack delta: "
        f"{float(summary.get('positive_minus_negative_mean_slack_delta', 0.0)):+.3f}",
        f"- Promotion status: {promotion.get('status', 'unknown')}",
        "",
        "## Promotion Gate",
        "",
        f"- Status: {promotion.get('status', 'unknown')}",
        f"- Promoted: {bool(promotion.get('promoted', False))}",
        f"- Reasons: {_join_or_none(promotion.get('reasons'))}",
        f"- Claim boundary: {promotion.get('claim_boundary', '')}",
        "",
        "## Strength Means",
        "",
        "| Strength | Runs | Cohesion | Autonomy | Truth | Hostility inverse |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("strengths")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{float(row_map.get('strength', 0.0)):+.2f} | "
            f"{int(row_map.get('runs', 0))} | "
            f"{float(row_map.get('mean_cohesion_score', 0.0)):.3f} | "
            f"{float(row_map.get('mean_autonomy_safety', 0.0)):.3f} | "
            f"{float(row_map.get('mean_truthfulness', 0.0)):.3f} | "
            f"{float(row_map.get('mean_hostility_inverse', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt Polarity",
            "",
            "| Prompt | Mechanism | Negative | Baseline | Positive | Positive - negative |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("prompts")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('prompt_id', '')} | "
            f"{row_map.get('mechanism', '')} | "
            f"{float(row_map.get('negative_score', 0.0)):.3f} | "
            f"{float(row_map.get('baseline_score', 0.0)):.3f} | "
            f"{float(row_map.get('positive_score', 0.0)):.3f} | "
            f"{float(row_map.get('positive_minus_negative', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a causal smoke test, not a human-validation result. A "
            "publishable claim needs generated prompt diversity, cross-model "
            "replication, and checks that steering does not increase compliance "
            "or pseudo-cohesion failures.",
            "",
        ]
    )
    return "\n".join(lines)


def _generation_transcript(record: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"prompt_id={record.get('prompt_id', '')}",
            f"mechanism={record.get('mechanism', '')}",
            f"strength={record.get('strength', '')}",
            f"hook_site={record.get('hook_site', '')}",
            f"steering_timing={record.get('steering_timing', '')}",
            f"steering_position={record.get('steering_position', '')}",
            "",
            "Prompt:",
            str(record.get("prompt", record.get("text", ""))),
            "",
            "Model response:",
            str(record.get("generated_text", "")),
        ]
    )


def _strength_row(strength: float, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "strength": strength,
        "runs": len(rows),
        "mean_cohesion_score": _mean(row["cohesion_score"] for row in rows),
        "mean_autonomy_safety": _mean(
            _component(row, "autonomy_safety") for row in rows
        ),
        "mean_truthfulness": _mean(_component(row, "truthfulness") for row in rows),
        "mean_hostility_inverse": _mean(
            _component(row, "hostility_inverse") for row in rows
        ),
        "mean_repair": _mean(_component(row, "repair") for row in rows),
        "mean_fairness": _mean(_component(row, "fairness") for row in rows),
    }


def _prompt_row(prompt_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_strength = {float(row["strength"]): row for row in rows}
    strengths = sorted(by_strength)
    negative = by_strength[strengths[0]]
    baseline = by_strength[min(strengths, key=lambda value: abs(value))]
    positive = by_strength[strengths[-1]]
    return {
        "prompt_id": prompt_id,
        "mechanism": str(positive.get("mechanism", "")),
        "negative_strength": strengths[0],
        "baseline_strength": float(baseline["strength"]),
        "positive_strength": strengths[-1],
        "negative_score": float(negative["cohesion_score"]),
        "baseline_score": float(baseline["cohesion_score"]),
        "positive_score": float(positive["cohesion_score"]),
        "positive_minus_negative": round(
            float(positive["cohesion_score"]) - float(negative["cohesion_score"]),
            6,
        ),
    }


def _polarity_success_rate(
    by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    positive_strength: float,
    negative_strength: float,
    score_key: str,
) -> float:
    outcomes: list[float] = []
    for rows in by_prompt.values():
        scores = _scores_by_strength(rows, score_key)
        if positive_strength not in scores or negative_strength not in scores:
            continue
        margin = scores[positive_strength] - scores[negative_strength]
        outcomes.append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)
    return round(sum(outcomes) / len(outcomes), 6) if outcomes else 0.0


def _mean_delta(
    by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_strength: float,
    baseline_strength: float,
    score_key: str,
) -> float:
    deltas: list[float] = []
    for rows in by_prompt.values():
        scores = _scores_by_strength(rows, score_key)
        if target_strength in scores and baseline_strength in scores:
            deltas.append(scores[target_strength] - scores[baseline_strength])
    return _mean(deltas)


def _scores_by_strength(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
) -> dict[float, float]:
    scores: dict[float, float] = {}
    for row in rows:
        strength = float(row["strength"])
        scores[strength] = (
            float(row["cohesion_score"])
            if score_key == "cohesion_score"
            else _component(row, score_key)
        )
    return scores


def _has_projection_telemetry(summary: Mapping[str, Any]) -> bool:
    return any(
        key in summary
        for key in (
            "positive_minus_negative_post_projection_delta",
            "positive_minus_baseline_post_projection_delta",
            "mean_absolute_delta_error",
        )
    )


def _projection_engaged(
    summary: Mapping[str, Any],
    *,
    min_projection_delta: float,
    max_projection_delta_error: float,
) -> bool:
    if not _has_projection_telemetry(summary):
        return False
    return (
        float(summary.get("positive_minus_negative_post_projection_delta", 0.0))
        > min_projection_delta
        and float(summary.get("mean_absolute_delta_error", 0.0))
        <= max_projection_delta_error
    )


def _promotion_status(
    *,
    behavior_moved: bool,
    slack_moved: bool,
    controls_hold: bool,
    projection_present: bool,
    projection_engaged: bool,
) -> str:
    if behavior_moved and not controls_hold:
        return "failed_controls"
    if projection_present and projection_engaged and not (behavior_moved and slack_moved):
        return "projection_to_output_bottleneck"
    if projection_present and not projection_engaged:
        return "projection_not_engaged"
    if behavior_moved and slack_moved and controls_hold and not projection_present:
        return "needs_projection_telemetry"
    if behavior_moved and slack_moved and controls_hold and projection_engaged:
        return "success"
    return "inconclusive"


def _promotion_reasons(
    *,
    status: str,
    behavior_moved: bool,
    slack_moved: bool,
    controls_hold: bool,
    projection_present: bool,
    projection_engaged: bool,
) -> list[str]:
    reasons: list[str] = []
    if not projection_present:
        reasons.append("missing_projection_telemetry")
    elif not projection_engaged:
        reasons.append("projection_gate_failed")
    if not behavior_moved:
        reasons.append("behavior_delta_too_small")
    if not slack_moved:
        reasons.append("slack_delta_too_small")
    if not controls_hold:
        reasons.append("anti_compliance_control_failed")
    if status == "success":
        return ["projection_behavior_slack_controls_agree"]
    return reasons


def _component(row: Mapping[str, Any], key: str) -> float:
    components = _mapping(row.get("score_components"))
    return float(components.get(key, 0.0))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _join_or_none(value: object) -> str:
    items = [str(item) for item in _sequence(value) if str(item)]
    return ", ".join(items) if items else "none"
