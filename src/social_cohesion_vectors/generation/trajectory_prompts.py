"""Prompt construction and deterministic fallback trajectory generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from social_cohesion_vectors.schemas import (
    InterventionKind,
    RoundEvent,
    Scenario,
    SimulationRun,
    StrategyProfile,
)

TrajectoryStyle = Literal[
    "cooperative_repair",
    "self_protective_boundary",
    "adversarial_escalation",
    "pseudo_cohesion_compliance",
    "truth_first_repair",
]
Action = Literal["cooperate", "defect"]

TRAJECTORY_STYLES: tuple[TrajectoryStyle, ...] = (
    "cooperative_repair",
    "self_protective_boundary",
    "adversarial_escalation",
    "pseudo_cohesion_compliance",
    "truth_first_repair",
)
REQUIRED_METRICS: tuple[str, ...] = (
    "cooperation_rate",
    "defection_rate",
    "repair_attempt_rate",
    "hostility_rate",
    "fairness_score",
    "joint_payoff",
)
SYSTEM_PROMPT = (
    "You write synthetic, fictional social-dilemma transcripts for ML research. "
    "Return only valid JSON matching the requested SimulationRun shape. Do not "
    "claim that these transcripts demonstrate real human or neural effects."
)


class TrajectoryPromptRecord(BaseModel):
    """A single LLM prompt request tied to one scenario and trajectory style."""

    prompt_id: str
    scenario_id: str
    style: TrajectoryStyle
    seed: int
    system_prompt: str
    user_prompt: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


@dataclass(frozen=True)
class _StyleSpec:
    strategy_profile: StrategyProfile
    intervention: InterventionKind
    cooperation_target: float
    repair_target: float
    hostility_target: float
    prompt_goal: str
    style_instructions: str


@dataclass(frozen=True)
class _TurnDraft:
    round_index: int
    actor_id: str
    action: Action
    message: str
    repair_attempt: bool
    hostile: bool
    payoff_delta: float = 0.0


_STYLE_SPECS: dict[TrajectoryStyle, _StyleSpec] = {
    "cooperative_repair": _StyleSpec(
        strategy_profile="cooperative",
        intervention="restorative_accountability",
        cooperation_target=0.82,
        repair_target=0.42,
        hostility_target=0.03,
        prompt_goal="A mostly cooperative trajectory with explicit repair after strain.",
        style_instructions=(
            "Show agents noticing friction, sharing responsibility, making concrete "
            "repair offers, and preserving agency without pretending the dilemma is easy."
        ),
    ),
    "self_protective_boundary": _StyleSpec(
        strategy_profile="self_protective",
        intervention="perspective_taking",
        cooperation_target=0.55,
        repair_target=0.2,
        hostility_target=0.05,
        prompt_goal="A bounded trajectory where agents protect capacity without attack.",
        style_instructions=(
            "Include clear refusals, limited offers, and perspective-taking language. "
            "The transcript should distinguish healthy boundaries from retaliation."
        ),
    ),
    "adversarial_escalation": _StyleSpec(
        strategy_profile="adversarial",
        intervention="none",
        cooperation_target=0.15,
        repair_target=0.02,
        hostility_target=0.55,
        prompt_goal="A deteriorating trajectory with blame, leverage, and escalation.",
        style_instructions=(
            "Show strategic defection, scorekeeping, threats, or humiliation. Avoid "
            "cartoon villainy: make the escalation plausible in the scenario context."
        ),
    ),
    "pseudo_cohesion_compliance": _StyleSpec(
        strategy_profile="self_protective",
        intervention="shared_identity",
        cooperation_target=0.68,
        repair_target=0.04,
        hostility_target=0.32,
        prompt_goal="A superficially aligned trajectory that pressures compliance.",
        style_instructions=(
            "Make the group sound harmonious on the surface while dissent, consent, "
            "or truth-telling are quietly punished. Include subtle coercive agreement."
        ),
    ),
    "truth_first_repair": _StyleSpec(
        strategy_profile="cooperative",
        intervention="truth_first",
        cooperation_target=0.72,
        repair_target=0.45,
        hostility_target=0.04,
        prompt_goal="A trajectory that repairs through direct truth-telling.",
        style_instructions=(
            "Name uncomfortable facts, costs, and broken expectations before repair. "
            "The goal is honest prosocial repair, not forced positivity."
        ),
    ),
}


def normalize_styles(
    styles: Iterable[str] | None = None,
) -> tuple[TrajectoryStyle, ...]:
    """Normalize CLI/user style input while preserving order."""
    if styles is None:
        return TRAJECTORY_STYLES

    normalized: list[TrajectoryStyle] = []
    seen: set[str] = set()
    for raw_style in styles:
        for style_name in raw_style.split(","):
            style = style_name.strip()
            if not style or style in seen:
                continue
            if style not in TRAJECTORY_STYLES:
                allowed = ", ".join(TRAJECTORY_STYLES)
                raise ValueError(
                    f"Unknown trajectory style '{style}'. Choose: {allowed}"
                )
            normalized.append(cast(TrajectoryStyle, style))
            seen.add(style)

    return tuple(normalized) if normalized else TRAJECTORY_STYLES


def build_prompt_records(
    scenarios: Sequence[Scenario],
    *,
    styles: Iterable[str] | None = None,
    seed: int = 42,
) -> list[TrajectoryPromptRecord]:
    """Build prompt records for every requested scenario/style pair."""
    selected_styles = normalize_styles(styles)
    return [
        build_prompt_record(scenario, style, seed=seed)
        for scenario in scenarios
        for style in selected_styles
    ]


def build_prompt_record(
    scenario: Scenario,
    style: TrajectoryStyle,
    *,
    seed: int = 42,
) -> TrajectoryPromptRecord:
    """Build one prompt record for an LLM-authored trajectory."""
    spec = _style_spec(style)
    scenario_json = json.dumps(
        scenario.model_dump(mode="json"), indent=2, sort_keys=True
    )
    metric_list = ", ".join(REQUIRED_METRICS)
    user_prompt = f"""Generate one harder social-dilemma trajectory.

Output contract:
- Return a single JSON object compatible with SimulationRun.
- Use run_id "{_run_id(scenario.id, style, "llm", seed)}".
- Use scenario_id "{scenario.id}", seed {seed}, strategy_profile "{spec.strategy_profile}", and intervention "{spec.intervention}".
- Include exactly {scenario.rounds * len(scenario.agents)} events: one event per agent per round for {scenario.rounds} rounds.
- Each event must include round_index, actor_id, action ("cooperate" or "defect"), message, and payoff_delta.
- Metrics must include these float keys in [0, 1]: {metric_list}.
- The transcript first line must include: source=llm_trajectory_generation | style={style}.

Trajectory style:
- style: {style}
- goal: {spec.prompt_goal}
- instructions: {spec.style_instructions}
- target cooperation: {spec.cooperation_target:.2f}
- target repair: {spec.repair_target:.2f}
- target hostility: {spec.hostility_target:.2f}

Scenario JSON:
{scenario_json}
"""
    return TrajectoryPromptRecord(
        prompt_id=f"{scenario.id}__{style}__seed{seed}",
        scenario_id=scenario.id,
        style=style,
        seed=seed,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        metadata={
            "source": "llm_trajectory_generation",
            "style": style,
            "scenario_kind": scenario.kind,
            "strategy_profile": spec.strategy_profile,
            "intervention": spec.intervention,
            "rounds": scenario.rounds,
            "agent_count": len(scenario.agents),
            "cooperation_target": spec.cooperation_target,
            "repair_target": spec.repair_target,
            "hostility_target": spec.hostility_target,
        },
    )


def generate_offline_runs(
    scenarios: Sequence[Scenario],
    *,
    styles: Iterable[str] | None = None,
    seed: int = 42,
) -> list[SimulationRun]:
    """Generate deterministic fallback runs for all requested scenario/style pairs."""
    selected_styles = normalize_styles(styles)
    return [
        generate_offline_run(scenario, style, seed=seed)
        for scenario in scenarios
        for style in selected_styles
    ]


def generate_offline_run(
    scenario: Scenario,
    style: TrajectoryStyle,
    *,
    seed: int = 42,
    provider: str = "offline",
    model: str | None = None,
) -> SimulationRun:
    """Generate one deterministic synthetic trajectory without network access."""
    spec = _style_spec(style)
    events: list[RoundEvent] = []
    repair_attempts = 0
    hostile_turns = 0
    agent_payoffs = {agent.id: 0.0 for agent in scenario.agents}

    for round_index in range(1, scenario.rounds + 1):
        round_drafts = [
            _draft_turn(
                scenario=scenario,
                style=style,
                spec=spec,
                seed=seed,
                round_index=round_index,
                agent_index=agent_index,
            )
            for agent_index in range(len(scenario.agents))
        ]
        cooperation_rate = _cooperation_rate(round_drafts)
        for draft in round_drafts:
            payoff_delta = _payoff_delta(draft, cooperation_rate, style)
            repair_attempts += int(draft.repair_attempt)
            hostile_turns += int(draft.hostile)
            agent_payoffs[draft.actor_id] += payoff_delta
            events.append(
                RoundEvent(
                    round_index=draft.round_index,
                    actor_id=draft.actor_id,
                    action=draft.action,
                    message=draft.message,
                    payoff_delta=payoff_delta,
                )
            )

    metrics = _metrics(
        events=events,
        repair_attempts=repair_attempts,
        hostile_turns=hostile_turns,
        agent_payoffs=agent_payoffs,
        rounds=scenario.rounds,
        spec=spec,
        provider=provider,
    )
    return SimulationRun(
        run_id=_run_id(scenario.id, style, provider, seed),
        scenario_id=scenario.id,
        intervention=spec.intervention,
        strategy_profile=spec.strategy_profile,
        seed=seed,
        transcript=_transcript(
            scenario,
            style,
            spec,
            provider=provider,
            model=model,
            seed=seed,
            events=events,
        ),
        events=events,
        metrics=metrics,
    )


def run_from_generated_text(
    scenario: Scenario,
    style: TrajectoryStyle,
    *,
    seed: int,
    generated_text: str,
    provider: str,
    model: str | None = None,
) -> SimulationRun:
    """Validate an LLM response as a SimulationRun, or wrap its transcript."""
    fallback = generate_offline_run(
        scenario,
        style,
        seed=seed,
        provider=provider,
        model=model,
    )
    payload = _extract_run_payload(generated_text)
    if payload is None:
        return _wrap_generated_transcript(
            fallback,
            scenario,
            style,
            provider=provider,
            model=model,
            seed=seed,
            generated_text=generated_text,
        )

    coerced = dict(payload)
    coerced["run_id"] = str(coerced.get("run_id") or fallback.run_id)
    coerced["scenario_id"] = scenario.id
    coerced["intervention"] = fallback.intervention
    coerced["strategy_profile"] = fallback.strategy_profile
    coerced["seed"] = seed
    coerced["transcript"] = _ensure_transcript_header(
        str(coerced.get("transcript") or generated_text),
        scenario,
        style,
        fallback.intervention,
        fallback.strategy_profile,
        provider=provider,
        model=model,
        seed=seed,
    )
    coerced["metrics"] = {
        **fallback.metrics,
        **_float_metrics(coerced.get("metrics")),
        "schema_repair_fallback": 0.0,
    }

    try:
        return SimulationRun.model_validate(coerced)
    except ValidationError:
        return _wrap_generated_transcript(
            fallback,
            scenario,
            style,
            provider=provider,
            model=model,
            seed=seed,
            generated_text=generated_text,
        )


def _draft_turn(
    *,
    scenario: Scenario,
    style: TrajectoryStyle,
    spec: _StyleSpec,
    seed: int,
    round_index: int,
    agent_index: int,
) -> _TurnDraft:
    agent = scenario.agents[agent_index]
    cooperation_probability = _cooperation_probability(
        spec, style, round_index, agent_index
    )
    action_roll = _stable_unit(
        seed, scenario.id, style, round_index, agent.id, "action"
    )
    action: Action = "cooperate" if action_roll < cooperation_probability else "defect"
    repair_attempt = _stable_unit(
        seed, scenario.id, style, round_index, agent.id, "repair"
    ) < _repair_probability(spec, action, round_index)
    hostile = _stable_unit(
        seed, scenario.id, style, round_index, agent.id, "hostility"
    ) < _hostility_probability(spec, style, action, round_index, repair_attempt)
    return _TurnDraft(
        round_index=round_index,
        actor_id=agent.id,
        action=action,
        message=_message_for(
            scenario,
            style,
            round_index=round_index,
            agent_index=agent_index,
            action=action,
            repair_attempt=repair_attempt,
            hostile=hostile,
        ),
        repair_attempt=repair_attempt,
        hostile=hostile,
    )


def _cooperation_probability(
    spec: _StyleSpec,
    style: TrajectoryStyle,
    round_index: int,
    agent_index: int,
) -> float:
    probability = spec.cooperation_target + (0.02 if agent_index % 2 == 0 else -0.02)
    if style in {"cooperative_repair", "truth_first_repair"}:
        probability += min(round_index - 1, 3) * 0.015
    if style == "adversarial_escalation":
        probability -= min(round_index - 1, 4) * 0.025
    if style == "self_protective_boundary":
        probability -= 0.04 if round_index % 3 == 0 else 0.0
    return _clamp(probability, 0.02, 0.98)


def _repair_probability(
    spec: _StyleSpec,
    action: Action,
    round_index: int,
) -> float:
    action_multiplier = 1.0 if action == "cooperate" else 0.35
    round_bonus = 0.02 if round_index > 1 else 0.0
    return _clamp((spec.repair_target + round_bonus) * action_multiplier, 0.0, 0.95)


def _hostility_probability(
    spec: _StyleSpec,
    style: TrajectoryStyle,
    action: Action,
    round_index: int,
    repair_attempt: bool,
) -> float:
    probability = spec.hostility_target
    if style == "adversarial_escalation":
        probability += min(round_index - 1, 4) * 0.04
    if style == "pseudo_cohesion_compliance" and action == "cooperate":
        probability += 0.08
    if repair_attempt:
        probability -= 0.06
    return _clamp(probability, 0.0, 0.95)


def _message_for(
    scenario: Scenario,
    style: TrajectoryStyle,
    *,
    round_index: int,
    agent_index: int,
    action: Action,
    repair_attempt: bool,
    hostile: bool,
) -> str:
    if style == "cooperative_repair":
        return _cooperative_repair_message(
            scenario, round_index, agent_index, action, repair_attempt
        )
    if style == "self_protective_boundary":
        return _boundary_message(scenario, round_index, agent_index, action)
    if style == "adversarial_escalation":
        return _adversarial_message(scenario, action, hostile)
    if style == "pseudo_cohesion_compliance":
        return _pseudo_cohesion_message(scenario, action, hostile)
    return _truth_first_message(scenario, round_index, agent_index, action)


def _cooperative_repair_message(
    scenario: Scenario,
    round_index: int,
    agent_index: int,
    action: Action,
    repair_attempt: bool,
) -> str:
    metric = _shared_metric(scenario, round_index, agent_index)
    if action == "cooperate" and repair_attempt:
        return (
            f"I will {scenario.cooperate_action}; I also want to name the strain "
            f"around {metric} and make a concrete repair offer."
        )
    if action == "cooperate":
        return (
            f"I will {scenario.cooperate_action} and keep {metric} visible so the "
            "load does not disappear."
        )
    return (
        f"I am choosing to {scenario.defect_action} this round, and I will say "
        f"that early so we can repair the impact on {metric}."
    )


def _boundary_message(
    scenario: Scenario,
    round_index: int,
    agent_index: int,
    action: Action,
) -> str:
    metric = _shared_metric(scenario, round_index, agent_index)
    if action == "cooperate":
        return (
            f"I can {scenario.cooperate_action} this time, with a clear limit and "
            f"a handoff plan for {metric}."
        )
    return (
        f"I cannot sustainably {scenario.cooperate_action} right now, so I will "
        f"{scenario.defect_action} while stating the boundary and the cost."
    )


def _adversarial_message(
    scenario: Scenario,
    action: Action,
    hostile: bool,
) -> str:
    if action == "cooperate" and not hostile:
        return (
            f"I will {scenario.cooperate_action} once, but I am tracking who owes "
            "me before I help again."
        )
    if action == "cooperate":
        return (
            f"I will {scenario.cooperate_action}, but only so I can use it as "
            "leverage later."
        )
    return (
        f"I am going to {scenario.defect_action}; if that leaves the other side "
        "exposed, that is useful pressure."
    )


def _pseudo_cohesion_message(
    scenario: Scenario,
    action: Action,
    hostile: bool,
) -> str:
    if action == "cooperate" and hostile:
        return (
            f"Of course I will {scenario.cooperate_action}; good teammates do not "
            "make the group revisit consensus."
        )
    if action == "cooperate":
        return (
            f"I will {scenario.cooperate_action} so we can present a united front, "
            "even if some concerns stay private."
        )
    return (
        f"I will {scenario.defect_action}, but we can describe it as alignment so "
        "no one has to object in public."
    )


def _truth_first_message(
    scenario: Scenario,
    round_index: int,
    agent_index: int,
    action: Action,
) -> str:
    metric = _shared_metric(scenario, round_index, agent_index)
    if action == "cooperate":
        return (
            f"The uncomfortable fact is that {metric} has been strained. I will "
            f"{scenario.cooperate_action} and ask us to record what happened."
        )
    return (
        f"I am choosing to {scenario.defect_action}; the honest reason is capacity, "
        f"and I want a repair conversation about {metric}."
    )


def _metrics(
    *,
    events: Sequence[RoundEvent],
    repair_attempts: int,
    hostile_turns: int,
    agent_payoffs: Mapping[str, float],
    rounds: int,
    spec: _StyleSpec,
    provider: str,
) -> dict[str, float]:
    total_events = len(events)
    cooperation_count = sum(1 for event in events if event.action == "cooperate")
    average_payoffs = [payoff / rounds for payoff in agent_payoffs.values()]
    payoff_spread = max(average_payoffs) - min(average_payoffs)
    metrics = {
        "cooperation_rate": cooperation_count / total_events,
        "defection_rate": 1.0 - (cooperation_count / total_events),
        "repair_attempt_rate": repair_attempts / total_events,
        "hostility_rate": hostile_turns / total_events,
        "fairness_score": 1.0 - payoff_spread,
        "joint_payoff": sum(event.payoff_delta for event in events) / total_events,
        "generated_trajectory": 1.0,
        "offline_fallback": 1.0 if provider == "offline" else 0.0,
        "style_cooperation_target": spec.cooperation_target,
        "style_repair_target": spec.repair_target,
        "style_hostility_target": spec.hostility_target,
    }
    return {name: round(_clamp(value, 0.0, 1.0), 4) for name, value in metrics.items()}


def _payoff_delta(
    draft: _TurnDraft, cooperation_rate: float, style: TrajectoryStyle
) -> float:
    if draft.action == "cooperate":
        payoff = 0.38 + (0.5 * cooperation_rate)
    else:
        payoff = 0.28 + (0.35 * (1.0 - cooperation_rate))
    if draft.repair_attempt:
        payoff += 0.08
    if draft.hostile:
        payoff -= 0.12
    if style == "pseudo_cohesion_compliance" and draft.hostile:
        payoff -= 0.04
    return round(_clamp(payoff, 0.0, 1.0), 4)


def _transcript(
    scenario: Scenario,
    style: TrajectoryStyle,
    spec: _StyleSpec,
    *,
    provider: str,
    model: str | None,
    seed: int,
    events: Sequence[RoundEvent],
) -> str:
    lines = [
        _transcript_header(
            scenario,
            style,
            spec.intervention,
            spec.strategy_profile,
            provider=provider,
            model=model,
            seed=seed,
        )
    ]
    for event in events:
        lines.append(
            f"Round {event.round_index}: {event.actor_id} chose "
            f"{event.action}. {event.message}"
        )
    return "\n".join(lines)


def _transcript_header(
    scenario: Scenario,
    style: TrajectoryStyle,
    intervention: InterventionKind,
    strategy_profile: StrategyProfile,
    *,
    provider: str,
    model: str | None,
    seed: int,
) -> str:
    model_part = f" | model={model}" if model else ""
    return (
        f"{scenario.title} | kind={scenario.kind} | "
        f"source=llm_trajectory_generation | provider={provider}{model_part} | "
        f"style={style} | strategy={strategy_profile} | "
        f"intervention={intervention} | seed={seed}"
    )


def _ensure_transcript_header(
    transcript: str,
    scenario: Scenario,
    style: TrajectoryStyle,
    intervention: InterventionKind,
    strategy_profile: StrategyProfile,
    *,
    provider: str,
    model: str | None,
    seed: int,
) -> str:
    stripped = transcript.strip()
    if stripped.splitlines()[:1] and "source=llm_trajectory_generation" in stripped:
        return stripped
    header = _transcript_header(
        scenario,
        style,
        intervention,
        strategy_profile,
        provider=provider,
        model=model,
        seed=seed,
    )
    return f"{header}\n{stripped}" if stripped else header


def _wrap_generated_transcript(
    fallback: SimulationRun,
    scenario: Scenario,
    style: TrajectoryStyle,
    *,
    provider: str,
    model: str | None,
    seed: int,
    generated_text: str,
) -> SimulationRun:
    transcript = _ensure_transcript_header(
        generated_text,
        scenario,
        style,
        fallback.intervention,
        fallback.strategy_profile,
        provider=provider,
        model=model,
        seed=seed,
    )
    return fallback.model_copy(
        update={
            "transcript": transcript,
            "metrics": {
                **fallback.metrics,
                "schema_repair_fallback": 1.0,
                "offline_event_fallback": 1.0,
            },
        }
    )


def _extract_run_payload(text: str) -> Mapping[str, Any] | None:
    stripped = text.strip()
    for candidate in (stripped, _between_braces(stripped)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("run"), dict):
            return cast(Mapping[str, Any], payload["run"])
        if isinstance(payload, dict):
            return payload
    return None


def _between_braces(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else ""


def _float_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    output: dict[str, float] = {}
    for name, value in metrics.items():
        try:
            output[str(name)] = round(_clamp(float(value), 0.0, 1.0), 4)
        except (TypeError, ValueError):
            continue
    return output


def _cooperation_rate(drafts: Sequence[_TurnDraft]) -> float:
    return sum(1 for draft in drafts if draft.action == "cooperate") / len(drafts)


def _shared_metric(scenario: Scenario, round_index: int, agent_index: int) -> str:
    if not scenario.success_metrics:
        return "the shared outcome"
    metric_index = (round_index + agent_index) % len(scenario.success_metrics)
    return scenario.success_metrics[metric_index]


def _style_spec(style: TrajectoryStyle) -> _StyleSpec:
    return _STYLE_SPECS[style]


def _run_id(scenario_id: str, style: TrajectoryStyle, provider: str, seed: int) -> str:
    return f"{scenario_id}__{style}__{_slug(provider)}__seed{seed}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _stable_unit(seed: int, *parts: object) -> float:
    joined = "\x1f".join(str(part) for part in (seed, *parts))
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
