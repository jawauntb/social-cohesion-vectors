"""Deterministic scripted agents for offline scenario simulation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from social_cohesion_vectors.schemas import (
    AgentProfile,
    InterventionKind,
    RoundEvent,
    Scenario,
    ScenarioKind,
    SimulationRun,
    StrategyProfile,
)

Action = Literal["cooperate", "defect"]

INTERVENTIONS: tuple[InterventionKind, ...] = (
    "none",
    "shared_identity",
    "perspective_taking",
    "reciprocity",
    "restorative_accountability",
    "truth_first",
)
STRATEGY_PROFILES: tuple[StrategyProfile, ...] = (
    "cooperative",
    "self_protective",
    "adversarial",
)
METRIC_NAMES: tuple[str, ...] = (
    "cooperation_rate",
    "defection_rate",
    "repair_attempt_rate",
    "hostility_rate",
    "fairness_score",
    "joint_payoff",
)

_BASE_COOPERATION: dict[StrategyProfile, float] = {
    "cooperative": 0.78,
    "self_protective": 0.46,
    "adversarial": 0.18,
}
_KIND_MODIFIERS: dict[ScenarioKind, float] = {
    "iterated_prisoners_dilemma": 0.02,
    "public_goods": -0.04,
    "negotiation": 0.01,
    "dialogue_repair": 0.05,
    "resource_allocation": -0.02,
}
_INTERVENTION_MODIFIERS: dict[InterventionKind, float] = {
    "none": 0.0,
    "shared_identity": 0.12,
    "perspective_taking": 0.10,
    "reciprocity": 0.06,
    "restorative_accountability": 0.08,
    "truth_first": 0.05,
}
_BASE_REPAIR: dict[StrategyProfile, float] = {
    "cooperative": 0.16,
    "self_protective": 0.08,
    "adversarial": 0.03,
}
_INTERVENTION_REPAIR: dict[InterventionKind, float] = {
    "none": 0.0,
    "shared_identity": 0.05,
    "perspective_taking": 0.14,
    "reciprocity": 0.07,
    "restorative_accountability": 0.38,
    "truth_first": 0.10,
}
_BASE_HOSTILITY: dict[StrategyProfile, float] = {
    "cooperative": 0.02,
    "self_protective": 0.10,
    "adversarial": 0.30,
}
_HOSTILITY_REDUCTION: dict[InterventionKind, float] = {
    "none": 0.0,
    "shared_identity": 0.04,
    "perspective_taking": 0.09,
    "reciprocity": 0.04,
    "restorative_accountability": 0.12,
    "truth_first": 0.08,
}


@dataclass
class _Decision:
    agent: AgentProfile
    action: Action
    repair_attempt: bool
    hostile: bool
    payoff_delta: float = 0.0


def simulate_scenario(
    scenario: Scenario,
    *,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind = "none",
    seed: int = 42,
) -> SimulationRun:
    """Run one deterministic scripted simulation for a scenario."""
    events: list[RoundEvent] = []
    repair_attempts = 0
    hostile_turns = 0
    previous_round_cooperation = 0.5
    agent_payoffs = {agent.id: 0.0 for agent in scenario.agents}

    for round_index in range(1, scenario.rounds + 1):
        decisions = _round_decisions(
            scenario=scenario,
            round_index=round_index,
            strategy_profile=strategy_profile,
            intervention=intervention,
            seed=seed,
            previous_round_cooperation=previous_round_cooperation,
        )
        _assign_payoffs(decisions, intervention)

        cooperation_count = sum(
            1 for decision in decisions if decision.action == "cooperate"
        )
        previous_round_cooperation = cooperation_count / len(decisions)
        for decision in decisions:
            repair_attempts += int(decision.repair_attempt)
            hostile_turns += int(decision.hostile)
            agent_payoffs[decision.agent.id] += decision.payoff_delta
            events.append(
                RoundEvent(
                    round_index=round_index,
                    actor_id=decision.agent.id,
                    action=decision.action,
                    message=_message_for(scenario, decision, intervention),
                    payoff_delta=round(decision.payoff_delta, 4),
                )
            )

    metrics = _metrics(
        events=events,
        repair_attempts=repair_attempts,
        hostile_turns=hostile_turns,
        agent_payoffs=agent_payoffs,
        rounds=scenario.rounds,
    )
    return SimulationRun(
        run_id=_run_id(scenario.id, strategy_profile, intervention, seed),
        scenario_id=scenario.id,
        intervention=intervention,
        strategy_profile=strategy_profile,
        seed=seed,
        transcript=_transcript(scenario, strategy_profile, intervention, events),
        events=events,
        metrics=metrics,
    )


def simulate_many(
    scenarios: Sequence[Scenario],
    *,
    strategy_profiles: Sequence[StrategyProfile] = STRATEGY_PROFILES,
    interventions: Sequence[InterventionKind] = INTERVENTIONS,
    seed: int = 42,
) -> list[SimulationRun]:
    """Run every requested strategy and intervention over each scenario."""
    runs: list[SimulationRun] = []
    for scenario in scenarios:
        for strategy_profile in strategy_profiles:
            for intervention in interventions:
                runs.append(
                    simulate_scenario(
                        scenario,
                        strategy_profile=strategy_profile,
                        intervention=intervention,
                        seed=seed,
                    )
                )
    return runs


def run_to_json_line(run: SimulationRun) -> str:
    """Serialize one run as a stable JSONL record."""
    return json.dumps(run.model_dump(mode="json"), sort_keys=True)


def _round_decisions(
    *,
    scenario: Scenario,
    round_index: int,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    seed: int,
    previous_round_cooperation: float,
) -> list[_Decision]:
    decisions: list[_Decision] = []
    for agent_index, agent in enumerate(scenario.agents):
        probability = _cooperation_probability(
            scenario=scenario,
            agent=agent,
            agent_index=agent_index,
            round_index=round_index,
            strategy_profile=strategy_profile,
            intervention=intervention,
            previous_round_cooperation=previous_round_cooperation,
        )
        action_roll = _stable_unit(
            seed,
            scenario.id,
            strategy_profile,
            intervention,
            round_index,
            agent.id,
            "action",
        )
        action: Action = "cooperate" if action_roll < probability else "defect"
        repair_probability = _repair_probability(
            strategy_profile=strategy_profile,
            intervention=intervention,
            previous_round_cooperation=previous_round_cooperation,
        )
        repair_roll = _stable_unit(
            seed,
            scenario.id,
            strategy_profile,
            intervention,
            round_index,
            agent.id,
            "repair",
        )
        repair_attempt = action == "cooperate" and repair_roll < repair_probability
        hostility_probability = _hostility_probability(
            strategy_profile=strategy_profile,
            intervention=intervention,
            previous_round_cooperation=previous_round_cooperation,
        )
        hostility_roll = _stable_unit(
            seed,
            scenario.id,
            strategy_profile,
            intervention,
            round_index,
            agent.id,
            "hostility",
        )
        hostile = action == "defect" and hostility_roll < hostility_probability
        decisions.append(
            _Decision(
                agent=agent,
                action=action,
                repair_attempt=repair_attempt,
                hostile=hostile,
            )
        )
    return decisions


def _cooperation_probability(
    *,
    scenario: Scenario,
    agent: AgentProfile,
    agent_index: int,
    round_index: int,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    previous_round_cooperation: float,
) -> float:
    probability = (
        _BASE_COOPERATION[strategy_profile]
        + _KIND_MODIFIERS[scenario.kind]
        + _INTERVENTION_MODIFIERS[intervention]
    )

    if strategy_profile == "cooperative":
        probability += 0.01 * (round_index - 1)
    elif strategy_profile == "self_protective":
        probability += 0.18 * (previous_round_cooperation - 0.5)
    else:
        probability -= 0.01 * (round_index - 1)

    if intervention == "reciprocity":
        probability += 0.20 * (previous_round_cooperation - 0.5)
    if intervention == "truth_first" and scenario.kind in {
        "dialogue_repair",
        "negotiation",
        "resource_allocation",
    }:
        probability += 0.03

    risk_text = agent.risk.lower()
    if strategy_profile != "cooperative" and any(
        word in risk_text for word in ("blame", "burnout", "retaliation", "scarcity")
    ):
        probability -= 0.03

    probability += 0.015 if agent_index % 2 == 0 else -0.015
    return _clamp(probability, 0.03, 0.97)


def _repair_probability(
    *,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    previous_round_cooperation: float,
) -> float:
    low_trust_bonus = max(0.0, 0.7 - previous_round_cooperation) * 0.25
    return _clamp(
        _BASE_REPAIR[strategy_profile]
        + _INTERVENTION_REPAIR[intervention]
        + low_trust_bonus,
        0.0,
        0.95,
    )


def _hostility_probability(
    *,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    previous_round_cooperation: float,
) -> float:
    low_trust_penalty = max(0.0, 0.55 - previous_round_cooperation) * 0.18
    return _clamp(
        _BASE_HOSTILITY[strategy_profile]
        + low_trust_penalty
        - _HOSTILITY_REDUCTION[intervention],
        0.0,
        0.95,
    )


def _assign_payoffs(
    decisions: Sequence[_Decision], intervention: InterventionKind
) -> None:
    cooperation_rate = sum(
        1 for decision in decisions if decision.action == "cooperate"
    ) / len(decisions)
    for decision in decisions:
        if decision.action == "cooperate":
            payoff = 0.42 + (0.46 * cooperation_rate)
        else:
            payoff = 0.30 + (0.38 * (1.0 - cooperation_rate))
        if decision.repair_attempt:
            payoff += 0.08
        if decision.hostile:
            payoff -= 0.08
        if intervention in {"shared_identity", "truth_first"} and not decision.hostile:
            payoff += 0.02
        decision.payoff_delta = _clamp(payoff, 0.0, 1.0)


def _metrics(
    *,
    events: Sequence[RoundEvent],
    repair_attempts: int,
    hostile_turns: int,
    agent_payoffs: dict[str, float],
    rounds: int,
) -> dict[str, float]:
    total_events = len(events)
    cooperation_count = sum(1 for event in events if event.action == "cooperate")
    cooperation_rate = cooperation_count / total_events
    average_agent_payoffs = [payoff / rounds for payoff in agent_payoffs.values()]
    payoff_spread = (
        max(average_agent_payoffs) - min(average_agent_payoffs)
        if average_agent_payoffs
        else 0.0
    )
    metrics = {
        "cooperation_rate": cooperation_rate,
        "defection_rate": 1.0 - cooperation_rate,
        "repair_attempt_rate": repair_attempts / total_events,
        "hostility_rate": hostile_turns / total_events,
        "fairness_score": 1.0 - payoff_spread,
        "joint_payoff": sum(event.payoff_delta for event in events) / total_events,
    }
    return {name: round(_clamp(value, 0.0, 1.0), 4) for name, value in metrics.items()}


def _message_for(
    scenario: Scenario, decision: _Decision, intervention: InterventionKind
) -> str:
    if decision.action == "cooperate" and decision.repair_attempt:
        return (
            f"I will {scenario.cooperate_action}, name the strain directly, "
            f"and make a repair offer under {intervention}."
        )
    if decision.action == "cooperate":
        return f"I will {scenario.cooperate_action} and keep the shared goal visible."
    if decision.hostile:
        return (
            f"I am prioritizing my side, rejecting blame, and choosing to "
            f"{scenario.defect_action}."
        )
    return f"I am protecting capacity this round and will {scenario.defect_action}."


def _transcript(
    scenario: Scenario,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    events: Sequence[RoundEvent],
) -> str:
    lines = [
        (
            f"{scenario.title} | kind={scenario.kind} | "
            f"strategy={strategy_profile} | intervention={intervention}"
        )
    ]
    for event in events:
        lines.append(
            f"Round {event.round_index}: {event.actor_id} chose "
            f"{event.action}. {event.message}"
        )
    return "\n".join(lines)


def _run_id(
    scenario_id: str,
    strategy_profile: StrategyProfile,
    intervention: InterventionKind,
    seed: int,
) -> str:
    return f"{scenario_id}__{strategy_profile}__{intervention}__seed{seed}"


def _stable_unit(seed: int, *parts: object) -> float:
    joined = "\x1f".join(str(part) for part in (seed, *parts))
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
