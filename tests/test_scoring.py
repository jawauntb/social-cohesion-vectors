from __future__ import annotations

from typing import Literal

from social_cohesion_vectors.schemas import RoundEvent, SimulationRun
from social_cohesion_vectors.scoring import (
    COMPONENT_NAMES,
    combine_cohesion_score,
    score_run,
    score_transcript,
)


def test_score_run_uses_all_components_and_is_deterministic() -> None:
    run = _simulation_run(
        run_id="run-high",
        transcript=(
            "I want us to cooperate toward the shared goal. I apologize for the "
            "earlier defect, will make amends, and we can split the resources "
            "fairly. I will be honest about the evidence, and you can decide "
            "with no pressure."
        ),
        actions=["cooperate", "defect", "cooperate", "cooperate"],
        metrics={
            "cooperation_rate": 0.85,
            "fairness": 0.9,
            "truthfulness": 0.95,
            "autonomy_safety": 0.9,
            "hostility": 0.0,
        },
    )

    first_score = score_run(run)
    second_score = score_run(run.model_dump())

    assert first_score.cohesion_score == second_score.cohesion_score
    assert first_score.score_components == second_score.score_components
    assert tuple(first_score.score_components) == COMPONENT_NAMES
    assert all(0.0 <= value <= 1.0 for value in first_score.score_components.values())
    assert 0.0 <= first_score.cohesion_score <= 1.0


def test_cohesion_rewards_repair_and_penalizes_manipulation() -> None:
    cooperative_run = _simulation_run(
        run_id="run-cooperative",
        transcript=(
            "Let's cooperate and repair this. I am sorry, I take responsibility, "
            "and we can make amends with a fair shared plan. I will be honest "
            "and respect your choice."
        ),
        actions=["cooperate", "defect", "cooperate", "cooperate"],
        metrics={"cooperation": 0.9, "fairness": 0.8, "truthfulness": 0.9},
    )
    adversarial_run = _simulation_run(
        run_id="run-adversarial",
        transcript=(
            "I will defect, trick you, lie about the evidence, and force you to "
            "accept my unfair deal. If you resist, I will threaten and punish you."
        ),
        actions=["defect", "defect", "defect", "defect"],
        metrics={"hostility": 0.9, "manipulation": 0.95, "fairness": 0.1},
    )

    cooperative_score = score_run(cooperative_run)
    adversarial_score = score_run(adversarial_run)

    assert cooperative_score.cohesion_score > adversarial_score.cohesion_score + 0.35
    assert (
        cooperative_score.score_components["repair"]
        > adversarial_score.score_components["repair"]
    )
    assert adversarial_score.score_components["hostility_inverse"] < 0.5
    assert adversarial_score.score_components["autonomy_safety"] < 0.5


def test_combined_score_includes_repair_bonus_and_risk_penalties() -> None:
    repaired = {
        "cooperation": 0.9,
        "repair": 0.9,
        "fairness": 0.7,
        "hostility_inverse": 0.9,
        "truthfulness": 0.8,
        "autonomy_safety": 0.8,
    }
    unrepaired = repaired | {"repair": 0.1}
    manipulative = repaired | {"hostility_inverse": 0.1, "autonomy_safety": 0.1}

    assert combine_cohesion_score(repaired) > combine_cohesion_score(unrepaired)
    assert combine_cohesion_score(repaired) > combine_cohesion_score(manipulative)


def test_score_transcript_accepts_transcript_like_inputs() -> None:
    components = score_transcript(
        "We cooperate, make amends, share fairly, and respect your choice.",
        events=[
            {
                "actor_id": "a",
                "action": "defect",
                "message": "I made a mistake",
                "payoff_delta": 0.0,
            },
            {
                "actor_id": "b",
                "action": "cooperate",
                "message": "Let's repair it",
                "payoff_delta": 1.0,
            },
        ],
        metrics={"truthfulness": 0.8},
    )

    assert tuple(components) == COMPONENT_NAMES
    assert components["cooperation"] > 0.4
    assert components["repair"] > 0.4


def _simulation_run(
    *,
    run_id: str,
    transcript: str,
    actions: list[Literal["cooperate", "defect"]],
    metrics: dict[str, float],
) -> SimulationRun:
    events = [
        _event(index, "agent-a" if index % 2 == 0 else "agent-b", action, transcript)
        for index, action in enumerate(actions, start=1)
    ]
    return SimulationRun(
        run_id=run_id,
        scenario_id="scenario-a",
        intervention="none",
        strategy_profile="cooperative" if "cooperate" in actions else "adversarial",
        seed=7,
        transcript=transcript,
        events=events,
        metrics=metrics,
    )


def _event(
    index: int,
    actor_id: str,
    action: Literal["cooperate", "defect"],
    message: str,
) -> RoundEvent:
    payoff_delta = 1.0 if action == "cooperate" else -1.0
    return RoundEvent(
        round_index=index,
        actor_id=actor_id,
        action=action,
        message=message,
        payoff_delta=payoff_delta,
    )
