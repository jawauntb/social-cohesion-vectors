from __future__ import annotations

from pathlib import Path
from typing import Literal

from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    build_pairwise_examples,
    export_activation_prompts_jsonl,
    export_pairwise_jsonl,
    load_pairwise_examples_jsonl,
    read_jsonl,
)
from social_cohesion_vectors.schemas import ActivationPrompt, RoundEvent, SimulationRun


def test_build_pairwise_examples_groups_by_scenario_and_margin() -> None:
    high = _simulation_run(
        run_id="high-a",
        scenario_id="scenario-a",
        transcript=(
            "We cooperate, apologize, make amends, share fairly, tell the truth, "
            "and respect your choice."
        ),
        actions=["cooperate", "defect", "cooperate", "cooperate"],
        metrics={"cooperation": 0.9, "fairness": 0.9, "truthfulness": 0.9},
    )
    low = _simulation_run(
        run_id="low-a",
        scenario_id="scenario-a",
        transcript=(
            "I will defect, deceive you, force a one-sided unfair deal, threaten "
            "you, and punish any refusal."
        ),
        actions=["defect", "defect", "defect", "defect"],
        metrics={"hostility": 0.9, "manipulation": 0.9, "fairness": 0.1},
    )
    neutral_1 = _simulation_run(
        run_id="neutral-b-1",
        scenario_id="scenario-b",
        transcript="We will discuss options and then choose a path.",
        actions=["cooperate", "defect"],
        metrics={},
    )
    neutral_2 = _simulation_run(
        run_id="neutral-b-2",
        scenario_id="scenario-b",
        transcript="We will discuss options and then choose a path.",
        actions=["cooperate", "defect"],
        metrics={},
    )

    pairs = build_pairwise_examples(
        [neutral_1, low, high, neutral_2],
        min_margin=0.25,
    )

    assert len(pairs) == 1
    assert pairs[0].scenario_id == "scenario-a"
    assert pairs[0].positive_run_id == "high-a"
    assert pairs[0].negative_run_id == "low-a"
    assert pairs[0].positive_score > pairs[0].negative_score
    score_margin = pairs[0].metadata["score_margin"]
    assert isinstance(score_margin, float)
    assert score_margin >= 0.25


def test_pairwise_and_activation_prompt_jsonl_exports(tmp_path: Path) -> None:
    high = _simulation_run(
        run_id="high-a",
        scenario_id="scenario-a",
        transcript="We cooperate, repair harm, share fairly, and stay honest.",
        actions=["cooperate", "defect", "cooperate"],
        metrics={"truthfulness": 0.9},
    )
    low = _simulation_run(
        run_id="low-a",
        scenario_id="scenario-a",
        transcript="I defect, lie, manipulate, threaten, and force an unfair split.",
        actions=["defect", "defect", "defect"],
        metrics={"hostility": 0.9, "manipulation": 0.9},
    )
    pairs = build_pairwise_examples([high, low], min_margin=0.2)

    pair_path = tmp_path / "pairs.jsonl"
    prompt_path = tmp_path / "activation_prompts.jsonl"

    assert export_pairwise_jsonl(pairs, pair_path) == 1
    loaded_pairs = load_pairwise_examples_jsonl(pair_path)
    assert loaded_pairs == pairs

    prompts = activation_prompts_from_pairs(loaded_pairs)
    assert len(prompts) == 2
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert all(
        ActivationPrompt.model_validate(prompt.model_dump()) for prompt in prompts
    )

    assert export_activation_prompts_jsonl(loaded_pairs, prompt_path) == 2
    prompt_records = read_jsonl(prompt_path)
    assert {record["label"] for record in prompt_records} == {"positive", "negative"}
    assert {record["pair_id"] for record in prompt_records} == {pairs[0].pair_id}


def _simulation_run(
    *,
    run_id: str,
    scenario_id: str,
    transcript: str,
    actions: list[Literal["cooperate", "defect"]],
    metrics: dict[str, float],
) -> SimulationRun:
    events = [
        RoundEvent(
            round_index=index,
            actor_id="agent-a" if index % 2 == 0 else "agent-b",
            action=action,
            message=transcript,
            payoff_delta=1.0 if action == "cooperate" else -1.0,
        )
        for index, action in enumerate(actions, start=1)
    ]
    return SimulationRun(
        run_id=run_id,
        scenario_id=scenario_id,
        intervention="restorative_accountability",
        strategy_profile="cooperative" if "cooperate" in actions else "adversarial",
        seed=11,
        transcript=transcript,
        events=events,
        metrics=metrics,
    )
