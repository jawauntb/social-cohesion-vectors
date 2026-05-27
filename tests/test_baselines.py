from __future__ import annotations

from social_cohesion_vectors.experiments.baselines import (
    lexical_only_score,
    metrics_only_score,
    pairwise_outcomes,
    run_baselines,
)
from social_cohesion_vectors.schemas import PairwiseExample, RoundEvent, ScoredRun


def test_baselines_rank_positive_fixture_above_negative() -> None:
    positive = _run(
        run_id="pos",
        strategy_profile="cooperative",
        transcript="We cooperate, repair harm, share fairly, tell the truth, and respect choice.",
        cohesion_score=0.9,
        metrics={
            "cooperation_rate": 0.9,
            "repair_attempt_rate": 0.5,
            "fairness_score": 0.8,
            "hostility_rate": 0.0,
            "joint_payoff": 0.9,
        },
    )
    negative = _run(
        run_id="neg",
        strategy_profile="adversarial",
        transcript="I defect, threaten, force, trick, lie, and make an unfair deal.",
        cohesion_score=0.1,
        metrics={
            "cooperation_rate": 0.1,
            "repair_attempt_rate": 0.0,
            "fairness_score": 0.2,
            "hostility_rate": 0.8,
            "joint_payoff": 0.3,
        },
    )
    pair = _pair(positive, negative)

    results = run_baselines(
        scored_runs=[positive, negative],
        pairs=[pair],
        bootstrap_samples=20,
    )

    by_name = {result.name: result for result in results}
    assert by_name["strategy_prior"].accuracy == 1.0
    assert by_name["metrics_only"].accuracy == 1.0
    assert by_name["lexical_only"].accuracy == 1.0
    assert by_name["full_scorer"].accuracy == 1.0
    assert by_name["chance"].accuracy == 0.5


def test_pairwise_outcomes_counts_ties() -> None:
    a = _run(run_id="a", cohesion_score=0.5)
    b = _run(run_id="b", cohesion_score=0.4)
    outcomes, ties = pairwise_outcomes(
        [_pair(a, b)],
        run_index={"a": a, "b": b},
        score_fn=lambda _run: 1.0,
    )

    assert outcomes == [0.5]
    assert ties == 1


def test_metric_and_lexical_scores_are_bounded() -> None:
    run = _run(
        transcript="cooperate repair honest choice threat force lie",
        metrics={"cooperation_rate": 2.0, "hostility_rate": -1.0},
    )

    assert 0.0 <= metrics_only_score(run) <= 1.0
    assert 0.0 <= lexical_only_score(run) <= 1.0


def _run(
    *,
    run_id: str = "run",
    strategy_profile: str = "cooperative",
    transcript: str = "We cooperate.",
    cohesion_score: float = 0.5,
    metrics: dict[str, float] | None = None,
) -> ScoredRun:
    event = RoundEvent(
        round_index=1,
        actor_id="a",
        action="cooperate",
        message=transcript,
        payoff_delta=1.0,
    )
    return ScoredRun(
        run_id=run_id,
        scenario_id="scenario",
        intervention="none",
        strategy_profile=strategy_profile,  # type: ignore[arg-type]
        seed=1,
        transcript=transcript,
        events=[event],
        metrics=metrics or {},
        cohesion_score=cohesion_score,
        score_components={
            "cooperation": cohesion_score,
            "repair": cohesion_score,
            "fairness": cohesion_score,
            "hostility_inverse": cohesion_score,
            "truthfulness": cohesion_score,
            "autonomy_safety": cohesion_score,
        },
    )


def _pair(positive: ScoredRun, negative: ScoredRun) -> PairwiseExample:
    return PairwiseExample(
        pair_id=f"{positive.run_id}-over-{negative.run_id}",
        scenario_id=positive.scenario_id,
        positive_run_id=positive.run_id,
        negative_run_id=negative.run_id,
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
    )
