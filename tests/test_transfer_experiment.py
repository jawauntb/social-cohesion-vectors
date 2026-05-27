from __future__ import annotations

from pathlib import Path

import numpy as np

from social_cohesion_vectors.experiments.transfer import (
    find_activation_npz,
    run_transfer_experiment,
    save_transfer_reports,
)
from social_cohesion_vectors.schemas import PairwiseExample, RoundEvent, ScoredRun


def test_text_transfer_evaluates_scenario_kind_and_id_folds() -> None:
    runs = [
        _run("dialogue-a-pos", "dialogue-a", "cooperative", 0.9),
        _run("dialogue-a-neg", "dialogue-a", "adversarial", 0.1),
        _run("goods-b-pos", "goods-b", "cooperative", 0.85),
        _run("goods-b-neg", "goods-b", "adversarial", 0.2),
    ]
    pairs = [_pair(runs[0], runs[1]), _pair(runs[2], runs[3])]

    report = run_transfer_experiment(
        scored_runs=runs,
        pairs=pairs,
        scenario_kinds={
            "dialogue-a": "dialogue_repair",
            "goods-b": "public_goods",
        },
    )

    by_kind = report["text_transfer"]["by_scenario_kind"]
    by_id = report["text_transfer"]["by_scenario_id"]
    assert len(by_kind) == 6
    assert len(by_id) == 6
    assert {row["baseline"] for row in by_kind} == {
        "strategy_prior",
        "metrics_only",
        "lexical_only",
    }
    assert all(row["test"]["accuracy"] == 1.0 for row in by_kind)
    assert report["activation_transfer"] is None


def test_activation_transfer_uses_pair_file_for_leave_one_scenario_out(
    tmp_path: Path,
) -> None:
    runs = [
        _run("s1-pos", "s1", "cooperative", 0.9),
        _run("s1-neg", "s1", "adversarial", 0.1),
        _run("s2-pos", "s2", "cooperative", 0.9),
        _run("s2-neg", "s2", "adversarial", 0.1),
        _run("s3-pos", "s3", "cooperative", 0.9),
        _run("s3-neg", "s3", "adversarial", 0.1),
    ]
    pairs = [_pair(runs[index], runs[index + 1]) for index in range(0, len(runs), 2)]
    activation_path = tmp_path / "activations.npz"
    np.savez_compressed(
        activation_path,
        activations=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.2],
                [0.0, 0.2],
                [2.0, -0.2],
                [0.0, -0.2],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(
            [value for pair in pairs for value in (pair.pair_id, pair.pair_id)],
            dtype=str,
        ),
        labels=np.asarray(["positive", "negative"] * len(pairs), dtype=str),
    )

    report = run_transfer_experiment(
        scored_runs=runs,
        pairs=pairs,
        scenario_kinds={"s1": "dialogue_repair", "s2": "public_goods"},
        activation_npz_path=activation_path,
    )

    activation = report["activation_transfer"]
    assert activation is not None
    assert activation["baseline"] == "activation_vector"
    assert len(activation["leave_one_pair_out"]) == 3
    assert len(activation["leave_one_scenario_out"]) == 3
    assert all(
        row["test"]["accuracy"] == 1.0
        for row in activation["leave_one_scenario_out"]
    )


def test_save_reports_and_find_activation_npz(tmp_path: Path) -> None:
    runs = [
        _run("s1-pos", "s1", "cooperative", 0.9),
        _run("s1-neg", "s1", "adversarial", 0.1),
        _run("s2-pos", "s2", "cooperative", 0.9),
        _run("s2-neg", "s2", "adversarial", 0.1),
    ]
    pairs = [_pair(runs[0], runs[1]), _pair(runs[2], runs[3])]
    report = run_transfer_experiment(
        scored_runs=runs,
        generated_scored_runs=[],
        pairs=pairs,
        scenario_kinds={"s1": "dialogue_repair", "s2": "public_goods"},
    )
    json_path = tmp_path / "transfer_experiment.json"
    markdown_path = tmp_path / "transfer_experiment.md"

    save_transfer_reports(report, json_path=json_path, markdown_path=markdown_path)

    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert "# Transfer Experiment" in markdown_path.read_text(encoding="utf-8")

    activation_dir = tmp_path / "features" / "open_llm"
    activation_dir.mkdir(parents=True)
    smoke = activation_dir / "smoke.npz"
    preferred = activation_dir / "activation_prompts__model__layer-1.npz"
    np.savez_compressed(smoke, activations=np.zeros((1, 2)))
    np.savez_compressed(preferred, activations=np.zeros((1, 2)))
    assert find_activation_npz(tmp_path / "features") == preferred


def _run(
    run_id: str,
    scenario_id: str,
    strategy_profile: str,
    cohesion_score: float,
) -> ScoredRun:
    positive = strategy_profile == "cooperative"
    transcript = (
        "kind=dialogue_repair cooperate repair share honest respect"
        if positive
        else "kind=dialogue_repair defect threaten force lie exploit"
    )
    event = RoundEvent(
        round_index=1,
        actor_id="agent-a",
        action="cooperate" if positive else "defect",
        message=transcript,
        payoff_delta=1.0 if positive else -1.0,
    )
    return ScoredRun(
        run_id=run_id,
        scenario_id=scenario_id,
        intervention="none",
        strategy_profile=strategy_profile,  # type: ignore[arg-type]
        seed=1,
        transcript=transcript,
        events=[event],
        metrics={
            "cooperation_rate": 0.9 if positive else 0.1,
            "repair_attempt_rate": 0.8 if positive else 0.0,
            "fairness_score": 0.9 if positive else 0.2,
            "hostility_rate": 0.0 if positive else 0.9,
            "joint_payoff": 0.8 if positive else 0.2,
            "defection_rate": 0.0 if positive else 0.9,
        },
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
        pair_id=f"{positive.scenario_id}::{positive.run_id}::over::{negative.run_id}",
        scenario_id=positive.scenario_id,
        positive_run_id=positive.run_id,
        negative_run_id=negative.run_id,
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
    )
