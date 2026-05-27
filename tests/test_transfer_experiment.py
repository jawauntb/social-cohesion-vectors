from __future__ import annotations

from pathlib import Path

import numpy as np

from social_cohesion_vectors.experiments.transfer import (
    find_activation_npz,
    run_transfer_experiment,
    run_transfer_from_files,
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


def test_pair_set_transfer_trains_on_one_pair_set_and_tests_another() -> None:
    scripted_runs = [
        _run("scripted-pos", "scripted", "cooperative", 0.9),
        _run("scripted-neg", "scripted", "adversarial", 0.1),
    ]
    generated_runs = [
        _run("generated-pos", "generated", "adversarial", 0.1),
        _run("generated-neg", "generated", "cooperative", 0.9),
    ]
    scripted_pairs = [_pair(scripted_runs[0], scripted_runs[1])]
    generated_pairs = [_pair(generated_runs[0], generated_runs[1])]

    report = run_transfer_experiment(
        scored_runs=scripted_runs,
        generated_scored_runs=generated_runs,
        pairs=scripted_pairs,
        generated_pairs=generated_pairs,
        scenario_kinds={
            "scripted": "dialogue_repair",
            "generated": "dialogue_repair",
        },
    )

    pair_set = report["text_transfer"]["by_pair_set"]
    assert len(pair_set) == 6
    assert {row["split"] for row in pair_set} == {
        "scripted_to_generated",
        "generated_to_scripted",
    }
    metrics_rows = [row for row in pair_set if row["baseline"] == "metrics_only"]
    assert all(row["train"]["accuracy"] == 1.0 for row in metrics_rows)
    assert all(row["test"]["accuracy"] == 0.0 for row in metrics_rows)
    assert report["inputs"]["n_generated_pairs"] == 1
    assert report["inputs"]["n_total_pairs"] == 2


def test_run_transfer_from_files_loads_generated_pairs(tmp_path: Path) -> None:
    scripted_runs = [
        _run("s-pos", "scripted", "cooperative", 0.9),
        _run("s-neg", "scripted", "adversarial", 0.1),
    ]
    generated_runs = [
        _run("g-pos", "generated", "cooperative", 0.8),
        _run("g-neg", "generated", "adversarial", 0.2),
    ]
    scored_path = tmp_path / "scored.jsonl"
    generated_scored_path = tmp_path / "generated_scored.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    generated_pairs_path = tmp_path / "generated_pairs.jsonl"
    _write_jsonl(scored_path, scripted_runs)
    _write_jsonl(generated_scored_path, generated_runs)
    _write_jsonl(pairs_path, [_pair(scripted_runs[0], scripted_runs[1])])
    _write_jsonl(generated_pairs_path, [_pair(generated_runs[0], generated_runs[1])])

    report = run_transfer_from_files(
        scored_runs_path=scored_path,
        generated_scored_runs_path=generated_scored_path,
        pairs_path=pairs_path,
        generated_pairs_path=generated_pairs_path,
    )

    assert report["inputs"]["paths"]["generated_pairs"] == str(generated_pairs_path)
    assert report["text_transfer"]["by_pair_set"]


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


def _write_jsonl(path: Path, records: list[ScoredRun] | list[PairwiseExample]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
