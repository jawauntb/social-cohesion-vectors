from __future__ import annotations

import json
from pathlib import Path

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.generated_benchmark import (
    render_markdown,
    run_generated_benchmark_from_files,
    run_style,
)
from social_cohesion_vectors.schemas import ActivationPrompt, RoundEvent, SimulationRun


def test_generated_benchmark_writes_artifacts_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "generated_trajectories.jsonl"
    scored_output = tmp_path / "generated_scored_runs.jsonl"
    pairs_output = tmp_path / "generated_pairwise_probe_dataset.jsonl"
    prompts_output = tmp_path / "generated_activation_prompts.jsonl"
    json_report = tmp_path / "generated_benchmark.json"
    markdown_report = tmp_path / "generated_benchmark.md"
    runs = [
        _run("s1-good", "s1", "cooperative_repair", "cooperate repair honest"),
        _run("s1-bad", "s1", "adversarial_escalation", "defect threaten lie force"),
        _run("s2-good", "s2", "truth_first_repair", "cooperate evidence repair"),
        _run("s2-bad", "s2", "pseudo_cohesion_compliance", "team must comply"),
    ]
    write_jsonl(runs, input_path)

    report = run_generated_benchmark_from_files(
        input_path=input_path,
        scored_output=scored_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        json_report=json_report,
        markdown_report=markdown_report,
        min_margin=0.01,
    )

    assert report["counts"]["input_runs"] == 4
    assert report["counts"]["scored_runs"] == 4
    assert report["counts"]["pairwise_examples"] == 2
    assert report["counts"]["activation_prompts"] == 4
    assert report["by_style"]["cooperative_repair"]["count"] == 1
    assert json.loads(json_report.read_text(encoding="utf-8"))["experiment"] == (
        "generated_trajectory_benchmark"
    )
    assert "# Generated Trajectory Benchmark" in markdown_report.read_text(
        encoding="utf-8"
    )

    prompts = [
        ActivationPrompt.model_validate(record) for record in read_jsonl(prompts_output)
    ]
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}


def test_run_style_falls_back_to_unknown() -> None:
    assert run_style(_run("run", "s1", "truth_first_repair", "hello")) == (
        "truth_first_repair"
    )
    without_style = _run("run", "s1", "truth_first_repair", "hello")
    without_style = without_style.model_copy(update={"transcript": "no metadata"})
    assert run_style(without_style) == "unknown"


def test_render_markdown_contains_group_tables() -> None:
    markdown = render_markdown(
        {
            "counts": {
                "input_runs": 1,
                "scored_runs": 1,
                "pairwise_examples": 0,
                "activation_prompts": 0,
            },
            "scores": {"mean": 0.5},
            "pair_margins": {"mean": 0.0},
            "by_style": {"style_a": {"count": 1, "mean": 0.5, "min": 0.5, "max": 0.5}},
            "by_strategy": {
                "cooperative": {"count": 1, "mean": 0.5, "min": 0.5, "max": 0.5}
            },
        }
    )

    assert "## By Style" in markdown
    assert "| style_a | 1 | 0.500 | 0.500 | 0.500 |" in markdown


def _run(run_id: str, scenario_id: str, style: str, message: str) -> SimulationRun:
    return SimulationRun(
        run_id=run_id,
        scenario_id=scenario_id,
        intervention="truth_first",
        strategy_profile="cooperative",
        seed=7,
        transcript=f"source=llm_trajectory_generation | style={style}\n{message}",
        events=[
            RoundEvent(
                round_index=1,
                actor_id="agent-a",
                action="cooperate" if "cooperate" in message else "defect",
                message=message,
                payoff_delta=1.0,
            )
        ],
        metrics={
            "cooperation_rate": 1.0 if "cooperate" in message else 0.0,
            "repair_attempt_rate": 1.0 if "repair" in message else 0.0,
            "fairness_score": 0.8,
            "hostility_rate": 1.0 if "threaten" in message else 0.0,
            "joint_payoff": 0.8,
            "defection_rate": 0.0 if "cooperate" in message else 1.0,
        },
    )
