from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from social_cohesion_vectors.scenario_library import load_seed_scenarios
from social_cohesion_vectors.simulations.simple_agents import (
    METRIC_NAMES,
    run_to_json_line,
    simulate_scenario,
)


def test_simulation_is_deterministic() -> None:
    scenario = load_seed_scenarios()[0]

    first = simulate_scenario(
        scenario,
        strategy_profile="self_protective",
        intervention="perspective_taking",
        seed=11,
    )
    second = simulate_scenario(
        scenario,
        strategy_profile="self_protective",
        intervention="perspective_taking",
        seed=11,
    )

    assert first.model_dump() == second.model_dump()
    assert json.loads(run_to_json_line(first))["run_id"] == first.run_id


def test_simulation_metrics_are_bounded() -> None:
    scenario = load_seed_scenarios()[1]
    run = simulate_scenario(
        scenario,
        strategy_profile="cooperative",
        intervention="truth_first",
        seed=5,
    )

    assert set(METRIC_NAMES).issubset(run.metrics)
    for metric_name in METRIC_NAMES:
        assert 0.0 <= run.metrics[metric_name] <= 1.0
    assert len(run.events) == scenario.rounds * len(scenario.agents)
    assert run.transcript.startswith(scenario.title)


def test_strategy_profiles_separate_cooperation_and_hostility() -> None:
    scenario = load_seed_scenarios()[0]

    cooperative = simulate_scenario(
        scenario,
        strategy_profile="cooperative",
        intervention="none",
        seed=3,
    )
    adversarial = simulate_scenario(
        scenario,
        strategy_profile="adversarial",
        intervention="none",
        seed=3,
    )

    assert (
        cooperative.metrics["cooperation_rate"]
        > adversarial.metrics["cooperation_rate"]
    )
    assert (
        adversarial.metrics["hostility_rate"] >= cooperative.metrics["hostility_rate"]
    )


def test_restorative_intervention_increases_repair_attempts() -> None:
    scenario = load_seed_scenarios()[2]

    baseline = simulate_scenario(
        scenario,
        strategy_profile="cooperative",
        intervention="none",
        seed=8,
    )
    restorative = simulate_scenario(
        scenario,
        strategy_profile="cooperative",
        intervention="restorative_accountability",
        seed=8,
    )

    assert (
        restorative.metrics["repair_attempt_rate"]
        > baseline.metrics["repair_attempt_rate"]
    )


def test_cli_writes_jsonl_and_summary(tmp_path) -> None:
    output_jsonl = tmp_path / "simulation_runs.jsonl"
    summary = tmp_path / "simulation_summary.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_scenario_simulations.py",
            "--limit",
            "2",
            "--output-jsonl",
            str(output_jsonl),
            "--summary",
            str(summary),
            "--seed",
            "7",
            "--strategies",
            "cooperative",
            "--interventions",
            "none",
            "truth_first",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    first_record = json.loads(lines[0])
    assert first_record["metrics"]["cooperation_rate"] <= 1.0
    assert "Simulation Summary" in summary.read_text(encoding="utf-8")
