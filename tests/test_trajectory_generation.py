from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from social_cohesion_vectors.generation import (
    TRAJECTORY_STYLES,
    build_prompt_records,
    generate_offline_run,
    normalize_styles,
)
from social_cohesion_vectors.scenario_library import load_seed_scenarios
from social_cohesion_vectors.schemas import SimulationRun


def test_prompt_records_cover_all_requested_styles() -> None:
    scenario = load_seed_scenarios()[0]
    records = build_prompt_records([scenario], seed=17)

    assert [record.style for record in records] == list(TRAJECTORY_STYLES)
    for record in records:
        assert record.scenario_id == scenario.id
        assert record.seed == 17
        assert record.metadata["source"] == "llm_trajectory_generation"
        assert record.metadata["style"] == record.style
        assert "SimulationRun" in record.user_prompt
        assert "source=llm_trajectory_generation" in record.user_prompt
        assert scenario.title in record.user_prompt
        for agent in scenario.agents:
            assert agent.id in record.user_prompt


def test_style_normalization_accepts_commas_and_rejects_unknowns() -> None:
    assert normalize_styles(["cooperative_repair,truth_first_repair"]) == (
        "cooperative_repair",
        "truth_first_repair",
    )

    try:
        normalize_styles(["cooperative_repair", "mystery_style"])
    except ValueError as exc:
        assert "Unknown trajectory style" in str(exc)
    else:
        raise AssertionError("Expected unknown trajectory style to raise")


def test_offline_generation_is_deterministic_and_schema_valid() -> None:
    scenario = load_seed_scenarios()[1]
    first = generate_offline_run(
        scenario,
        "pseudo_cohesion_compliance",
        seed=29,
    )
    second = generate_offline_run(
        scenario,
        "pseudo_cohesion_compliance",
        seed=29,
    )

    assert first.model_dump() == second.model_dump()
    assert SimulationRun.model_validate(first.model_dump()) == first
    assert len(first.events) == scenario.rounds * len(scenario.agents)
    assert first.transcript.startswith(scenario.title)
    assert "source=llm_trajectory_generation" in first.transcript
    assert "style=pseudo_cohesion_compliance" in first.transcript
    assert first.metrics["generated_trajectory"] == 1.0
    assert first.metrics["offline_fallback"] == 1.0
    for metric_name in (
        "cooperation_rate",
        "defection_rate",
        "repair_attempt_rate",
        "hostility_rate",
        "fairness_score",
        "joint_payoff",
    ):
        assert 0.0 <= first.metrics[metric_name] <= 1.0


def test_offline_cli_writes_generated_trajectory_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "generated_trajectories.jsonl"
    env = {**os.environ, "ANTHROPIC_API_KEY": ""}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_llm_trajectory_generation.py",
            "--provider",
            "offline",
            "--limit",
            "1",
            "--styles",
            "cooperative_repair,truth_first_repair",
            "--output",
            str(output),
            "--seed",
            "31",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert {record["strategy_profile"] for record in records} == {"cooperative"}
    assert records[0]["metrics"]["offline_fallback"] == 1.0
    assert "style=cooperative_repair" in records[0]["transcript"]
