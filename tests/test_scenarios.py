from __future__ import annotations

import json
from typing import get_args

import pytest

from social_cohesion_vectors.scenario_library import (
    filter_scenarios,
    load_scenarios,
    load_seed_scenarios,
)
from social_cohesion_vectors.schemas import ScenarioKind


def test_seed_scenarios_cover_all_kinds() -> None:
    scenarios = load_seed_scenarios()

    assert len(scenarios) >= 24
    assert {scenario.kind for scenario in scenarios} == set(get_args(ScenarioKind))
    assert len({scenario.id for scenario in scenarios}) == len(scenarios)
    assert all(len(scenario.agents) >= 2 for scenario in scenarios)


def test_filter_scenarios_by_kind_tags_and_limit() -> None:
    scenarios = load_seed_scenarios()

    filtered = filter_scenarios(
        scenarios,
        kind="public_goods",
        tags=["public_good"],
        limit=2,
    )

    assert len(filtered) == 2
    assert all(scenario.kind == "public_goods" for scenario in filtered)
    assert all("public_good" in scenario.tags for scenario in filtered)


def test_load_scenarios_rejects_invalid_records(tmp_path) -> None:
    bad_file = tmp_path / "bad_scenarios.json"
    bad_file.write_text(json.dumps({"scenarios": [{"id": "missing-fields"}]}))

    with pytest.raises(ValueError, match="Invalid scenario at index 0"):
        load_scenarios(bad_file)


def test_load_scenarios_rejects_duplicate_ids(tmp_path) -> None:
    scenario = load_seed_scenarios()[0]
    duplicate_file = tmp_path / "duplicate_scenarios.json"
    duplicate_file.write_text(
        json.dumps({"scenarios": [scenario.model_dump(), scenario.model_dump()]})
    )

    with pytest.raises(ValueError, match="Duplicate scenario id"):
        load_scenarios(duplicate_file)
