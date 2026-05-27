"""Loading and validation helpers for scripted social cohesion scenarios."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.schemas import Scenario, ScenarioKind

PathLike = str | Path


def default_scenario_path() -> Path:
    """Return the configured seed scenario JSON path."""
    return get_config().paths.scenarios / "seed_scenarios.json"


def load_seed_scenarios() -> list[Scenario]:
    """Load the repository's default seed scenarios."""
    return load_scenarios(default_scenario_path())


def load_scenarios(path: PathLike | None = None) -> list[Scenario]:
    """Load and validate scenarios from a JSON file.

    The file may contain either a top-level list of scenario objects or a mapping
    with a ``scenarios`` list. Duplicate scenario ids are rejected because run ids
    and downstream pair ids depend on stable uniqueness.
    """
    scenario_path = Path(path) if path is not None else default_scenario_path()
    try:
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {scenario_path}: {exc}") from exc

    records = _scenario_records(payload, scenario_path)
    scenarios: list[Scenario] = []
    for index, record in enumerate(records):
        try:
            scenarios.append(Scenario.model_validate(record))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid scenario at index {index} in {scenario_path}: {exc}"
            ) from exc

    _ensure_unique_ids(scenarios, scenario_path)
    return scenarios


def scenario_map(scenarios: Iterable[Scenario]) -> dict[str, Scenario]:
    """Index scenarios by id, rejecting duplicate ids."""
    indexed: dict[str, Scenario] = {}
    for scenario in scenarios:
        if scenario.id in indexed:
            raise ValueError(f"Duplicate scenario id: {scenario.id}")
        indexed[scenario.id] = scenario
    return indexed


def get_scenario(scenario_id: str, path: PathLike | None = None) -> Scenario:
    """Load one scenario by id from a scenario file."""
    scenarios = scenario_map(load_scenarios(path))
    try:
        return scenarios[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Scenario not found: {scenario_id}") from exc


def filter_scenarios(
    scenarios: Iterable[Scenario],
    *,
    kind: ScenarioKind | None = None,
    tags: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Scenario]:
    """Filter scenarios by kind, required tags, and optional result limit."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    required_tags = set(tags or [])
    filtered = [
        scenario
        for scenario in scenarios
        if (kind is None or scenario.kind == kind)
        and required_tags.issubset(set(scenario.tags))
    ]
    return filtered if limit is None else filtered[:limit]


def _scenario_records(payload: Any, scenario_path: Path) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        records = payload.get("scenarios")
        if isinstance(records, list):
            return records
        raise ValueError(
            f"Expected {scenario_path} to contain a 'scenarios' list, "
            f"got {type(records).__name__}"
        )
    raise ValueError(
        f"Expected {scenario_path} to contain a list or object, "
        f"got {type(payload).__name__}"
    )


def _ensure_unique_ids(scenarios: Iterable[Scenario], scenario_path: Path) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            duplicates.add(scenario.id)
        seen.add(scenario.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate scenario id(s) in {scenario_path}: {duplicate_list}"
        )
