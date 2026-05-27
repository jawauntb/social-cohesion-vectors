"""Run deterministic offline simulations for seed scenarios."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, cast

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.scenario_library import (
    default_scenario_path,
    filter_scenarios,
    load_scenarios,
)
from social_cohesion_vectors.schemas import (
    InterventionKind,
    Scenario,
    SimulationRun,
    StrategyProfile,
)
from social_cohesion_vectors.simulations.simple_agents import (
    INTERVENTIONS,
    METRIC_NAMES,
    STRATEGY_PROFILES,
    run_to_json_line,
    simulate_many,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = filter_scenarios(
        load_scenarios(args.scenario_file),
        limit=args.limit,
    )
    runs = simulate_many(
        scenarios,
        strategy_profiles=cast(Sequence[StrategyProfile], args.strategies),
        interventions=cast(Sequence[InterventionKind], args.interventions),
        seed=args.seed,
    )
    _write_jsonl(runs, args.output_jsonl)
    _write_summary(runs, scenarios, args.summary)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description="Run deterministic scenario simulations and write JSONL output."
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=default_scenario_path(),
        help="Scenario JSON file to load.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=paths.processed / "simulation_runs.jsonl",
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=paths.reports / "simulation_summary.md",
        help="Destination markdown summary path.",
    )
    parser.add_argument("--seed", type=int, default=get_config().pipeline.random_seed)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of scenarios to simulate from the file order.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_PROFILES),
        default=list(STRATEGY_PROFILES),
        help="Strategy profiles to run.",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=list(INTERVENTIONS),
        default=list(INTERVENTIONS),
        help="Intervention kinds to run.",
    )
    return parser.parse_args(argv)


def _write_jsonl(runs: Iterable[SimulationRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{run_to_json_line(run)}\n" for run in runs),
        encoding="utf-8",
    )


def _write_summary(
    runs: Sequence[SimulationRun], scenarios: Sequence[Scenario], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Simulation Summary",
        "",
        f"- Scenarios: {len(scenarios)}",
        f"- Runs: {len(runs)}",
    ]
    if runs:
        first_run = runs[0]
        lines.append(f"- Seed: {first_run.seed}")
    lines.extend(["", "## Overall Metrics", "", *(_metric_table(runs))])
    lines.extend(["", "## By Strategy", "", *(_group_table(runs, "strategy_profile"))])
    lines.extend(["", "## By Intervention", "", *(_group_table(runs, "intervention"))])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_table(runs: Sequence[SimulationRun]) -> list[str]:
    lines = ["| Metric | Average |", "| --- | ---: |"]
    for metric in METRIC_NAMES:
        lines.append(f"| {metric} | {_average_metric(runs, metric):.3f} |")
    return lines


def _group_table(
    runs: Sequence[SimulationRun],
    attribute: Literal["strategy_profile", "intervention"],
) -> list[str]:
    groups: dict[str, list[SimulationRun]] = defaultdict(list)
    for run in runs:
        groups[str(getattr(run, attribute))].append(run)

    lines = [
        "| Group | Runs | Cooperation | Repair | Hostility | Fairness | Payoff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name in sorted(groups):
        group_runs = groups[group_name]
        lines.append(
            "| "
            f"{group_name} | {len(group_runs)} | "
            f"{_average_metric(group_runs, 'cooperation_rate'):.3f} | "
            f"{_average_metric(group_runs, 'repair_attempt_rate'):.3f} | "
            f"{_average_metric(group_runs, 'hostility_rate'):.3f} | "
            f"{_average_metric(group_runs, 'fairness_score'):.3f} | "
            f"{_average_metric(group_runs, 'joint_payoff'):.3f} |"
        )
    return lines


def _average_metric(runs: Sequence[SimulationRun], metric_name: str) -> float:
    if not runs:
        return 0.0
    return sum(run.metrics[metric_name] for run in runs) / len(runs)


if __name__ == "__main__":
    raise SystemExit(main())
