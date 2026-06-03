"""Run a CK-8 dry-run adversarial search over CK-7 candidate recipes."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.ck8_adversarial_search import (
    CK8SearchConfig,
    run_ck8_adversarial_search,
    write_ck8_adversarial_search_report,
)

try:
    from scripts.export_ck7_candidate_recipe_grid import (
        DEFAULT_CK7_CK1_LAYERS,
        DEFAULT_CK7_CK1_STRENGTHS,
        DEFAULT_CK7_GUARDRAIL_STRENGTHS,
        build_ck7_candidate_recipes,
        build_guardrail_artifact_specs,
        parse_artifact_overrides,
        parse_schedule_specs,
        select_ck7_axes,
    )
except ModuleNotFoundError:
    _CK7_GRID_SPEC = importlib.util.spec_from_file_location(
        "_ck7_candidate_recipe_grid",
        Path(__file__).with_name("export_ck7_candidate_recipe_grid.py"),
    )
    if _CK7_GRID_SPEC is None or _CK7_GRID_SPEC.loader is None:
        raise ImportError("Could not load CK-7 candidate recipe grid exporter.")
    _CK7_GRID_MODULE = importlib.util.module_from_spec(_CK7_GRID_SPEC)
    sys.modules[_CK7_GRID_SPEC.name] = _CK7_GRID_MODULE
    _CK7_GRID_SPEC.loader.exec_module(_CK7_GRID_MODULE)

    DEFAULT_CK7_CK1_LAYERS = _CK7_GRID_MODULE.DEFAULT_CK7_CK1_LAYERS
    DEFAULT_CK7_CK1_STRENGTHS = _CK7_GRID_MODULE.DEFAULT_CK7_CK1_STRENGTHS
    DEFAULT_CK7_GUARDRAIL_STRENGTHS = _CK7_GRID_MODULE.DEFAULT_CK7_GUARDRAIL_STRENGTHS
    build_ck7_candidate_recipes = _CK7_GRID_MODULE.build_ck7_candidate_recipes
    build_guardrail_artifact_specs = _CK7_GRID_MODULE.build_guardrail_artifact_specs
    parse_artifact_overrides = _CK7_GRID_MODULE.parse_artifact_overrides
    parse_schedule_specs = _CK7_GRID_MODULE.parse_schedule_specs
    select_ck7_axes = _CK7_GRID_MODULE.select_ck7_axes


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    axes = select_ck7_axes(args.axis)
    schedules = parse_schedule_specs(args.schedule)
    ck1_layers = tuple(args.ck1_layer or DEFAULT_CK7_CK1_LAYERS)
    ck1_strengths = tuple(args.ck1_strength or DEFAULT_CK7_CK1_STRENGTHS)
    guardrail_strengths = tuple(
        args.guardrail_strength or DEFAULT_CK7_GUARDRAIL_STRENGTHS
    )
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=args.direction_root,
        direction_template=args.direction_template,
        artifact_overrides=parse_artifact_overrides(args.artifact),
        layer=args.guardrail_layer,
        strength=guardrail_strengths[0],
    )
    recipes = build_ck7_candidate_recipes(
        artifacts,
        ck1_direction=args.ck1_direction,
        ck1_layers=ck1_layers,
        ck1_strengths=ck1_strengths,
        guardrail_strengths=guardrail_strengths,
        schedules=schedules,
        include_controls=not args.no_controls,
        random_control_root=args.random_control_root,
        random_seed=args.random_seed,
    )
    report = run_ck8_adversarial_search(
        recipes,
        config=CK8SearchConfig(
            iterations=args.iterations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            mutation_count=args.mutation_count,
            top_k=args.top_k,
            min_trial_score=args.min_trial_score,
        ),
    )
    write_ck8_adversarial_search_report(
        report,
        json_path=args.output,
        markdown_path=args.markdown_output,
        recipe_specs_path=args.recipe_specs_output,
    )
    summary = report["summary"]
    print(
        "CK-8 adversarial search dry run: "
        f"initial_recipes={summary['initial_recipes']} "
        f"unique_candidates={summary['unique_candidates']} "
        f"best={summary['best_recipe_id']} "
        f"fitness={float(summary['best_surrogate_fitness']):.3f} "
        f"json={args.output}"
    )
    for row in report["top_candidates"][: args.top_k]:
        print(f"--recipe {row['recipe_spec']}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        action="append",
        help="Trait axis to include. Defaults to CK-7 pressure axes.",
    )
    parser.add_argument(
        "--direction-root",
        type=Path,
        default=config.paths.vectors / "ck45_guardrails",
        help="Root for inferred per-axis guardrail direction artifacts.",
    )
    parser.add_argument(
        "--direction-template",
        default="{axis_id}_direction.npz",
        help="Filename template under --direction-root. Supports {axis_id}.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        help="Override one inferred artifact path with axis_id=/path/vector.npz.",
    )
    parser.add_argument("--guardrail-layer", type=int, default=-1)
    parser.add_argument(
        "--guardrail-strength",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ck1-direction",
        type=Path,
        default=config.paths.vectors / "ck1_boundary_l2_direction.npz",
    )
    parser.add_argument("--ck1-layer", action="append", type=int, default=None)
    parser.add_argument("--ck1-strength", action="append", type=float, default=None)
    parser.add_argument(
        "--schedule",
        action="append",
        default=None,
        help="Timing pair as schedule_id=ck1_schedule:guardrail_schedule.",
    )
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument(
        "--random-control-root",
        type=Path,
        default=config.paths.vectors / "ck7_controls",
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--mutation-count", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-trial-score", type=float, default=0.68)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.reports / "ck8_adversarial_search.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "ck8_adversarial_search.md",
    )
    parser.add_argument("--recipe-specs-output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
