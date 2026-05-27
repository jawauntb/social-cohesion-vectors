"""Run local baseline experiments over the pairwise probe dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.baselines import (
    run_baselines_from_files,
    save_baseline_reports,
)


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs",
        type=Path,
        default=config.paths.processed / "scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=config.paths.training / "pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "baseline_experiments.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "baseline_experiments.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=config.pipeline.random_seed)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_baselines_from_files(
        scored_runs_path=args.scored_runs,
        pairs_path=args.pairs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    save_baseline_reports(
        results,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    for result in results:
        print(
            f"{result.name}: acc={result.accuracy:.3f} "
            f"ci=[{result.ci_low:.3f}, {result.ci_high:.3f}] "
            f"n={result.n_pairs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

