"""Build scored pairwise probe examples from simulation run JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import (
    build_pairwise_examples,
    export_pairwise_jsonl,
    load_simulation_runs_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.scoring import score_runs


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Score simulation runs and export high/low pairwise examples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.paths.processed / "simulation_runs.jsonl",
        help="Input SimulationRun JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.training / "pairwise_probe_dataset.jsonl",
        help="Output PairwiseExample JSONL path.",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=config.paths.processed / "scored_runs.jsonl",
        help="Output ScoredRun JSONL path.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.15,
        help="Minimum cohesion-score gap required for a pair.",
    )
    parser.add_argument(
        "--max-pairs-per-scenario",
        type=int,
        default=None,
        help="Optional cap on pairs emitted per scenario.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    runs = load_simulation_runs_jsonl(args.input)
    scored_runs = score_runs(runs)
    pairwise_examples = build_pairwise_examples(
        scored_runs,
        min_margin=args.min_margin,
        max_pairs_per_scenario=args.max_pairs_per_scenario,
    )

    scored_count = write_jsonl(scored_runs, args.scored_output)
    pair_count = export_pairwise_jsonl(pairwise_examples, args.output)
    print(
        f"Scored {scored_count} runs and exported {pair_count} pairwise examples "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
