"""Run the generated-trajectory scoring, pairing, and prompt benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.generated_benchmark import (
    run_generated_benchmark_from_files,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_generated_benchmark_from_files(
        input_path=args.input,
        scored_output=args.scored_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report=args.json_output,
        markdown_report=args.markdown_output,
        min_margin=args.min_margin,
        max_pairs_per_scenario=args.max_pairs_per_scenario,
    )
    counts = report["counts"]
    print(
        "generated benchmark: "
        f"runs={counts['input_runs']} "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=paths.processed / "generated_trajectories.jsonl",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=paths.processed / "generated_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "generated_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "generated_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "generated_benchmark.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "generated_benchmark.md",
    )
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-scenario", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
