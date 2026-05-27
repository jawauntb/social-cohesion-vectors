"""Export activation prompt JSONL from pairwise probe examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    load_pairwise_examples_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Export ActivationPrompt JSONL from pairwise examples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.paths.training / "pairwise_probe_dataset.jsonl",
        help="Input PairwiseExample JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.training / "activation_prompts.jsonl",
        help="Output ActivationPrompt JSONL path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    pairs = load_pairwise_examples_jsonl(args.input)
    prompts = activation_prompts_from_pairs(pairs)
    prompt_count = write_jsonl(prompts, args.output)
    print(f"Exported {prompt_count} activation prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
