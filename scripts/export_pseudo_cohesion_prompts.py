"""Export pseudo-cohesion hard negatives as activation prompt pairs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    export_pseudo_cohesion_activation_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = export_pseudo_cohesion_activation_inputs(
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
    )
    print(
        "exported pseudo-cohesion activation inputs: "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.pairs_output}")
    print(f"wrote {args.prompts_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "pseudo_cohesion_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "pseudo_cohesion_activation_prompts.jsonl",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
