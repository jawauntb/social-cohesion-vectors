"""Export an expanded pseudo-cohesion hard-negative activation batch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    export_pseudo_cohesion_activation_inputs,
)
from social_cohesion_vectors.experiments.pseudo_cohesion_expansion import (
    expanded_examples,
    select_variants,
    variant_names,
    variant_set_names,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    variants = select_variants(args.variants, variant_set=args.variant_set)
    examples = expanded_examples(variants=variants, include_seed=not args.no_seed)
    counts = export_pseudo_cohesion_activation_inputs(
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        examples=examples,
    )
    print(
        "exported expanded pseudo-cohesion activation inputs: "
        f"variant_set={args.variant_set} "
        f"variants={','.join(variant.name for variant in variants)} "
        f"include_seed={not args.no_seed} "
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
        default=paths.training / "pseudo_cohesion_expanded_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "pseudo_cohesion_expanded_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=variant_names(),
        help="Specific expansion variants to include. Overrides --variant-set.",
    )
    parser.add_argument(
        "--variant-set",
        choices=variant_set_names(),
        default="wrapped",
        help="Default variant family to export when --variants is omitted.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Export only generated variants instead of seed plus variants.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
