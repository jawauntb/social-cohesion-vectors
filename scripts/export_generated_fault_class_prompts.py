"""Export generated fault-class pseudo-cohesion benchmark artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    export_generated_fault_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    variants = (
        tuple(
            variant
            for variant in DEFAULT_VARIANTS
            if variant.name in set(args.variants)
        )
        if args.variants
        else DEFAULT_VARIANTS
    )
    counts = export_generated_fault_dataset(
        scored_runs_output=args.scored_runs_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        prompt_records_output=args.prompt_records_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
        variants=variants,
        style=args.style,
    )
    print(
        "exported generated fault-class benchmark: "
        f"variants={','.join(variant.name for variant in variants)} "
        f"style={args.style} "
        f"scored_runs={counts['scored_runs']} "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']} "
        f"prompt_records={counts['prompt_records']}"
    )
    print(f"wrote {args.scored_runs_output}")
    print(f"wrote {args.pairs_output}")
    print(f"wrote {args.prompts_output}")
    print(f"wrote {args.prompt_records_output}")
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "generated_fault_class_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "generated_fault_class_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--prompt-records-output",
        type=Path,
        default=paths.raw / "generated_fault_class_prompt_records.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "generated_fault_class_dataset.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "generated_fault_class_dataset.md",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=variant_names,
        default=None,
        help="Optional subset of generation variants.",
    )
    parser.add_argument(
        "--style",
        choices=["template", "cue_balanced", "lexical_hardened"],
        default="template",
        type=str,
        help="Deterministic text style to export.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
