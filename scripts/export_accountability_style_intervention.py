"""Export source-style intervention pairs for the accountability residual."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.accountability_style_intervention import (
    export_accountability_style_intervention,
    load_generated_accountability_reference,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generated_reference = (
        load_generated_accountability_reference(
            args.generated_reference_pairs,
            pair_id=args.generated_reference_pair_id,
            base_contrast_id=args.generated_reference_base_contrast_id,
        )
        if args.generated_reference_pairs is not None
        else None
    )
    counts, report = export_accountability_style_intervention(
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
        generated_reference=generated_reference,
    )
    summary = report["summary"]
    print(
        "accountability style intervention: "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']} "
        f"generated_refs={summary['generated_reference_pairs']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-reference-pairs",
        type=Path,
        help=(
            "Optional JSONL pair file containing the generated residual reference. "
            "The generated text stays external to git."
        ),
    )
    parser.add_argument("--generated-reference-pair-id")
    parser.add_argument(
        "--generated-reference-base-contrast-id",
        default="accountability_after_harm",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.processed / "accountability_style_intervention_pairs.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training
        / "accountability_style_intervention_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "accountability_style_intervention.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "accountability_style_intervention.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
