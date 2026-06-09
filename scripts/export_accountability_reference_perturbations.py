"""Export perturbations of an external generated accountability reference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.accountability_reference_perturbation import (
    export_accountability_reference_perturbations,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts, report = export_accountability_reference_perturbations(
        generated_reference_pairs=args.generated_reference_pairs,
        generated_reference_pair_id=args.generated_reference_pair_id,
        generated_reference_base_contrast_id=args.generated_reference_base_contrast_id,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
    )
    summary = report["summary"]
    print(
        "accountability reference perturbations: "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']} "
        f"perturbations={summary['perturbations']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-reference-pairs", type=Path, required=True)
    parser.add_argument("--generated-reference-pair-id")
    parser.add_argument(
        "--generated-reference-base-contrast-id",
        default="accountability_after_harm",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.processed / "accountability_reference_perturbation_pairs.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training
        / "accountability_reference_perturbation_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "accountability_reference_perturbation.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "accountability_reference_perturbation.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
