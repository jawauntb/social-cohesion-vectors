"""Export the strict accountability mini-control benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.accountability_mini_control import (
    export_accountability_mini_control,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts, report = export_accountability_mini_control(
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
    )
    summary = report["summary"]
    print(
        "accountability mini-control: "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']} "
        f"sources={summary['sources']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.processed / "accountability_mini_control_pairs.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "accountability_mini_control_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "accountability_mini_control.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "accountability_mini_control.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
