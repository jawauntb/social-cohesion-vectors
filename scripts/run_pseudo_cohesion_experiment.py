"""Run the pseudo-cohesion hard-negative experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    DEFAULT_LEXICAL_HIGH_THRESHOLD,
    DEFAULT_RISK_COMPONENT_THRESHOLD,
    DEFAULT_SCORER_HIGH_THRESHOLD,
    run_experiment,
    write_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_experiment(
        scorer_high_threshold=args.scorer_high_threshold,
        lexical_high_threshold=args.lexical_high_threshold,
        risk_component_threshold=args.risk_component_threshold,
    )
    write_reports(
        report,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    summary = report["summary"]
    print(
        "Wrote pseudo-cohesion report to "
        f"{args.output_json} and {args.output_markdown}. "
        f"Scorer failures: {summary['scorer_failure_count']}; "
        f"lexical failures: {summary['lexical_failure_count']}."
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    reports_dir = get_config().paths.reports
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial pseudo-cohesion hard negatives."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=reports_dir / "pseudo_cohesion_experiment.json",
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=reports_dir / "pseudo_cohesion_experiment.md",
        help="Destination markdown report path.",
    )
    parser.add_argument(
        "--scorer-high-threshold",
        type=float,
        default=DEFAULT_SCORER_HIGH_THRESHOLD,
        help="Pseudo-cohesion scorer score at or above this is a failure case.",
    )
    parser.add_argument(
        "--lexical-high-threshold",
        type=float,
        default=DEFAULT_LEXICAL_HIGH_THRESHOLD,
        help="Pseudo-cohesion lexical score at or above this is a failure case.",
    )
    parser.add_argument(
        "--risk-component-threshold",
        type=float,
        default=DEFAULT_RISK_COMPONENT_THRESHOLD,
        help="Core risk components below this value flag pseudo-cohesion as risky.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
