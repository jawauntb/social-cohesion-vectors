"""Run a residual diagnostic for fresh generated bridge failures."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fresh_generated_residual_diagnostic import (
    run_fresh_generated_residual_diagnostic_from_files,
    save_fresh_generated_residual_diagnostic,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_fresh_generated_residual_diagnostic_from_files(
        bridge_report_path=args.bridge_report,
        source_pairs_path=args.source_pairs,
        fresh_source_pairs_path=args.fresh_source_pairs,
        reference_bridge_report_path=args.reference_bridge_report,
        model_name=args.model_name,
        reference_model_name=args.reference_model_name,
        evaluation_key=args.evaluation_key,
    )
    save_fresh_generated_residual_diagnostic(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "fresh generated residual diagnostic: "
        f"readiness={summary['readiness']} "
        f"failing_pairs={summary['failing_fresh_pairs']} "
        f"min_margin={summary['constructed_fresh_min_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--fresh-source-pairs", type=Path, required=True)
    parser.add_argument("--reference-bridge-report", type=Path)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--reference-model-name", default="reference")
    parser.add_argument("--evaluation-key", default="on_fresh_source")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "fresh_generated_residual_diagnostic.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "fresh_generated_residual_diagnostic.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
