"""Summarize constructed bridge preservation across evaluation slices."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.bridge_preservation_summary import (
    BridgePreservationReportInput,
    run_bridge_preservation_summary_from_files,
    save_bridge_preservation_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_bridge_preservation_summary_from_files(
        bridge_reports=[_parse_report_arg(value) for value in args.bridge_report],
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_margin=args.min_margin,
    )
    save_bridge_preservation_summary(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "bridge preservation summary: "
        f"models={summary['models']} "
        f"ready={summary['ready_for_preservation_claims']} "
        f"failures={summary['failed_pair_evaluations']} "
        f"worst={summary['worst_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-report",
        action="append",
        required=True,
        help="Named report in the form model_id=/path/to/bridge_diagnostic.json.",
    )
    parser.add_argument("--min-pairwise-accuracy", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "bridge_preservation_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "bridge_preservation_summary.md",
    )
    return parser.parse_args(argv)


def _parse_report_arg(raw_value: str) -> BridgePreservationReportInput:
    if "=" not in raw_value:
        msg = f"Expected model_id=path bridge report argument, got {raw_value!r}."
        raise argparse.ArgumentTypeError(msg)
    model_id, path = raw_value.split("=", 1)
    if not model_id or not path:
        msg = f"Expected non-empty model_id and path in {raw_value!r}."
        raise argparse.ArgumentTypeError(msg)
    return BridgePreservationReportInput(model_id=model_id, path=Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
