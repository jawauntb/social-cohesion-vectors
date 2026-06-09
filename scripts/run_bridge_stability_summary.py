"""Summarize constructed bridge failures across perturbation diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.bridge_stability_summary import (
    BridgeReportInput,
    run_bridge_stability_summary_from_files,
    save_bridge_stability_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_bridge_stability_summary_from_files(
        bridge_reports=[_parse_report_arg(value) for value in args.bridge_report],
        fresh_source_pairs_path=args.fresh_source_pairs,
    )
    save_bridge_stability_summary(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "bridge stability summary: "
        f"models={summary['models']} "
        f"failures={summary['constructed_failure_rows']} "
        f"worst={summary['worst_margin']:+.3f} "
        f"top={summary['most_failed_perturbation_id']}"
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
    parser.add_argument("--fresh-source-pairs", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "bridge_stability_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "bridge_stability_summary.md",
    )
    return parser.parse_args(argv)


def _parse_report_arg(raw_value: str) -> BridgeReportInput:
    if "=" not in raw_value:
        msg = f"Expected model_id=path bridge report argument, got {raw_value!r}."
        raise argparse.ArgumentTypeError(msg)
    model_id, path = raw_value.split("=", 1)
    if not model_id or not path:
        msg = f"Expected non-empty model_id and path in {raw_value!r}."
        raise argparse.ArgumentTypeError(msg)
    return BridgeReportInput(model_id=model_id, path=Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
