"""Export Drosophila-inspired toy substrate sweep artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.drosophila_substrate import (
    TransmitterLabel,
    default_coefficients,
    export_drosophila_toy_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = export_drosophila_toy_artifacts(
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
        jsonl_output=args.jsonl_output,
        transmitter=args.transmitter,
        coefficients=args.coefficients,
        source_class=args.source_class,
    )
    print(
        "exported drosophila toy substrate sweep: "
        f"results={counts['results']} "
        f"transmitter={args.transmitter} "
        f"json_report={args.json_report_output}"
    )
    if args.markdown_report_output is not None:
        print(f"wrote {args.markdown_report_output}")
    if args.jsonl_output is not None:
        print(f"wrote {args.jsonl_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description=(
            "Export a synthetic Drosophila-inspired graph edge-scaling fixture."
        )
    )
    parser.add_argument(
        "--transmitter",
        choices=(
            "acetylcholine",
            "gaba",
            "glutamate",
            "dopamine",
            "serotonin",
            "octopamine",
        ),
        default="acetylcholine",
        type=str,
        help="Toy transmitter label whose matching edges are scaled.",
    )
    parser.add_argument(
        "--coefficient",
        "--coefficients",
        dest="coefficients",
        action="append",
        type=float,
        help=(
            "Coefficient to include in the sweep. Repeat to override the "
            "default grid."
        ),
    )
    parser.add_argument(
        "--source-class",
        type=str,
        default=None,
        help="Optional source node class gate for local edge scaling.",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "drosophila_toy_substrate_sweep.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        "--markdown-summary",
        type=Path,
        default=paths.reports / "drosophila_toy_substrate_sweep.md",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=paths.processed / "drosophila_toy_substrate_sweep.jsonl",
        help="Optional JSONL path for one record per sweep coefficient.",
    )
    args = parser.parse_args(argv)
    args.transmitter = cast(TransmitterLabel, args.transmitter)
    args.coefficients = tuple(args.coefficients or default_coefficients())
    return args


if __name__ == "__main__":
    raise SystemExit(main())
