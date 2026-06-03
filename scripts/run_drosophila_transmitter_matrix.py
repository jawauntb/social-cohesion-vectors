"""Export a Drosophila-inspired toy transmitter-by-coefficient matrix."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.drosophila_substrate import (
    TransmitterLabel,
    default_coefficients,
    export_drosophila_transmitter_matrix_artifacts,
    toy_transmitter_labels,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = export_drosophila_transmitter_matrix_artifacts(
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
        jsonl_output=args.jsonl_output,
        transmitters=args.transmitters,
        coefficients=args.coefficients,
        source_class=args.source_class,
    )
    print(
        "exported drosophila toy transmitter matrix: "
        f"results={counts['results']} "
        f"transmitters={len(args.transmitters)} "
        f"coefficients={len(args.coefficients)} "
        f"json_report={args.json_report_output}"
    )
    if args.markdown_report_output is not None:
        print(f"wrote {args.markdown_report_output}")
    if args.jsonl_output is not None:
        print(f"wrote {args.jsonl_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    labels = toy_transmitter_labels()
    parser = argparse.ArgumentParser(
        description=(
            "Export a synthetic Drosophila-inspired transmitter-by-coefficient "
            "edge-scaling matrix."
        )
    )
    parser.add_argument(
        "--transmitter",
        dest="transmitters",
        choices=labels,
        action="append",
        type=str,
        help=(
            "Toy transmitter label to include. Repeat to override the default "
            "all-label matrix."
        ),
    )
    parser.add_argument(
        "--coefficient",
        "--coefficients",
        dest="coefficients",
        action="append",
        type=float,
        help=(
            "Coefficient to include in the matrix. Repeat to override the "
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
        default=paths.reports / "drosophila_transmitter_matrix.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        "--markdown-summary",
        type=Path,
        default=paths.reports / "drosophila_transmitter_matrix.md",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=paths.processed / "drosophila_transmitter_matrix.jsonl",
        help="Optional JSONL path for one record per matrix row.",
    )
    args = parser.parse_args(argv)
    args.transmitters = tuple(
        cast(TransmitterLabel, transmitter)
        for transmitter in (args.transmitters or labels)
    )
    args.coefficients = tuple(args.coefficients or default_coefficients())
    return args


if __name__ == "__main__":
    raise SystemExit(main())
