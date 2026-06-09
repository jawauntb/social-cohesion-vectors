"""Recover PairwiseExample JSONL manifests from activation NPZ payloads."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.experiments.activation_pair_manifest_recovery import (
    DEFAULT_SOURCE_FAMILY_MAP,
    write_recovered_pair_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = write_recovered_pair_manifest(
        args.activation_npz,
        dataset_kind=args.dataset_kind,
        pairs_output=args.pairs_output,
        json_report_output=args.json_report_output,
        source_family_map=_source_family_map(args.source_family_map),
    )
    summary = result.report["summary"]
    print(
        "activation pair manifest recovery: "
        f"pairs={summary['pairwise_examples']} "
        f"sources={summary['sources']} "
        f"wrote={args.pairs_output}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-npz", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=("generated_fault", "procedural_control"),
        required=True,
    )
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--json-report-output", type=Path)
    parser.add_argument(
        "--source-family-map",
        action="append",
        default=[],
        metavar="RAW=SOURCE",
        help=(
            "Override a generated source-family mapping. May be repeated. "
            f"Defaults include: {', '.join(sorted(DEFAULT_SOURCE_FAMILY_MAP))}."
        ),
    )
    return parser.parse_args(argv)


def _source_family_map(raw_values: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            msg = f"Expected --source-family-map RAW=SOURCE, got {raw_value!r}."
            raise ValueError(msg)
        raw_key, source = raw_value.split("=", 1)
        mapping[raw_key.strip()] = source.strip()
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
