"""Run the practical future-path availability audit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.availability_audit import (
    run_availability_audit_from_file,
    save_availability_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_availability_audit_from_file(
        args.pairs,
        option_metadata_key=args.option_metadata_key,
        group_metadata_key=args.group_metadata_key,
        min_pairs_per_option=args.min_pairs_per_option,
        min_availability_margin=args.min_availability_margin,
    )
    save_availability_audit(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "availability audit: "
        f"status={summary['activation_readiness']} "
        f"ready={summary['ready_for_activation']} "
        f"pairs={summary['pairs']} "
        f"paths={summary['paths']} "
        f"options={summary['options_covered']}/{summary['required_options']}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--option-metadata-key", default="slack_options_tested")
    parser.add_argument("--group-metadata-key", default="primary_fault_class")
    parser.add_argument("--min-pairs-per-option", type=int, default=1)
    parser.add_argument("--min-availability-margin", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "availability_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "availability_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
