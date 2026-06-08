"""Run held-out metadata transfer over saved activation prompts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.activation_metadata_transfer import (
    run_activation_metadata_transfer_from_files,
    save_activation_metadata_transfer_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_activation_metadata_transfer_from_files(
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        metadata_key=args.metadata_key,
        coverage_metadata_keys=args.coverage_metadata_key,
        required_coverage_metadata_keys=args.required_coverage_metadata_key,
        min_coverage_groups_per_key=args.min_coverage_groups_per_key,
        min_transfer_metadata_groups=args.min_transfer_metadata_groups,
        min_transfer_test_pairs_per_fold=args.min_transfer_test_pairs_per_fold,
        min_transfer_test_accuracy=args.min_transfer_test_accuracy,
        min_transfer_min_margin=args.min_transfer_min_margin,
    )
    save_activation_metadata_transfer_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "activation metadata transfer: "
        f"folds={summary['folds']} "
        f"test_pairs={summary['test_pairs']} "
        f"mean_acc={summary['mean_test_accuracy']:.3f} "
        f"mean_margin={summary['mean_test_margin']:+.3f}"
    )
    readiness = report["readiness"]
    print(
        "metadata coverage readiness: "
        f"status={readiness['status']} "
        f"ready={readiness['ready']}"
    )
    transfer_readiness = report["transfer_readiness"]
    print(
        "transfer readiness: "
        f"status={transfer_readiness['status']} "
        f"ready={transfer_readiness['ready']}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--metadata-key", default="primary_fault_class")
    parser.add_argument(
        "--coverage-metadata-key",
        action="append",
        default=None,
        help=(
            "Additional pair metadata key to summarize for audit coverage. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--required-coverage-metadata-key",
        action="append",
        default=None,
        help=(
            "Metadata key that must be complete for metadata coverage readiness. "
            "Defaults to all summarized coverage keys. May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--min-coverage-groups-per-key",
        type=int,
        default=1,
        help="Minimum observed value groups required for each required coverage key.",
    )
    parser.add_argument(
        "--min-transfer-metadata-groups",
        type=int,
        default=2,
        help="Minimum held-out metadata groups required for transfer claims.",
    )
    parser.add_argument(
        "--min-transfer-test-pairs-per-fold",
        type=int,
        default=1,
        help="Minimum evaluated pair count required in every held-out fold.",
    )
    parser.add_argument(
        "--min-transfer-test-accuracy",
        type=float,
        default=1.0,
        help="Minimum held-out test accuracy required in every fold.",
    )
    parser.add_argument(
        "--min-transfer-min-margin",
        type=float,
        default=0.0,
        help="Minimum positive-minus-negative projection margin required in every fold.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "activation_metadata_transfer.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "activation_metadata_transfer.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
