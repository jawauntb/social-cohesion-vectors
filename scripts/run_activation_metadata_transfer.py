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
