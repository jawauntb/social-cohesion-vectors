"""Run fault-class held-out transfer over generated pseudo-cohesion pairs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fault_heldout import (
    run_fault_heldout_transfer_from_files,
    save_fault_heldout_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_fault_heldout_transfer_from_files(
        scored_runs_path=args.scored_runs,
        pairs_path=args.pairs,
        metadata_key=args.metadata_key,
        min_metadata_groups=args.min_metadata_groups,
        min_test_pairs_per_fold=args.min_test_pairs_per_fold,
        source_metadata_key=args.source_metadata_key,
        min_source_groups=args.min_source_groups,
        min_source_test_pairs_per_fold=args.min_source_test_pairs_per_fold,
    )
    save_fault_heldout_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    inputs = report["inputs"]
    print(
        "fault-held-out transfer: "
        f"scored_runs={inputs['scored_runs']} "
        f"pairs={inputs['pairs']} "
        f"fault_classes={inputs['fault_classes']} "
        f"source_groups={inputs['source_groups']} "
        f"fold_rows={len(report['folds'])} "
        f"readiness={report['readiness']['status']} "
        f"source_readiness={report['source_transfer']['readiness']['status']}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs",
        type=Path,
        default=paths.processed / "generated_fault_class_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--metadata-key",
        default="primary_fault_class",
        help="Pair metadata key to hold out.",
    )
    parser.add_argument("--min-metadata-groups", type=int, default=2)
    parser.add_argument("--min-test-pairs-per-fold", type=int, default=1)
    parser.add_argument(
        "--source-metadata-key",
        default="source",
        help="Pair metadata key to hold out for source-transfer checks.",
    )
    parser.add_argument("--min-source-groups", type=int, default=2)
    parser.add_argument("--min-source-test-pairs-per-fold", type=int, default=1)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "generated_fault_class_heldout_transfer.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "generated_fault_class_heldout_transfer.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
