"""Run cross-benchmark direction-transfer audits in one activation space."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.cross_benchmark_direction_transfer import (
    run_cross_benchmark_direction_transfer_from_files,
    save_cross_benchmark_direction_transfer_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_cross_benchmark_direction_transfer_from_files(
        source_vector_npz=args.source_vector_npz,
        source_activation_npz=args.source_activation_npz,
        target_vector_npz=args.target_vector_npz,
        target_activation_npz=args.target_activation_npz,
        source_name=args.source_name,
        target_name=args.target_name,
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_margin=args.min_margin,
    )
    save_cross_benchmark_direction_transfer_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "cross-benchmark direction transfer: "
        f"ready={summary['ready_for_direction_transfer_claims']} "
        f"cos={summary['direction_cosine']:+.3f} "
        f"s2t_acc={summary['source_to_target_accuracy']:.3f} "
        f"s2t_min={summary['source_to_target_min_margin']:+.3f} "
        f"t2s_acc={summary['target_to_source_accuracy']:.3f} "
        f"t2s_min={summary['target_to_source_min_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vector-npz", type=Path, required=True)
    parser.add_argument("--source-activation-npz", type=Path, required=True)
    parser.add_argument("--target-vector-npz", type=Path, required=True)
    parser.add_argument("--target-activation-npz", type=Path, required=True)
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--min-pairwise-accuracy", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "cross_benchmark_direction_transfer.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "cross_benchmark_direction_transfer.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
