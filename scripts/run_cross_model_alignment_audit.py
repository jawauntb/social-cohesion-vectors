"""Run cross-model representation alignment and direction-transfer audits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.cross_model_alignment import (
    run_cross_model_alignment_audit_from_files,
    save_cross_model_alignment_audit_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_cross_model_alignment_audit_from_files(
        source_activation_npz=args.source_activation_npz,
        target_activation_npz=args.target_activation_npz,
        knn_k=args.knn_k,
        ridge_alpha=args.ridge_alpha,
        max_leave_one_pair_out_pairs=args.max_loo_pairs,
    )
    save_cross_model_alignment_audit_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "cross-model alignment audit: "
        f"cka={summary['linear_cka']:.3f} "
        f"mnn={summary['mutual_knn_overlap']:.3f} "
        f"s2t_acc={summary['source_to_target_mapped_accuracy']:.3f} "
        f"s2t_cos={summary['source_to_target_direction_cosine']:+.3f} "
        f"t2s_acc={summary['target_to_source_mapped_accuracy']:.3f} "
        f"t2s_cos={summary['target_to_source_direction_cosine']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-activation-npz", type=Path, required=True)
    parser.add_argument("--target-activation-npz", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--max-loo-pairs",
        type=int,
        default=40,
        help=(
            "Maximum deterministic pair-LOO folds to run for direction transfer. "
            "Use 0 for all pairs."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "cross_model_alignment_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "cross_model_alignment_audit.md",
    )
    args = parser.parse_args(argv)
    if args.max_loo_pairs == 0:
        args.max_loo_pairs = None
    return args


if __name__ == "__main__":
    raise SystemExit(main())
