"""Run signed/absolute cosine audits over metadata-group activation directions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.direction_geometry import (
    run_direction_geometry_audit_from_files,
    save_direction_geometry_audit_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_direction_geometry_audit_from_files(
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        metadata_key=args.metadata_key,
        high_abs_threshold=args.high_abs_threshold,
    )
    save_direction_geometry_audit_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "direction geometry audit: "
        f"groups={summary['groups']} "
        f"mean_signed={summary['mean_signed_off_diagonal_cosine']:+.3f} "
        f"mean_abs={summary['mean_absolute_off_diagonal_cosine']:.3f} "
        f"anti_aligned={summary['strongly_anti_aligned_pairs']}"
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
        default=paths.training / "generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--metadata-key", default="primary_fault_class")
    parser.add_argument("--high-abs-threshold", type=float, default=0.8)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "direction_geometry_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "direction_geometry_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
