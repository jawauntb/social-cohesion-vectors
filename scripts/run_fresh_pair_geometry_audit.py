"""Run a pair-level geometry audit for a hard fresh generated residual."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fresh_pair_geometry_audit import (
    run_fresh_pair_geometry_audit_from_files,
    save_fresh_pair_geometry_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_fresh_pair_geometry_audit_from_files(
        source_activation_npz=args.source_activation_npz,
        source_pairs_path=args.source_pairs,
        target_activation_npz=args.target_activation_npz,
        target_pairs_path=args.target_pairs,
        fresh_source_activation_npz=args.fresh_source_activation_npz,
        fresh_source_pairs_path=args.fresh_source_pairs,
        focus_base_contrast_id=args.focus_base_contrast_id,
        focus_pair_id=args.focus_pair_id,
        model_name=args.model_name,
        nearest_k=args.nearest_k,
    )
    save_fresh_pair_geometry_audit(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "fresh pair geometry audit: "
        f"status={summary['readiness']} "
        f"original={summary['original_joint_focus_margin']:+.3f} "
        f"full={summary['full_augmented_focus_margin']:+.3f} "
        f"fresh_only={summary['fresh_only_focus_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-activation-npz", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--target-activation-npz", type=Path, required=True)
    parser.add_argument("--target-pairs", type=Path, required=True)
    parser.add_argument("--fresh-source-activation-npz", type=Path, required=True)
    parser.add_argument("--fresh-source-pairs", type=Path, required=True)
    parser.add_argument("--focus-base-contrast-id", required=True)
    parser.add_argument("--focus-pair-id")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--nearest-k", type=int, default=8)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "fresh_pair_geometry_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "fresh_pair_geometry_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
