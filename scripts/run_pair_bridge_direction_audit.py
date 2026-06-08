"""Run pair-level path-stratified bridge ablation audits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.heldout_domain_direction_audit import (
    run_pair_bridge_direction_audit_from_files,
    save_pair_bridge_direction_audit_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_pair_bridge_direction_audit_from_files(
        source_activation_npz=args.source_activation_npz,
        source_pairs_path=args.source_pairs,
        target_activation_npz=args.target_activation_npz,
        target_pairs_path=args.target_pairs,
        source_name=args.source_name,
        target_name=args.target_name,
        source_group_key=args.source_group_key,
        target_group_key=args.target_group_key,
        bridge_stratum_key=args.bridge_stratum_key,
        max_subsets_per_count=args.max_subsets_per_count,
        max_bridge_pair_count=args.max_bridge_pair_count,
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_margin=args.min_margin,
    )
    save_pair_bridge_direction_audit_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "pair bridge direction audit: "
        f"ready={summary['ready_for_pair_bridge_claims']} "
        f"source_min_pairs={summary['source_min_ready_bridge_pairs']} "
        f"target_min_pairs={summary['target_min_ready_bridge_pairs']} "
        f"source_failed_subsets={summary['source_failed_subset_count']} "
        f"target_failed_subsets={summary['target_failed_subset_count']}"
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
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--source-group-key", default="source")
    parser.add_argument("--target-group-key", default="source")
    parser.add_argument("--bridge-stratum-key", default="slack_options_tested")
    parser.add_argument("--max-subsets-per-count", type=int, default=32)
    parser.add_argument("--max-bridge-pair-count", type=int)
    parser.add_argument("--min-pairwise-accuracy", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "pair_bridge_direction_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "pair_bridge_direction_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
