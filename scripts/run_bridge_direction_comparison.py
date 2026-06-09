"""Run constructed bridge direction comparison audits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.heldout_domain_direction_audit import (
    run_bridge_direction_comparison_from_files,
    save_bridge_direction_comparison_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_bridge_direction_comparison_from_files(
        source_activation_npz=args.source_activation_npz,
        source_pairs_path=args.source_pairs,
        target_activation_npz=args.target_activation_npz,
        target_pairs_path=args.target_pairs,
        source_name=args.source_name,
        target_name=args.target_name,
        source_group_key=args.source_group_key,
        target_group_key=args.target_group_key,
        bridge_stratum_key=args.bridge_stratum_key,
        composition_keys=_parse_comma_separated(args.composition_keys),
        bridge_pair_count=args.bridge_pair_count,
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_margin=args.min_margin,
        min_direction_cosine=args.min_direction_cosine,
    )
    save_bridge_direction_comparison_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "bridge direction comparison: "
        f"ready={summary['ready_for_constructed_bridge_direction_claims']} "
        f"min_cos={summary['constructed_min_joint_cosine']:+.3f} "
        f"source_min={summary['constructed_source_min_margin']:+.3f} "
        f"target_min={summary['constructed_target_min_margin']:+.3f}"
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
    parser.add_argument("--composition-keys", default="source,primary_fault_class")
    parser.add_argument("--bridge-pair-count", type=int, default=6)
    parser.add_argument("--min-pairwise-accuracy", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-direction-cosine", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "bridge_direction_comparison.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "bridge_direction_comparison.md",
    )
    return parser.parse_args(argv)


def _parse_comma_separated(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
