"""Run cross-model constructed bridge direction transport audits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.heldout_domain_direction_audit import (
    run_cross_model_bridge_transport_from_files,
    save_cross_model_bridge_transport_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_cross_model_bridge_transport_from_files(
        model_a_source_activation_npz=args.model_a_source_activation_npz,
        model_a_source_pairs_path=args.model_a_source_pairs,
        model_a_target_activation_npz=args.model_a_target_activation_npz,
        model_a_target_pairs_path=args.model_a_target_pairs,
        model_b_source_activation_npz=args.model_b_source_activation_npz,
        model_b_source_pairs_path=args.model_b_source_pairs,
        model_b_target_activation_npz=args.model_b_target_activation_npz,
        model_b_target_pairs_path=args.model_b_target_pairs,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        source_name=args.source_name,
        target_name=args.target_name,
        source_group_key=args.source_group_key,
        target_group_key=args.target_group_key,
        bridge_stratum_key=args.bridge_stratum_key,
        composition_keys=tuple(args.composition_keys.split(",")),
        bridge_pair_count=args.bridge_pair_count,
        knn_k=args.knn_k,
        ridge_alpha=args.ridge_alpha,
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_margin=args.min_margin,
        min_mapped_direction_cosine=args.min_mapped_direction_cosine,
    )
    save_cross_model_bridge_transport_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "cross-model bridge transport: "
        f"ready={summary['ready_for_cross_model_bridge_transport_claims']} "
        f"cka={summary['linear_cka']:.3f} "
        f"a2b_cos={summary['model_a_to_b_min_bridge_cosine']:+.3f} "
        f"a2b_source_min={summary['model_a_to_b_source_min_margin']:+.3f} "
        f"a2b_target_min={summary['model_a_to_b_target_min_margin']:+.3f} "
        f"a2b_leave_min={summary['model_a_to_b_leave_heldout_min_margin']:+.3f} "
        f"b2a_cos={summary['model_b_to_a_min_bridge_cosine']:+.3f} "
        f"b2a_source_min={summary['model_b_to_a_source_min_margin']:+.3f} "
        f"b2a_target_min={summary['model_b_to_a_target_min_margin']:+.3f} "
        f"b2a_leave_min={summary['model_b_to_a_leave_heldout_min_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-source-activation-npz", type=Path, required=True)
    parser.add_argument("--model-a-source-pairs", type=Path, required=True)
    parser.add_argument("--model-a-target-activation-npz", type=Path, required=True)
    parser.add_argument("--model-a-target-pairs", type=Path, required=True)
    parser.add_argument("--model-b-source-activation-npz", type=Path, required=True)
    parser.add_argument("--model-b-source-pairs", type=Path, required=True)
    parser.add_argument("--model-b-target-activation-npz", type=Path, required=True)
    parser.add_argument("--model-b-target-pairs", type=Path, required=True)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--source-group-key", default="source")
    parser.add_argument("--target-group-key", default="source")
    parser.add_argument("--bridge-stratum-key", default="slack_options_tested")
    parser.add_argument("--composition-keys", default="source,primary_fault_class")
    parser.add_argument("--bridge-pair-count", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-pairwise-accuracy", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-mapped-direction-cosine", type=float, default=0.0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "cross_model_bridge_transport.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "cross_model_bridge_transport.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
