"""Run residual/subspace checks after removing a global activation direction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.residual_subspace_audit import (
    run_residual_subspace_audit_from_files,
    save_residual_subspace_audit_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_residual_subspace_audit_from_files(
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        metadata_key=args.metadata_key,
    )
    save_residual_subspace_audit_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "residual subspace audit: "
        f"global_acc={summary['global_accuracy']:.3f} "
        f"residual_global_acc={summary['residual_global_accuracy']:.3f} "
        f"residual_group_mean_acc={summary['residual_group_mean_accuracy']:.3f} "
        f"signal_groups={summary['residual_groups_with_positive_signal']}"
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
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "residual_subspace_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "residual_subspace_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
