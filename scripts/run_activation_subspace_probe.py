"""Run signed-vs-squared multi-direction activation subspace probes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.subspace_probe import (
    run_activation_subspace_probe_from_files,
    save_activation_subspace_probe_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_activation_subspace_probe_from_files(
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        metadata_key=args.metadata_key,
        max_components=args.max_components,
    )
    save_activation_subspace_probe_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "activation subspace probe: "
        f"pair_loo_signed={summary['best_pair_loo_signed_vote_accuracy']:.3f} "
        f"pair_loo_energy={summary['best_pair_loo_squared_energy_accuracy']:.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--metadata-key", default="primary_fault_class")
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "activation_subspace_probe.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "activation_subspace_probe.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
