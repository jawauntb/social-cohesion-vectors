"""Run generated benchmark audit reports as one provenance bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=args.scored_runs,
        pairs_path=args.pairs,
        activation_npz=args.activation_npz,
        output_dir=args.output_dir,
        group_metadata_key=args.group_metadata_key,
        source_metadata_key=args.source_metadata_key,
    )
    summary = manifest["summary"]
    print(
        "generated benchmark audit bundle: "
        f"status={summary['status']} "
        f"ready={summary['ready']} "
        f"steps={summary['steps']} "
        f"skipped={summary['skipped_steps']}"
    )
    print(f"wrote {manifest['manifest_markdown_path']}")
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
    parser.add_argument("--activation-npz", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.reports / "generated_benchmark_audit_bundle",
    )
    parser.add_argument("--group-metadata-key", default="primary_fault_class")
    parser.add_argument("--source-metadata-key", default="source")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
