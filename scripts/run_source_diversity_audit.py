"""Run source-diversity coverage gates for generated benchmark pairs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.source_diversity_audit import (
    run_source_diversity_audit_from_file,
    save_source_diversity_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_source_diversity_audit_from_file(
        args.pairs,
        source_metadata_key=args.source_metadata_key,
        group_metadata_key=args.group_metadata_key,
        min_sources=args.min_sources,
        min_pairs_per_source=args.min_pairs_per_source,
        min_groups_per_source=args.min_groups_per_source,
        min_shared_groups=args.min_shared_groups,
    )
    save_source_diversity_audit(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "source diversity audit: "
        f"status={summary['activation_readiness']} "
        f"ready={summary['ready_for_activation']} "
        f"pairs={summary['pairs']} "
        f"sources={summary['sources']} "
        f"shared_groups={summary['shared_groups']}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--source-metadata-key", default="source")
    parser.add_argument("--group-metadata-key", default="primary_fault_class")
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--min-pairs-per-source", type=int, default=2)
    parser.add_argument("--min-groups-per-source", type=int, default=2)
    parser.add_argument("--min-shared-groups", type=int, default=2)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "source_diversity_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "source_diversity_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
