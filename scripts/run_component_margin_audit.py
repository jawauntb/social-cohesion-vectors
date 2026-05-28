"""Run a component-margin audit over pairwise scored benchmark artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.component_audit import (
    run_component_margin_audit_from_files,
    save_component_margin_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_component_margin_audit_from_files(
        scored_runs_path=args.scored_runs,
        pairs_path=args.pairs,
        group_metadata_key=args.group_metadata_key,
    )
    save_component_margin_audit(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "component audit: "
        f"pairs={summary['pairs']} "
        f"score_accuracy={summary['score_accuracy']:.3f} "
        f"mean_score_margin={summary['mean_score_margin']:+.3f}"
    )
    print(f"wrote {args.markdown_output}")
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
    parser.add_argument("--group-metadata-key", default="primary_fault_class")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "generated_fault_class_component_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "generated_fault_class_component_audit.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
