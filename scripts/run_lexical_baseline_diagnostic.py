"""Run term-level diagnostics for fixed lexical baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.lexical_baseline_diagnostic import (
    run_lexical_baseline_diagnostic_from_file,
    save_lexical_baseline_diagnostic,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_lexical_baseline_diagnostic_from_file(
        args.pairs,
        group_metadata_key=args.group_metadata_key,
    )
    save_lexical_baseline_diagnostic(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "lexical baseline diagnostic: "
        f"pairs={summary['pairs']} "
        f"label_aligned_terms={summary['label_aligned_terms']} "
        f"mean_cue_margin={summary['mean_pair_cue_margin']:+.3f} "
        f"term_polarized={summary['aggregate_balanced_term_polarized']}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_fault_source_bundle_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument("--group-metadata-key", default="primary_fault_class")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "lexical_baseline_diagnostic.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "lexical_baseline_diagnostic.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
