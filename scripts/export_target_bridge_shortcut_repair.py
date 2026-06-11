"""Export shortcut-neutralized target/control bridge repair rows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.target_bridge_shortcut_repair import (
    export_target_bridge_shortcut_repair,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts, report = export_target_bridge_shortcut_repair(
        repair_pairs_output=args.repair_pairs_output,
        augmented_pairs_output=args.augmented_pairs_output,
        augmented_prompts_output=args.augmented_prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
    )
    summary = report["summary"]
    print(
        "target bridge shortcut repair: "
        f"repair_pairs={counts['repair_pairwise_examples']} "
        f"augmented_pairs={counts['augmented_pairwise_examples']} "
        f"prompts={counts['augmented_activation_prompts']} "
        f"sources={summary['repair_sources']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-pairs-output",
        type=Path,
        default=paths.training / "target_bridge_shortcut_repair_pairs.jsonl",
    )
    parser.add_argument(
        "--augmented-pairs-output",
        type=Path,
        default=paths.training
        / "target_bridge_shortcut_repair_augmented_pairs.jsonl",
    )
    parser.add_argument(
        "--augmented-prompts-output",
        type=Path,
        default=paths.training
        / "target_bridge_shortcut_repair_augmented_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "target_bridge_shortcut_repair.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "target_bridge_shortcut_repair.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
