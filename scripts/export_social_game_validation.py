"""Export local social-game validation artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.social_games import (
    export_social_game_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = export_social_game_artifacts(
        scored_runs_output=args.scored_runs_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
    )
    print(
        "exported social-game validation: "
        f"scored_runs={counts['scored_runs']} "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "social_game_validation_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "social_game_validation_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "social_game_validation_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "social_game_validation.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "social_game_validation.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
