"""Export social-state modulator activation prompt JSONL and summaries."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.social_state_modulators import (
    SocialStateModulatorVariantSet,
    export_social_state_modulator_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = export_social_state_modulator_artifacts(
        scored_runs_output=args.scored_runs_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        json_report_output=args.json_report_output,
        markdown_report_output=args.markdown_report_output,
        variant_set=args.variant_set,
    )
    print(
        "exported social-state modulator benchmark: "
        f"scored_runs={counts['scored_runs']} "
        f"pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']} "
        f"variant_set={args.variant_set}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description=(
            "Export hand-authored social-state modulator ActivationPrompt JSONL."
        )
    )
    parser.add_argument(
        "--variant-set",
        choices=("seed", "cue_balanced", "expanded"),
        default="seed",
        type=str,
        help=(
            "Prompt variant set to export. 'seed' preserves the original small "
            "benchmark; 'cue_balanced' exports only matched cue-balanced "
            "contrasts; 'expanded' combines both."
        ),
    )
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "social_state_modulator_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "social_state_modulator_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        "--output",
        type=Path,
        default=paths.training / "social_state_modulator_activation_prompts.jsonl",
        help="Output ActivationPrompt JSONL path.",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "social_state_modulator_benchmark.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        "--markdown-summary",
        type=Path,
        default=paths.reports / "social_state_modulator_benchmark.md",
        help="Output benchmark markdown report path.",
    )
    args = parser.parse_args(argv)
    args.variant_set = cast(SocialStateModulatorVariantSet, args.variant_set)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
