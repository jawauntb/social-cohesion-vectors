"""Export social-state modulator activation prompt JSONL and summaries."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.social_state_modulators import (
    activation_prompts_from_social_state_modulators,
    canonical_social_state_modulators,
    render_markdown_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    modulators = canonical_social_state_modulators()
    prompts = activation_prompts_from_social_state_modulators(modulators)
    prompt_count = write_jsonl(prompts, args.output)

    message = (
        f"Exported {prompt_count} social-state modulator activation prompts "
        f"to {args.output}"
    )
    if args.markdown_summary is not None:
        args.markdown_summary.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_summary.write_text(
            render_markdown_summary(modulators=modulators, prompts=prompts),
            encoding="utf-8",
        )
        message += f" and summary to {args.markdown_summary}"
    print(message)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description=(
            "Export hand-authored social-state modulator ActivationPrompt JSONL."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.training / "social_state_modulator_activation_prompts.jsonl",
        help="Output ActivationPrompt JSONL path.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        nargs="?",
        const=paths.reports / "social_state_modulator_prompt_summary.md",
        default=None,
        help=(
            "Optional markdown summary path. If the flag is passed without a "
            "path, writes to "
            f"{paths.reports / 'social_state_modulator_prompt_summary.md'}."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
