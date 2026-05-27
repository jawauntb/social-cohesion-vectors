"""Export trait-axis activation prompt JSONL and optional markdown summary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.trait_axes import (
    activation_prompts_from_trait_axes,
    canonical_trait_axes,
    render_markdown_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    axes = canonical_trait_axes()
    prompts = activation_prompts_from_trait_axes(axes)
    prompt_count = write_jsonl(prompts, args.output)

    message = f"Exported {prompt_count} trait-axis activation prompts to {args.output}"
    if args.markdown_summary is not None:
        args.markdown_summary.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_summary.write_text(
            render_markdown_summary(axes=axes, prompts=prompts),
            encoding="utf-8",
        )
        message += f" and summary to {args.markdown_summary}"
    print(message)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description="Export hand-authored trait-axis ActivationPrompt JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.training / "trait_axis_activation_prompts.jsonl",
        help="Output ActivationPrompt JSONL path.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        nargs="?",
        const=paths.reports / "trait_axis_prompt_summary.md",
        default=None,
        help=(
            "Optional markdown summary path. If the flag is passed without a "
            "path, writes to "
            f"{paths.reports / 'trait_axis_prompt_summary.md'}."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
