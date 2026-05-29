"""Build activation prompts from steered generation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.steered_generation_projection import (
    projection_prompt_records_from_report_paths,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = projection_prompt_records_from_report_paths(
        args.reports,
        text_mode=args.text_mode,
    )
    write_jsonl(records, args.output)
    print(f"wrote {args.output} records={len(records)} text_mode={args.text_mode}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--text-mode",
        choices=("response", "prompt_response"),
        default="response",
        help="Embed generated response only, or prompt plus generated response.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.training / "steered_generation_projection_prompts.jsonl",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
