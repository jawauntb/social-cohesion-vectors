"""Project steered-generation activations onto the source direction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.steered_generation_projection import (
    summarize_generation_projection,
    write_generation_projection_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = summarize_generation_projection(
        activation_npz=args.activation_npz,
        direction_npz=args.direction,
    )
    write_generation_projection_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    best = report["rows"][0] if report["rows"] else {}
    print(
        "steered generation projection: "
        f"rows={len(report['rows'])} "
        f"best={best.get('report', '')} "
        f"projection_delta={float(best.get('projection_positive_minus_negative_mean_delta', 0.0)):+.4f} "
        f"score_delta={float(best.get('score_positive_minus_negative_mean_delta', 0.0)):+.4f}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    default_direction = (
        config.paths.vectors
        / "open_llm"
        / "boundary_prior_cue_balanced_expanded__Qwen__Qwen2.5-0.5B-Instruct__layer-1_direction.npz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument("--direction", type=Path, default=default_direction)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "steered_generation_projection.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "steered_generation_projection.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
