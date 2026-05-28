"""Run multi-axis guardrail monitoring over trait-axis activations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.guardrail_monitoring import (
    run_guardrail_monitoring_from_npz,
    save_guardrail_monitoring_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    activation_npz = args.activation_npz_arg or args.activation_npz
    report = run_guardrail_monitoring_from_npz(activation_npz)
    save_guardrail_monitoring_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    print(
        "guardrail monitoring: "
        f"axes={summary['axes']} "
        f"pairs={summary['pairs']} "
        f"alerts={summary['alerts']} "
        f"mean_margin={summary['mean_margin']:.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "activation_npz_arg",
        type=Path,
        nargs="?",
        help="Optional activation NPZ path. Overrides --activation-npz.",
    )
    parser.add_argument(
        "--activation-npz",
        type=Path,
        default=(
            paths.features
            / "open_llm"
            / "trait_axis_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "guardrail_monitoring.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "guardrail_monitoring.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
