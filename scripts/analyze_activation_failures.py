"""Analyze leave-one-pair-out activation-vector failure cases."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.activation_failures import (
    analyze_failures_from_files,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = analyze_failures_from_files(
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )
    print(
        "activation failures: "
        f"pairs={report['n_pairs']} "
        f"failures={report['n_failures']} "
        f"accuracy={float(report['accuracy']):.3f}"
    )
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=paths.training / "generated_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "activation_failure_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "activation_failure_analysis.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
