"""Train and save an affect-residualized activation direction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.affect_residualized_direction import (
    train_affect_residualized_direction,
    write_affect_residualized_direction_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _load_activation_payload(args.activation_npz)
    pairs = load_pairwise_examples_jsonl(args.pairs)
    _, report = train_affect_residualized_direction(
        activations=payload["activations"],
        pair_ids=payload["pair_ids"].tolist(),
        labels=payload["labels"].tolist(),
        pairs=[pair.model_dump(mode="json") for pair in pairs],
        activation_npz=args.activation_npz,
        pairs_path=args.pairs,
        output_path=args.output,
    )
    write_affect_residualized_direction_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("direction report summary is not a dictionary")
    print(
        "affect-residualized direction: "
        f"rank={int(summary['affect_subspace_rank'])} "
        f"retained={float(summary['retained_norm_fraction']):.3f} "
        f"raw_margin={float(summary['raw_mean_margin']):+.3f} "
        f"residual_margin={float(summary['residualized_mean_margin']):+.3f}"
    )
    return 0


def _load_activation_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "activations": np.asarray(data["activations"], dtype=np.float64),
            "pair_ids": np.asarray(data["pair_ids"], dtype=str),
            "labels": np.asarray(data["labels"], dtype=str),
        }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/training/affect_control_pairwise_probe_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination NPZ for the residualized direction.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/reports/affect_residualized_direction.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/reports/affect_residualized_direction.md"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
