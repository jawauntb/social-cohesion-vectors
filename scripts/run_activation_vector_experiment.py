"""Train/evaluate a contrastive vector from saved open-LLM activations."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from social_cohesion_vectors.activations.contrastive import (
    save_direction,
    train_direction_from_arrays,
)
from social_cohesion_vectors.config import get_config


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_npz", type=Path)
    parser.add_argument(
        "--vector-output",
        type=Path,
        default=config.paths.vectors / "open_llm_cohesion_direction.npz",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "activation_vector_experiment.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "activation_vector_experiment.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_activation_payload(args.activation_npz)
    direction = train_direction_from_arrays(
        payload["activations"],
        labels=payload["labels"],
    )
    save_direction(args.vector_output, direction)
    projections = direction.project(payload["activations"])
    in_sample = pairwise_projection_metrics(
        pair_ids=payload["pair_ids"],
        labels=payload["labels"],
        projections=projections,
    )
    leave_one_pair_out = leave_one_pair_out_metrics(
        activations=payload["activations"],
        pair_ids=payload["pair_ids"],
        labels=payload["labels"],
    )
    report = {
        "activation_npz": str(args.activation_npz),
        "vector_output": str(args.vector_output),
        "n_prompts": int(payload["activations"].shape[0]),
        "activation_dim": int(payload["activations"].shape[1]),
        "direction_top_count": direction.top_count,
        "direction_bottom_count": direction.bottom_count,
        "in_sample": in_sample,
        "leave_one_pair_out": leave_one_pair_out,
    }
    write_reports(
        report, json_path=args.json_output, markdown_path=args.markdown_output
    )
    print(
        "activation vector pairwise accuracy="
        f"in_sample={in_sample['pairwise_accuracy']:.3f} "
        f"loo={leave_one_pair_out['pairwise_accuracy']:.3f} "
        f"pairs={leave_one_pair_out['n_pairs']} dim={report['activation_dim']}"
    )
    return 0


def load_activation_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "activations": np.asarray(data["activations"], dtype=np.float64),
            "pair_ids": np.asarray(data["pair_ids"], dtype=str),
            "labels": np.asarray(data["labels"], dtype=str),
        }


def pairwise_projection_metrics(
    *,
    pair_ids: np.ndarray,
    labels: np.ndarray,
    projections: np.ndarray,
) -> dict[str, float | int]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for pair_id, label, projection in zip(pair_ids, labels, projections, strict=True):
        grouped[str(pair_id)][str(label)] = float(projection)

    outcomes: list[float] = []
    margins: list[float] = []
    for scores in grouped.values():
        if "positive" not in scores or "negative" not in scores:
            continue
        margin = scores["positive"] - scores["negative"]
        margins.append(margin)
        outcomes.append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)

    accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
    return {
        "n_pairs": len(outcomes),
        "pairwise_accuracy": round(accuracy, 6),
        "mean_projection_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_projection_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_projection_margin": round(float(np.max(margins)), 6) if margins else 0.0,
    }


def leave_one_pair_out_metrics(
    *,
    activations: np.ndarray,
    pair_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    outcomes: list[float] = []
    margins: list[float] = []
    for pair_id in sorted(set(str(pair_id) for pair_id in pair_ids)):
        test_mask = pair_ids == pair_id
        train_mask = ~test_mask
        if int(test_mask.sum()) != 2 or int(train_mask.sum()) < 2:
            continue
        direction = train_direction_from_arrays(
            activations[train_mask],
            labels=labels[train_mask],
        )
        test_projection = direction.project(activations[test_mask])
        scores = {
            str(label): float(projection)
            for label, projection in zip(labels[test_mask], test_projection, strict=True)
        }
        if "positive" not in scores or "negative" not in scores:
            continue
        margin = scores["positive"] - scores["negative"]
        margins.append(margin)
        outcomes.append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)

    accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
    return {
        "n_pairs": len(outcomes),
        "pairwise_accuracy": round(accuracy, 6),
        "mean_projection_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_projection_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_projection_margin": round(float(np.max(margins)), 6) if margins else 0.0,
    }


def write_reports(
    report: dict[str, object], *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, object]) -> str:
    activation_name = re.sub(r".*/", "", str(report["activation_npz"]))
    in_sample = report["in_sample"]
    leave_one_pair_out = report["leave_one_pair_out"]
    if not isinstance(in_sample, dict) or not isinstance(leave_one_pair_out, dict):
        raise TypeError("report is missing nested metric dictionaries")
    return "\n".join(
        [
            "# Activation Vector Experiment",
            "",
            f"- Activation file: `{activation_name}`",
            f"- Prompts: {report['n_prompts']}",
            f"- Activation dim: {report['activation_dim']}",
            f"- In-sample pairs evaluated: {in_sample['n_pairs']}",
            f"- In-sample pairwise accuracy: {float(in_sample['pairwise_accuracy']):.3f}",
            f"- In-sample mean positive-minus-negative projection margin: {float(in_sample['mean_projection_margin']):+.4f}",
            f"- Leave-one-pair-out pairs evaluated: {leave_one_pair_out['n_pairs']}",
            f"- Leave-one-pair-out pairwise accuracy: {float(leave_one_pair_out['pairwise_accuracy']):.3f}",
            f"- Leave-one-pair-out mean projection margin: {float(leave_one_pair_out['mean_projection_margin']):+.4f}",
            "",
            "This is still a scripted-data sanity check. The next report should use held-out LLM-generated trajectories or cross-scenario folds.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
