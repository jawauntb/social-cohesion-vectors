"""Train and evaluate a local contrastive activation direction."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import (
    load_activation_file,
    project_activations,
    save_direction,
    train_contrastive_direction,
)
from social_cohesion_vectors.config import get_config

_POSITIVE_LABELS = {"positive", "top", "cooperative", "high"}
_NEGATIVE_LABELS = {"negative", "bottom", "adversarial", "low"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Train and evaluate a contrastive activation vector."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_activation_path(),
        help="Activation NPZ produced by run_modal_activation_extraction.py.",
    )
    parser.add_argument(
        "--vector-output",
        type=Path,
        default=None,
        help="Output contrastive direction NPZ path under data/models/vectors.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=config.paths.reports / "activation_vector_experiment.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=config.paths.reports / "activation_vector_experiment.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction used by score-based contrastive training.",
    )
    return parser.parse_args(argv)


def model_slug(model_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id.strip()).strip("-")
    return slug.lower() or "model"


def default_activation_path(
    *,
    model_id: str | None = None,
    layer: int | None = None,
    features_root: str | Path | None = None,
) -> Path:
    config = get_config()
    resolved_model_id = model_id or config.model_ids.open_llm
    resolved_layer = config.model_ids.activation_layer if layer is None else layer
    root = Path(features_root) if features_root is not None else config.paths.features
    return (
        root
        / "open_llm"
        / f"activation_prompts__{model_slug(resolved_model_id)}__layer{resolved_layer}.npz"
    )


def default_vector_output_path(
    activation_path: str | Path,
    *,
    vectors_root: str | Path | None = None,
) -> Path:
    config = get_config()
    root = Path(vectors_root) if vectors_root is not None else config.paths.vectors
    return root / f"{Path(activation_path).stem}__contrastive_direction.npz"


def run_activation_vector_experiment(
    *,
    activation_path: str | Path,
    vector_output_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    report_md_path: str | Path | None = None,
    top_fraction: float = 0.25,
) -> dict[str, Any]:
    config = get_config()
    input_path = Path(activation_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Activation file does not exist: {input_path}")

    vector_path = (
        Path(vector_output_path)
        if vector_output_path is not None
        else default_vector_output_path(input_path)
    )
    json_path = (
        Path(report_json_path)
        if report_json_path is not None
        else config.paths.reports / "activation_vector_experiment.json"
    )
    md_path = (
        Path(report_md_path)
        if report_md_path is not None
        else config.paths.reports / "activation_vector_experiment.md"
    )

    batch = load_activation_file(input_path)
    direction = train_contrastive_direction(input_path, top_fraction=top_fraction)
    projections = project_activations(batch.activations, direction)
    metadata = load_activation_metadata(input_path)
    pairwise = evaluate_pairwise_accuracy(
        projections=projections,
        labels=metadata["labels"],
        pair_ids=metadata["pair_ids"],
        sample_ids=metadata["sample_ids"],
    )
    save_direction(vector_path, direction)

    summary = {
        "activation_path": str(input_path),
        "vector_path": str(vector_path),
        "report_json_path": str(json_path),
        "report_md_path": str(md_path),
        "model_id": metadata["model_id"],
        "layer": metadata["layer"],
        "pooling": metadata["pooling"],
        "activation_count": int(batch.count),
        "activation_dim": int(batch.dim),
        "direction_dim": int(direction.dim),
        "top_count": int(direction.top_count),
        "bottom_count": int(direction.bottom_count),
        "training_metadata": dict(direction.metadata),
        "pairwise_accuracy": pairwise,
        "note": (
            "Local in-sample activation diagnostic only; this is not evidence "
            "of real human behavioral or neural effects."
        ),
    }
    write_json_report(json_path, summary)
    write_markdown_report(md_path, summary)
    return summary


def load_activation_metadata(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "sample_ids": _optional_array(data, "sample_ids"),
            "pair_ids": _optional_array(data, "pair_ids"),
            "labels": _optional_array(data, "labels"),
            "model_id": _scalar(data, "model_id", default=""),
            "layer": _scalar(data, "layer", default=None),
            "pooling": _scalar(data, "pooling", default=""),
        }


def evaluate_pairwise_accuracy(
    *,
    projections: np.ndarray,
    labels: np.ndarray | None,
    pair_ids: np.ndarray | None,
    sample_ids: np.ndarray | None,
) -> dict[str, Any]:
    if labels is None:
        raise ValueError("Activation file must include labels for pairwise evaluation.")

    projection_array = np.asarray(projections, dtype=np.float64).reshape(-1)
    if labels.shape[0] != projection_array.shape[0]:
        raise ValueError("Label count does not match projection count.")

    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    skipped = 0
    for index, raw_label in enumerate(labels):
        pair_id = _pair_id_at(index, pair_ids=pair_ids, sample_ids=sample_ids)
        label = str(raw_label).strip().lower()
        if not pair_id or label not in _POSITIVE_LABELS | _NEGATIVE_LABELS:
            skipped += 1
            continue
        bucket = "positive" if label in _POSITIVE_LABELS else "negative"
        groups[pair_id][bucket].append(float(projection_array[index]))

    pairs = []
    correct = 0
    ties = 0
    for pair_id in sorted(groups):
        positives = groups[pair_id]["positive"]
        negatives = groups[pair_id]["negative"]
        if not positives or not negatives:
            skipped += 1
            continue
        positive_projection = float(np.mean(positives))
        negative_projection = float(np.mean(negatives))
        margin = positive_projection - negative_projection
        is_correct = margin > 0
        correct += int(is_correct)
        ties += int(margin == 0)
        pairs.append(
            {
                "pair_id": pair_id,
                "positive_projection": positive_projection,
                "negative_projection": negative_projection,
                "margin": float(margin),
                "correct": is_correct,
            }
        )

    if not pairs:
        raise ValueError("No positive/negative pairs were available for evaluation.")

    return {
        "accuracy": float(correct / len(pairs)),
        "correct": int(correct),
        "pair_count": int(len(pairs)),
        "ties": int(ties),
        "skipped_records_or_pairs": int(skipped),
        "pairs": pairs,
    }


def write_json_report(path: str | Path, summary: MappingSerializable) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def write_markdown_report(path: str | Path, summary: MappingSerializable) -> Path:
    pairwise = summary["pairwise_accuracy"]
    accuracy = pairwise["accuracy"]
    lines = [
        "# Activation Vector Experiment",
        "",
        f"- Activation file: `{summary['activation_path']}`",
        f"- Vector file: `{summary['vector_path']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Layer: `{summary['layer']}`",
        f"- Activation count: {summary['activation_count']}",
        f"- Activation dimension: {summary['activation_dim']}",
        f"- Split method: `{summary['training_metadata']['split_method']}`",
        f"- Pairwise accuracy: {accuracy:.3f} "
        f"({pairwise['correct']}/{pairwise['pair_count']})",
        f"- Skipped records or incomplete pairs: {pairwise['skipped_records_or_pairs']}",
        "",
        "This is a local in-sample activation diagnostic only; it is not evidence "
        "of real human behavioral or neural effects.",
        "",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _optional_array(data: Any, key: str) -> np.ndarray | None:
    return np.asarray(data[key]) if key in data else None


def _scalar(data: Any, key: str, *, default: Any) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else str(value)


def _pair_id_at(
    index: int,
    *,
    pair_ids: np.ndarray | None,
    sample_ids: np.ndarray | None,
) -> str:
    if pair_ids is not None:
        pair_id = str(pair_ids[index]).strip()
        if pair_id:
            return pair_id
    if sample_ids is None:
        return ""
    sample_id = str(sample_ids[index]).strip()
    for suffix in (":positive", ":negative", "/positive", "/negative"):
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return ""


MappingSerializable = dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_activation_vector_experiment(
        activation_path=args.input,
        vector_output_path=args.vector_output,
        report_json_path=args.report_json,
        report_md_path=args.report_md,
        top_fraction=args.top_fraction,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
