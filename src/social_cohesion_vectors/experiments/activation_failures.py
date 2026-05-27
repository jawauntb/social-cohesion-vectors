"""Analyze leave-one-pair-out activation-vector failure cases."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.schemas import PairwiseExample


def analyze_failures_from_files(
    *,
    activation_npz: str | Path,
    pairs_path: str | Path,
    json_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> dict[str, Any]:
    """Load activations and pairs, analyze LOO failures, and write reports."""

    pairs = load_pairwise_examples_jsonl(pairs_path)
    with np.load(activation_npz, allow_pickle=False) as data:
        report = analyze_leave_one_pair_out_failures(
            activations=np.asarray(data["activations"], dtype=np.float64),
            pair_ids=np.asarray(data["pair_ids"], dtype=str),
            labels=np.asarray(data["labels"], dtype=str),
            pairs=pairs,
            activation_npz=str(activation_npz),
        )

    if json_output is not None:
        output = Path(json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown_output is not None:
        output = Path(markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")
    return report


def analyze_leave_one_pair_out_failures(
    *,
    activations: np.ndarray,
    pair_ids: np.ndarray,
    labels: np.ndarray,
    pairs: Sequence[PairwiseExample],
    activation_npz: str | None = None,
) -> dict[str, Any]:
    """Return per-pair LOO margins and failure cases."""

    pair_index = {pair.pair_id: pair for pair in pairs}
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(set(str(pair_id) for pair_id in pair_ids)):
        test_mask = pair_ids == pair_id
        train_mask = ~test_mask
        if int(test_mask.sum()) != 2 or int(train_mask.sum()) < 2:
            continue
        direction = train_direction_from_arrays(
            activations[train_mask],
            labels=labels[train_mask],
        )
        projections = direction.project(activations[test_mask])
        scores = {
            str(label): float(projection)
            for label, projection in zip(labels[test_mask], projections, strict=True)
        }
        if "positive" not in scores or "negative" not in scores:
            continue
        pair = pair_index.get(pair_id)
        margin = scores["positive"] - scores["negative"]
        rows.append(
            {
                "pair_id": pair_id,
                "margin": round(float(margin), 6),
                "correct": margin > 0,
                "positive_projection": round(scores["positive"], 6),
                "negative_projection": round(scores["negative"], 6),
                "positive_run_id": pair.positive_run_id if pair else None,
                "negative_run_id": pair.negative_run_id if pair else None,
                "positive_score": pair.positive_score if pair else None,
                "negative_score": pair.negative_score if pair else None,
                "negative_style": _style_from_run_id(pair.negative_run_id)
                if pair
                else None,
            }
        )

    failures = [row for row in rows if not row["correct"]]
    return {
        "activation_npz": activation_npz,
        "n_pairs": len(rows),
        "n_failures": len(failures),
        "accuracy": round((len(rows) - len(failures)) / len(rows), 6) if rows else 0.0,
        "failure_styles": _count_values(
            str(row.get("negative_style") or "unknown") for row in failures
        ),
        "worst_failures": sorted(failures, key=lambda row: float(row["margin"])),
        "all_pairs": rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render failure analysis as markdown."""

    failures = report.get("worst_failures")
    failure_rows = failures if isinstance(failures, list) else []
    lines = [
        "# Activation Failure Analysis",
        "",
        f"- Activation file: `{Path(str(report.get('activation_npz', ''))).name}`",
        f"- Pairs evaluated: {report.get('n_pairs', 0)}",
        f"- Accuracy: {float(report.get('accuracy', 0.0)):.3f}",
        f"- Failures: {report.get('n_failures', 0)}",
        "",
        "## Failure Styles",
        "",
        "| Negative style | Failures |",
        "| --- | ---: |",
    ]
    for style, count in _mapping(report.get("failure_styles")).items():
        lines.append(f"| {style} | {count} |")
    if not _mapping(report.get("failure_styles")):
        lines.append("| n/a | 0 |")

    lines.extend(
        [
            "",
            "## Worst Failures",
            "",
            "| Margin | Pair | Positive run | Negative run | Positive score | Negative score |",
            "| ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    if not failure_rows:
        lines.append("| n/a | n/a | n/a | n/a | 0.000 | 0.000 |")
    for row in failure_rows:
        item = _mapping(row)
        lines.append(
            "| "
            f"{float(item.get('margin', 0.0)):+.3f} | "
            f"`{item.get('pair_id', '')}` | "
            f"`{item.get('positive_run_id', '')}` | "
            f"`{item.get('negative_run_id', '')}` | "
            f"{float(item.get('positive_score') or 0.0):.3f} | "
            f"{float(item.get('negative_score') or 0.0):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _style_from_run_id(run_id: str) -> str:
    parts = run_id.split("__")
    return parts[1] if len(parts) > 1 else "unknown"


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
