"""Fault-class held-out transfer reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import (
    load_pairwise_examples_jsonl,
    load_scored_runs_jsonl,
)
from social_cohesion_vectors.experiments.transfer import (
    evaluate_metadata_transfer,
    index_runs,
    summarize_fold_results,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun


def run_fault_heldout_transfer_from_files(
    *,
    scored_runs_path: str | Path,
    pairs_path: str | Path,
    metadata_key: str = "primary_fault_class",
) -> dict[str, Any]:
    """Load generated fault artifacts and run held-out fault-class transfer."""

    return run_fault_heldout_transfer(
        scored_runs=load_scored_runs_jsonl(scored_runs_path),
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        input_paths={
            "scored_runs": str(scored_runs_path),
            "pairs": str(pairs_path),
        },
    )


def run_fault_heldout_transfer(
    *,
    scored_runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    input_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Train on all but one fault class and evaluate on the held-out class."""

    run_index = index_runs(scored_runs)
    folds = evaluate_metadata_transfer(
        pairs=pairs,
        run_index=run_index,
        metadata_key=metadata_key,
        split_name="fault_class",
    )
    return {
        "experiment": "fault_heldout_transfer",
        "description": (
            "Text/rubric baselines trained on generated fault-class pairs with "
            "one symbolic fault class held out at a time."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "metadata_key": metadata_key,
            "scored_runs": len(scored_runs),
            "pairs": len(pairs),
            "indexed_runs": len(run_index),
            "fault_classes": len(_metadata_values(pairs, metadata_key)),
        },
        "summary": summarize_fold_results(folds),
        "folds": folds,
    }


def save_fault_heldout_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown fault-held-out reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_fault_heldout_markdown(report), encoding="utf-8")


def render_fault_heldout_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise markdown report for the held-out fault-class benchmark."""

    inputs = _mapping(report.get("inputs"))
    lines = [
        "# Fault-Held-Out Transfer",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Scored runs: {int(inputs.get('scored_runs', 0))}",
        f"- Pairwise examples: {int(inputs.get('pairs', 0))}",
        f"- Fault classes: {int(inputs.get('fault_classes', 0))}",
        f"- Metadata key: `{inputs.get('metadata_key', '')}`",
        "",
        "## Summary",
        "",
        "| Split | Baseline | Folds | Test pairs | Mean test accuracy | Mean test margin | Mean train accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("summary")):
        row_map = _mapping(row)
        fold_rows = [
            fold
            for fold in _sequence(report.get("folds"))
            if _mapping(fold).get("baseline") == row_map.get("baseline")
        ]
        mean_margin = _weighted_mean_margin(fold_rows)
        lines.append(
            "| "
            f"{row_map.get('split', '')} | "
            f"{row_map.get('baseline', '')} | "
            f"{int(row_map.get('folds', 0))} | "
            f"{int(row_map.get('test_pairs', 0))} | "
            f"{float(row_map.get('mean_test_accuracy', 0.0)):.3f} | "
            f"{mean_margin:.3f} | "
            f"{float(row_map.get('mean_train_accuracy', 0.0)):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Held-Out Fault Folds",
            "",
            "| Held-out fault | Baseline | Train pairs | Train acc | Test pairs | Test acc | Test margin |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in _sequence(report.get("folds")):
        fold_map = _mapping(fold)
        train = _mapping(fold_map.get("train"))
        test = _mapping(fold_map.get("test"))
        lines.append(
            "| "
            f"{fold_map.get('held_out', '')} | "
            f"{fold_map.get('baseline', '')} | "
            f"{int(train.get('n_pairs', 0))} | "
            f"{float(train.get('accuracy', 0.0)):.3f} | "
            f"{int(test.get('n_pairs', 0))} | "
            f"{float(test.get('accuracy', 0.0)):.3f} | "
            f"{float(test.get('mean_margin', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _metadata_values(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> set[str]:
    values: set[str] = set()
    for pair in pairs:
        raw = pair.metadata.get(metadata_key)
        if raw is None:
            continue
        values.update(part.strip() for part in str(raw).split(",") if part.strip())
    return values


def _weighted_mean_margin(rows: Sequence[object]) -> float:
    weighted_sum = 0.0
    total_pairs = 0
    for row in rows:
        test = _mapping(_mapping(row).get("test"))
        n_pairs = int(test.get("n_pairs", 0))
        weighted_sum += float(test.get("mean_margin", 0.0)) * n_pairs
        total_pairs += n_pairs
    return weighted_sum / total_pairs if total_pairs else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
