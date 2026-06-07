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
    min_metadata_groups: int = 2,
    min_test_pairs_per_fold: int = 1,
    source_metadata_key: str = "source",
    min_source_groups: int = 2,
    min_source_test_pairs_per_fold: int = 1,
) -> dict[str, Any]:
    """Load generated fault artifacts and run held-out fault-class transfer."""

    return run_fault_heldout_transfer(
        scored_runs=load_scored_runs_jsonl(scored_runs_path),
        pairs=load_pairwise_examples_jsonl(pairs_path),
        metadata_key=metadata_key,
        min_metadata_groups=min_metadata_groups,
        min_test_pairs_per_fold=min_test_pairs_per_fold,
        source_metadata_key=source_metadata_key,
        min_source_groups=min_source_groups,
        min_source_test_pairs_per_fold=min_source_test_pairs_per_fold,
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
    min_metadata_groups: int = 2,
    min_test_pairs_per_fold: int = 1,
    source_metadata_key: str = "source",
    min_source_groups: int = 2,
    min_source_test_pairs_per_fold: int = 1,
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
    metadata_values = _metadata_values(pairs, metadata_key)
    summary = summarize_fold_results(folds)
    source_folds = evaluate_metadata_transfer(
        pairs=pairs,
        run_index=run_index,
        metadata_key=source_metadata_key,
        split_name="source",
    )
    source_values = _metadata_values(pairs, source_metadata_key)
    source_summary = summarize_fold_results(source_folds)
    readiness = _transfer_readiness(
        pairs=pairs,
        folds=folds,
        metadata_key=metadata_key,
        metadata_values=metadata_values,
        min_metadata_groups=min_metadata_groups,
        min_test_pairs_per_fold=min_test_pairs_per_fold,
    )
    source_readiness = _transfer_readiness(
        pairs=pairs,
        folds=source_folds,
        metadata_key=source_metadata_key,
        metadata_values=source_values,
        min_metadata_groups=min_source_groups,
        min_test_pairs_per_fold=min_source_test_pairs_per_fold,
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
            "fault_classes": len(metadata_values),
            "metadata_values": sorted(metadata_values),
            "missing_metadata_pairs": _missing_metadata_pair_count(
                pairs,
                metadata_key,
            ),
            "source_counts": _source_counts(pairs),
            "source_metadata_key": source_metadata_key,
            "source_groups": len(source_values),
            "source_values": sorted(source_values),
            "missing_source_pairs": _missing_metadata_pair_count(
                pairs,
                source_metadata_key,
            ),
            "min_metadata_groups": min_metadata_groups,
            "min_test_pairs_per_fold": min_test_pairs_per_fold,
            "min_source_groups": min_source_groups,
            "min_source_test_pairs_per_fold": min_source_test_pairs_per_fold,
        },
        "summary": summary,
        "readiness": readiness,
        "folds": folds,
        "source_transfer": {
            "metadata_key": source_metadata_key,
            "summary": source_summary,
            "readiness": source_readiness,
            "folds": source_folds,
        },
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
    source_transfer = _mapping(report.get("source_transfer"))
    source_readiness = _mapping(source_transfer.get("readiness"))
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
        f"- Missing metadata pairs: {int(inputs.get('missing_metadata_pairs', 0))}",
        f"- Source groups: {int(inputs.get('source_groups', 0))}",
        f"- Missing source pairs: {int(inputs.get('missing_source_pairs', 0))}",
        f"- Transfer readiness: "
        f"`{_mapping(report.get('readiness')).get('status', 'not_ready')}`",
        f"- Source transfer readiness: "
        f"`{source_readiness.get('status', 'not_ready')}`",
        f"- Ready for downstream claims: "
        f"{bool(_mapping(report.get('readiness')).get('ready', False))}",
        "",
        "## Readiness Gates",
        "",
    ]
    readiness = _mapping(report.get("readiness"))
    lines.extend(_readiness_gate_lines(readiness))
    failed_groups = _sequence(readiness.get("failed_metadata_values"))
    if failed_groups:
        lines.extend(
            [
                "",
                "Not ready for downstream claims: held-out folds are incomplete "
                f"for {', '.join(str(group) for group in failed_groups)}.",
            ]
        )
    source_counts = _mapping(inputs.get("source_counts"))
    if source_counts:
        lines.extend(
            [
                "",
                "## Source Coverage",
                "",
                "| Source | Pairs |",
                "| --- | ---: |",
            ]
        )
        for source, count in sorted(source_counts.items()):
            lines.append(f"| `{source}` | {int(count)} |")
    if source_transfer:
        lines.extend(_source_transfer_lines(source_transfer))
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Split | Baseline | Folds | Test pairs | Mean test accuracy | Mean test margin | Mean train accuracy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
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


def _readiness_gate_lines(readiness: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for gate in _sequence(readiness.get("gates")):
        gate_map = _mapping(gate)
        lines.append(
            "| "
            f"{gate_map.get('gate_id', '')} | "
            f"{float(gate_map.get('value', 0.0)):.3f} | "
            f"{float(gate_map.get('threshold', 0.0)):.3f} | "
            f"{bool(gate_map.get('passed', False))} |"
        )
    return lines


def _source_transfer_lines(source_transfer: Mapping[str, Any]) -> list[str]:
    readiness = _mapping(source_transfer.get("readiness"))
    folds = _sequence(source_transfer.get("folds"))
    lines = [
        "",
        "## Source-Held-Out Transfer",
        "",
        f"- Source metadata key: `{source_transfer.get('metadata_key', 'source')}`",
        f"- Source transfer readiness: `{readiness.get('status', 'not_ready')}`",
        f"- Ready for source-transfer claims: "
        f"{bool(readiness.get('ready', False))}",
        "",
        "### Source Readiness Gates",
        "",
    ]
    lines.extend(_readiness_gate_lines(readiness))
    failed_sources = _sequence(readiness.get("failed_metadata_values"))
    if failed_sources:
        lines.extend(
            [
                "",
                "Not ready for source-transfer claims: held-out source folds "
                f"are incomplete for {', '.join(str(item) for item in failed_sources)}.",
            ]
        )
    lines.extend(
        _summary_table_lines(
            title="Source Transfer Summary",
            summary=_sequence(source_transfer.get("summary")),
            folds=folds,
        )
    )
    lines.extend(
        _fold_table_lines(
            title="Held-Out Source Folds",
            folds=folds,
            held_out_label="Held-out source",
        )
    )
    return lines


def _summary_table_lines(
    *,
    title: str,
    summary: Sequence[object],
    folds: Sequence[object],
) -> list[str]:
    if not summary:
        return []
    lines = [
        "",
        f"### {title}",
        "",
        "| Split | Baseline | Folds | Test pairs | Mean test accuracy | Mean test margin | Mean train accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        row_map = _mapping(row)
        fold_rows = [
            fold
            for fold in folds
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
    return lines


def _fold_table_lines(
    *,
    title: str,
    folds: Sequence[object],
    held_out_label: str,
) -> list[str]:
    if not folds:
        return []
    lines = [
        "",
        f"### {title}",
        "",
        f"| {held_out_label} | Baseline | Train pairs | Train acc | Test pairs | Test acc | Test margin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in folds:
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
    return lines


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


def _missing_metadata_pair_count(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> int:
    return sum(1 for pair in pairs if not str(pair.metadata.get(metadata_key, "")).strip())


def _source_counts(pairs: Sequence[PairwiseExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        source = str(pair.metadata.get("source", "unknown") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _transfer_readiness(
    *,
    pairs: Sequence[PairwiseExample],
    folds: Sequence[Mapping[str, Any]],
    metadata_key: str,
    metadata_values: set[str],
    min_metadata_groups: int,
    min_test_pairs_per_fold: int,
) -> dict[str, Any]:
    missing_metadata_pairs = _missing_metadata_pair_count(pairs, metadata_key)
    held_out_values = {str(fold.get("held_out", "")) for fold in folds}
    failed_metadata_values = sorted(
        value
        for value in metadata_values
        if _min_test_pairs_for_held_out(folds, value) < min_test_pairs_per_fold
    )
    skipped_pairs = sum(
        int(_mapping(fold.get("train")).get("skipped_pairs", 0))
        + int(_mapping(fold.get("test")).get("skipped_pairs", 0))
        for fold in folds
    )
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(len(pairs)),
            "threshold": 1.0,
            "passed": len(pairs) > 0,
        },
        {
            "gate_id": "metadata_group_count",
            "value": float(len(metadata_values)),
            "threshold": float(min_metadata_groups),
            "passed": len(metadata_values) >= min_metadata_groups,
        },
        {
            "gate_id": "metadata_values_have_folds",
            "value": float(len(held_out_values & metadata_values)),
            "threshold": float(len(metadata_values)),
            "passed": metadata_values <= held_out_values,
        },
        {
            "gate_id": "min_test_pairs_per_fold",
            "value": float(
                min(
                    (
                        _min_test_pairs_for_held_out(folds, value)
                        for value in metadata_values
                    ),
                    default=0,
                )
            ),
            "threshold": float(min_test_pairs_per_fold),
            "passed": not failed_metadata_values,
        },
        {
            "gate_id": "missing_metadata_pairs",
            "value": float(missing_metadata_pairs),
            "threshold": 0.0,
            "passed": missing_metadata_pairs == 0,
        },
        {
            "gate_id": "skipped_pairs",
            "value": float(skipped_pairs),
            "threshold": 0.0,
            "passed": skipped_pairs == 0,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "transfer_ready" if ready else "not_ready_for_transfer_claims",
        "ready": ready,
        "metadata_key": metadata_key,
        "min_metadata_groups": min_metadata_groups,
        "min_test_pairs_per_fold": min_test_pairs_per_fold,
        "failed_metadata_values": failed_metadata_values,
        "source_counts": _source_counts(pairs),
        "gates": gates,
    }


def _min_test_pairs_for_held_out(
    folds: Sequence[Mapping[str, Any]],
    held_out: str,
) -> int:
    counts = [
        int(_mapping(fold.get("test")).get("n_pairs", 0))
        for fold in folds
        if str(fold.get("held_out", "")) == held_out
    ]
    return min(counts) if counts else 0


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
