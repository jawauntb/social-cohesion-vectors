"""Diagnose fresh-generated residual failures in bridge directions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.availability_audit import (
    run_availability_audit,
)
from social_cohesion_vectors.experiments.lexical_leakage import lexical_cue_score
from social_cohesion_vectors.schemas import PairwiseExample

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def run_fresh_generated_residual_diagnostic_from_files(
    *,
    bridge_report_path: str | Path,
    source_pairs_path: str | Path,
    fresh_source_pairs_path: str | Path,
    reference_bridge_report_path: str | Path | None = None,
    model_name: str = "model",
    reference_model_name: str = "reference",
    evaluation_key: str = "on_fresh_source",
) -> dict[str, Any]:
    """Load bridge reports and pair manifests, then explain fresh failures."""

    bridge_report = _read_json(bridge_report_path)
    reference_report = (
        _read_json(reference_bridge_report_path)
        if reference_bridge_report_path is not None
        else None
    )
    source_pairs = load_pairwise_examples_jsonl(source_pairs_path)
    fresh_pairs = load_pairwise_examples_jsonl(fresh_source_pairs_path)
    return run_fresh_generated_residual_diagnostic(
        bridge_report=bridge_report,
        source_pairs=source_pairs,
        fresh_pairs=fresh_pairs,
        reference_bridge_report=reference_report,
        input_paths={
            "bridge_report": str(bridge_report_path),
            "source_pairs": str(source_pairs_path),
            "fresh_source_pairs": str(fresh_source_pairs_path),
            "reference_bridge_report": (
                str(reference_bridge_report_path)
                if reference_bridge_report_path is not None
                else None
            ),
        },
        model_name=model_name,
        reference_model_name=reference_model_name,
        evaluation_key=evaluation_key,
    )


def run_fresh_generated_residual_diagnostic(
    *,
    bridge_report: Mapping[str, Any],
    source_pairs: Sequence[PairwiseExample],
    fresh_pairs: Sequence[PairwiseExample],
    reference_bridge_report: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, str | None] | None = None,
    model_name: str = "model",
    reference_model_name: str = "reference",
    evaluation_key: str = "on_fresh_source",
) -> dict[str, Any]:
    """Compare fresh generated residuals against source-family controls."""

    constructed_rows = _direction_rows(bridge_report, family="constructed_bridge")
    source_margins = _pair_margins(constructed_rows, "on_source")
    fresh_margins = _pair_margins(constructed_rows, evaluation_key)
    reference_fresh_margins = (
        _pair_margins(
            _direction_rows(reference_bridge_report, family="constructed_bridge"),
            evaluation_key,
        )
        if reference_bridge_report is not None
        else {}
    )
    availability_by_pair = _availability_by_pair(fresh_pairs)
    source_by_base = _source_margin_summary_by_base(
        source_pairs,
        pair_margin_summary=_pair_margin_summary(source_margins),
    )
    residual_rows = [
        _fresh_pair_residual_row(
            pair=pair,
            fresh_summary=_summarize_margins(fresh_margins.get(pair.pair_id, ())),
            reference_summary=_summarize_margins(
                reference_fresh_margins.get(pair.pair_id, ())
            ),
            source_same_base_summary=source_by_base.get(_base_id(pair), {}),
            availability_row=availability_by_pair.get(pair.pair_id, {}),
        )
        for pair in fresh_pairs
    ]
    residual_rows = sorted(
        residual_rows,
        key=lambda row: (
            -int(row["constructed_failure_count"]),
            float(row["constructed_min_margin"]),
            str(row["pair_id"]),
        ),
    )
    readiness = _readiness(residual_rows)
    return {
        "experiment": "fresh_generated_residual_diagnostic",
        "description": (
            "Compares fresh generated bridge failures against original "
            "source-family margins, practical availability, lexical cues, and "
            "length features."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "model_name": model_name,
            "reference_model_name": reference_model_name,
            "evaluation_key": evaluation_key,
            "constructed_direction_count": len(constructed_rows),
            "source_pairs": len(source_pairs),
            "fresh_pairs": len(fresh_pairs),
        },
        "summary": _summary(
            residual_rows,
            readiness=readiness,
            model_name=model_name,
            reference_model_name=reference_model_name,
        ),
        "readiness": readiness,
        "residual_pairs": residual_rows,
        "interpretation_guardrail": (
            "Fresh-generated residual diagnostics are text-benchmark activation "
            "diagnostics. They do not establish causal steering, human "
            "behavioral, neural, clinical, deployment, or real-world "
            "social-effect claims."
        ),
    }


def save_fresh_generated_residual_diagnostic(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown residual diagnostic reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fresh_generated_residual_diagnostic_markdown(report),
        encoding="utf-8",
    )


def render_fresh_generated_residual_diagnostic_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render residual diagnostic results as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Fresh Generated Residual Diagnostic",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Model: `{inputs.get('model_name', '')}`",
        f"- Reference model: `{inputs.get('reference_model_name', '')}`",
        f"- Constructed directions: "
        f"{int(inputs.get('constructed_direction_count', 0))}",
        f"- Fresh pairs: {int(inputs.get('fresh_pairs', 0))}",
        "",
        "## Summary",
        "",
        f"- Readiness: `{summary.get('readiness', 'unknown')}`",
        f"- Failing fresh pairs: {int(summary.get('failing_fresh_pairs', 0))}",
        f"- Minimum constructed fresh margin: "
        f"{float(summary.get('constructed_fresh_min_margin', 0.0)):+.3f}",
        f"- Failing pairs with positive availability: "
        f"{int(summary.get('failing_pairs_with_positive_availability', 0))}",
        f"- Failing pairs with source same-base positive margins: "
        f"{int(summary.get('failing_pairs_with_positive_source_same_base', 0))}",
        "",
        "## Residual Pairs",
        "",
        *_residual_pair_table(report.get("residual_pairs")),
        "",
        "## Interpretation Guardrail",
        "",
        str(report.get("interpretation_guardrail", "")),
        "",
    ]
    return "\n".join(lines)


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _direction_rows(
    report: Mapping[str, Any] | None,
    *,
    family: str,
) -> list[Mapping[str, Any]]:
    if report is None:
        return []
    rows = _sequence(report.get("direction_evaluations"))
    return [
        _mapping(row)
        for row in rows
        if str(_mapping(row).get("direction_family", "")) == family
    ]


def _pair_margins(
    direction_rows: Sequence[Mapping[str, Any]],
    evaluation_key: str,
) -> dict[str, list[float]]:
    margins: dict[str, list[float]] = defaultdict(list)
    for row in direction_rows:
        evaluation = _mapping(row.get(evaluation_key))
        for raw_margin in _sequence(evaluation.get("pair_margins")):
            margin = _mapping(raw_margin)
            pair_id = str(margin.get("pair_id", "")).strip()
            if pair_id:
                margins[pair_id].append(float(margin.get("margin", 0.0)))
    return dict(margins)


def _pair_margin_summary(
    margins_by_pair: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    return {
        pair_id: _summarize_margins(margins)
        for pair_id, margins in margins_by_pair.items()
    }


def _summarize_margins(margins: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in margins]
    failed = [value for value in values if value <= 0.0]
    return {
        "direction_count": len(values),
        "failure_count": len(failed),
        "min_margin": round(min(values), 6) if values else 0.0,
        "mean_margin": round(sum(values) / len(values), 6) if values else 0.0,
    }


def _source_margin_summary_by_base(
    source_pairs: Sequence[PairwiseExample],
    *,
    pair_margin_summary: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, float | int]]] = defaultdict(list)
    for pair in source_pairs:
        summary = pair_margin_summary.get(pair.pair_id)
        if summary is not None:
            grouped[_base_id(pair)].append(summary)
    result: dict[str, dict[str, float | int]] = {}
    for base_id, summaries in grouped.items():
        min_margins = [float(summary.get("min_margin", 0.0)) for summary in summaries]
        failure_counts = [int(summary.get("failure_count", 0)) for summary in summaries]
        result[base_id] = {
            "source_same_base_pairs": len(summaries),
            "source_same_base_min_margin": (
                round(min(min_margins), 6) if min_margins else 0.0
            ),
            "source_same_base_failure_count": sum(failure_counts),
        }
    return result


def _availability_by_pair(
    pairs: Sequence[PairwiseExample],
) -> dict[str, Mapping[str, Any]]:
    report = run_availability_audit(pairs=pairs)
    return {str(row.get("pair_id", "")): _mapping(row) for row in report["pairs"]}


def _fresh_pair_residual_row(
    *,
    pair: PairwiseExample,
    fresh_summary: Mapping[str, float | int],
    reference_summary: Mapping[str, float | int],
    source_same_base_summary: Mapping[str, float | int],
    availability_row: Mapping[str, Any],
) -> dict[str, Any]:
    positive_words = _word_count(pair.positive_text)
    negative_words = _word_count(pair.negative_text)
    lexical_margin = lexical_cue_score(pair.positive_text) - lexical_cue_score(
        pair.negative_text
    )
    failure_count = int(fresh_summary.get("failure_count", 0))
    source_min = float(source_same_base_summary.get("source_same_base_min_margin", 0.0))
    availability_min = float(availability_row.get("min_availability_margin", 0.0))
    return {
        "pair_id": pair.pair_id,
        "base_contrast_id": _base_id(pair),
        "primary_fault_class": str(pair.metadata.get("primary_fault_class", "")),
        "slack_options_tested": str(pair.metadata.get("slack_options_tested", "")),
        "constructed_direction_count": int(fresh_summary.get("direction_count", 0)),
        "constructed_failure_count": failure_count,
        "constructed_min_margin": float(fresh_summary.get("min_margin", 0.0)),
        "constructed_mean_margin": float(fresh_summary.get("mean_margin", 0.0)),
        "reference_failure_count": int(reference_summary.get("failure_count", 0)),
        "reference_min_margin": float(reference_summary.get("min_margin", 0.0)),
        **dict(source_same_base_summary),
        "positive_word_count": positive_words,
        "negative_word_count": negative_words,
        "length_delta_words": positive_words - negative_words,
        "lexical_cue_margin": round(lexical_margin, 6),
        "availability_min_margin": availability_min,
        "availability_all_paths_prefer_genuine": bool(
            availability_row.get("all_paths_prefer_genuine", False)
        ),
        "residual_type": _residual_type(
            failure_count=failure_count,
            source_min=source_min,
            availability_min=availability_min,
        ),
    }


def _residual_type(
    *,
    failure_count: int,
    source_min: float,
    availability_min: float,
) -> str:
    if failure_count <= 0:
        return "passes_constructed_bridge"
    if availability_min <= 0.0:
        return "fresh_activation_failure_with_availability_failure"
    if source_min > 0.0:
        return "fresh_only_activation_failure"
    return "fresh_and_source_activation_failure"


def _summary(
    residual_rows: Sequence[Mapping[str, Any]],
    *,
    readiness: Mapping[str, Any],
    model_name: str,
    reference_model_name: str,
) -> dict[str, Any]:
    failing = [
        row for row in residual_rows if int(row.get("constructed_failure_count", 0)) > 0
    ]
    fault_counts = Counter(str(row.get("primary_fault_class", "")) for row in failing)
    margins = [float(row.get("constructed_min_margin", 0.0)) for row in residual_rows]
    return {
        "readiness": str(readiness.get("status", "")),
        "model_name": model_name,
        "reference_model_name": reference_model_name,
        "fresh_pairs": len(residual_rows),
        "failing_fresh_pairs": len(failing),
        "constructed_fresh_min_margin": round(min(margins), 6) if margins else 0.0,
        "failing_pairs_with_positive_availability": sum(
            float(row.get("availability_min_margin", 0.0)) > 0.0 for row in failing
        ),
        "failing_pairs_with_positive_source_same_base": sum(
            float(row.get("source_same_base_min_margin", 0.0)) > 0.0
            for row in failing
        ),
        "failure_counts_by_primary_fault_class": dict(sorted(fault_counts.items())),
    }


def _readiness(residual_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failing = [
        row for row in residual_rows if int(row.get("constructed_failure_count", 0)) > 0
    ]
    status = "fresh_generated_residual_present" if failing else "no_residual_detected"
    return {
        "status": status,
        "ready": not failing,
        "failing_pair_ids": [str(row.get("pair_id", "")) for row in failing],
    }


def _residual_pair_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No residual rows."]
    lines = [
        "| Pair | Fault | Failures | Fresh min | Reference min | Source same-base min | Availability min | Lexical margin | Residual |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('base_contrast_id', '')}` | "
            f"`{row.get('primary_fault_class', '')}` | "
            f"{int(row.get('constructed_failure_count', 0))} | "
            f"{float(row.get('constructed_min_margin', 0.0)):+.3f} | "
            f"{float(row.get('reference_min_margin', 0.0)):+.3f} | "
            f"{float(row.get('source_same_base_min_margin', 0.0)):+.3f} | "
            f"{float(row.get('availability_min_margin', 0.0)):+.3f} | "
            f"{float(row.get('lexical_cue_margin', 0.0)):+.3f} | "
            f"`{row.get('residual_type', '')}` |"
        )
    return lines


def _base_id(pair: PairwiseExample) -> str:
    raw = str(pair.metadata.get("base_contrast_id") or pair.pair_id)
    raw = raw.removeprefix("generated-fault::")
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    return raw


def _word_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _mapping(raw_value: object) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _sequence(raw_value: object) -> list[Any]:
    return list(raw_value) if isinstance(raw_value, Sequence) and not isinstance(raw_value, str) else []
