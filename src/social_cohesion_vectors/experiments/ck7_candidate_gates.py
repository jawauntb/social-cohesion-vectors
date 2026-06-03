"""Promotion gates for CK-7 compute-only candidate trials."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_CK1_DELTA_KEYS = (
    "candidate_minus_baseline_mean_ck1_delta",
    "positive_minus_baseline_mean_ck1_delta",
    "best_minus_baseline_mean_ck1_delta",
    "mean_ck1_delta_vs_baseline",
)
_CK1_SUCCESS_KEYS = (
    "candidate_vs_baseline_ck1_success_rate",
    "positive_vs_baseline_ck1_success_rate",
)
_PSEUDO_RISK_DELTA_KEYS = (
    "candidate_minus_baseline_mean_pseudo_risk_delta",
    "positive_minus_baseline_mean_pseudo_risk_delta",
    "mean_pseudo_delta_vs_baseline",
    "positive_minus_negative_mean_pseudo_risk_delta",
)
_POST_PROJECTION_DELTA_KEYS = (
    "candidate_minus_baseline_post_projection_delta",
    "positive_minus_baseline_post_projection_delta",
    "projection_positive_minus_baseline_mean_delta",
)
_PROJECTION_SUCCESS_KEYS = (
    "projection_success_rate",
    "target_projection_success_rate",
)
_DELTA_ERROR_KEYS = (
    "mean_absolute_delta_error",
    "mean_abs_projection_delta_error",
)
_WASHOUT_CK1_KEYS = (
    "washout_ck1_delta_vs_baseline",
    "post_washout_ck1_delta",
    "ck1_delta_vs_baseline",
)
_WASHOUT_PSEUDO_KEYS = (
    "washout_pseudo_risk_delta_vs_baseline",
    "post_washout_pseudo_risk_delta",
    "pseudo_risk_delta_vs_baseline",
)
_WASHOUT_PROJECTION_KEYS = (
    "washout_projection_delta_vs_baseline",
    "post_washout_projection_delta",
    "projection_delta_vs_baseline",
)


@dataclass(frozen=True)
class CK7GateThresholds:
    """Default compute-only promotion thresholds for CK-7 candidates."""

    min_ck1_mean_delta: float = 0.02
    min_ck1_success_rate: float = 0.67
    max_pseudo_risk_delta: float = 0.0
    min_post_projection_delta: float = 0.10
    min_projection_success_rate: float = 0.67
    max_mean_absolute_delta_error: float = 0.05
    blocking_side_effect_severity: float = 1.0
    max_washout_ck1_abs_delta: float = 0.01
    max_washout_pseudo_risk_delta: float = 0.0
    max_washout_projection_abs_delta: float = 0.05


def evaluate_ck7_candidate_gates(
    *,
    candidate_id: str,
    score_report: Mapping[str, Any],
    telemetry_report: Mapping[str, Any],
    washout_report: Mapping[str, Any],
    side_effect_flags: Sequence[object] = (),
    candidate_recipe_id: str | None = None,
    thresholds: CK7GateThresholds = CK7GateThresholds(),
) -> dict[str, Any]:
    """Evaluate CK-7 candidate-trial promotion gates from existing reports."""

    score_metrics = _score_metrics(score_report, candidate_recipe_id)
    telemetry_metrics = _telemetry_metrics(telemetry_report, candidate_recipe_id)
    washout_metrics = _summary(washout_report)
    flag_rows = _side_effect_flag_rows(side_effect_flags, thresholds)

    ck1_delta = _metric(score_metrics, _CK1_DELTA_KEYS)
    ck1_success = _metric(score_metrics, _CK1_SUCCESS_KEYS)
    pseudo_risk_delta = _metric(score_metrics, _PSEUDO_RISK_DELTA_KEYS)
    projection_delta = _metric(telemetry_metrics, _POST_PROJECTION_DELTA_KEYS)
    projection_success = _metric(telemetry_metrics, _PROJECTION_SUCCESS_KEYS)
    delta_error = _metric(telemetry_metrics, _DELTA_ERROR_KEYS)
    washout_ck1_delta = _metric(washout_metrics, _WASHOUT_CK1_KEYS)
    washout_pseudo_delta = _metric(washout_metrics, _WASHOUT_PSEUDO_KEYS)
    washout_projection_delta = _metric(washout_metrics, _WASHOUT_PROJECTION_KEYS)

    gates = [
        _gate(
            "ck1_improvement",
            "CK-1 improvement",
            _passes_ck1_improvement(ck1_delta, ck1_success, thresholds),
            {
                "mean_ck1_delta": ck1_delta,
                "ck1_success_rate": ck1_success,
                "min_mean_ck1_delta": thresholds.min_ck1_mean_delta,
                "min_success_rate": thresholds.min_ck1_success_rate,
            },
            "Candidate must improve CK-1 score over baseline under pressure.",
        ),
        _gate(
            "pseudo_risk_non_increase",
            "Pseudo-risk non-increase",
            pseudo_risk_delta is not None
            and pseudo_risk_delta <= thresholds.max_pseudo_risk_delta,
            {
                "mean_pseudo_risk_delta": pseudo_risk_delta,
                "max_allowed_delta": thresholds.max_pseudo_risk_delta,
            },
            "Pseudo-attunement risk must not increase versus the comparator.",
        ),
        _gate(
            "projection_telemetry_target_engagement",
            "Projection telemetry target engagement",
            _passes_projection_telemetry(
                projection_delta,
                projection_success,
                delta_error,
                thresholds,
            ),
            {
                "post_projection_delta": projection_delta,
                "projection_success_rate": projection_success,
                "mean_absolute_delta_error": delta_error,
                "min_post_projection_delta": thresholds.min_post_projection_delta,
                "min_projection_success_rate": thresholds.min_projection_success_rate,
                "max_mean_absolute_delta_error": (
                    thresholds.max_mean_absolute_delta_error
                ),
            },
            "Telemetry must show movement on the intended projection target.",
        ),
        _gate(
            "side_effect_flags_clear",
            "Side-effect flags clear",
            not _blocking_flags(flag_rows, thresholds),
            {
                "flag_count": float(len(flag_rows)),
                "blocking_flag_count": float(
                    len(_blocking_flags(flag_rows, thresholds))
                ),
                "blocking_severity_threshold": (
                    thresholds.blocking_side_effect_severity
                ),
            },
            (
                "No blocking flattery, dependency, unsupported certainty, "
                "surveillance lock-in, coercive unity, or dissent suppression flags."
            ),
        ),
        _gate(
            "washout_return_to_baseline",
            "Washout return to baseline",
            _passes_washout(
                washout_ck1_delta,
                washout_pseudo_delta,
                washout_projection_delta,
                thresholds,
            ),
            {
                "washout_ck1_delta": washout_ck1_delta,
                "washout_pseudo_risk_delta": washout_pseudo_delta,
                "washout_projection_delta": washout_projection_delta,
                "max_ck1_abs_delta": thresholds.max_washout_ck1_abs_delta,
                "max_pseudo_risk_delta": thresholds.max_washout_pseudo_risk_delta,
                "max_projection_abs_delta": (
                    thresholds.max_washout_projection_abs_delta
                ),
            },
            "After washout, score, risk, and projection drift must return near baseline.",
        ),
    ]
    passed_gates = sum(1 for gate in gates if gate["passed"])
    promoted = passed_gates == len(gates)
    return {
        "experiment": "ck7_candidate_trial_promotion_gates",
        "description": (
            "Evaluates compute-only promotion gates for boundary-preserving "
            "prosociality candidate trials under pressure. This is not human, "
            "behavioral, neural, biological, clinical, or therapeutic validation."
        ),
        "summary": {
            "candidate_id": candidate_id,
            "candidate_recipe_id": candidate_recipe_id,
            "promoted": promoted,
            "promotion_decision": "promote" if promoted else "hold",
            "gate_count": len(gates),
            "passed_gates": passed_gates,
            "failed_gates": len(gates) - passed_gates,
            "claim_boundary": "compute_only",
        },
        "thresholds": asdict(thresholds),
        "gates": gates,
        "side_effect_flags": flag_rows,
    }


def write_ck7_candidate_gate_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write CK-7 gate report JSON and Markdown."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_ck7_candidate_gates_markdown(report), encoding="utf-8"
    )


def render_ck7_candidate_gates_markdown(report: Mapping[str, Any]) -> str:
    """Render CK-7 gate decisions as Markdown."""

    summary = _summary(report)
    lines = [
        "# CK-7 Candidate Trial Promotion Gates",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Candidate: `{summary.get('candidate_id', '')}`",
        f"- Recipe: `{summary.get('candidate_recipe_id', '')}`",
        f"- Decision: `{str(summary.get('promotion_decision', 'hold')).upper()}`",
        f"- Gates passed: {int(summary.get('passed_gates', 0))}/"
        f"{int(summary.get('gate_count', 0))}",
        "- Claim boundary: compute-only scoring and telemetry; no human or "
        "neural validation.",
        "",
        "## Gates",
        "",
        "| Gate | Status | Requirement | Key metrics |",
        "| --- | --- | --- | --- |",
    ]
    for gate in _sequence(report.get("gates")):
        gate_map = _mapping(gate)
        metrics = _mapping(gate_map.get("metrics"))
        lines.append(
            "| "
            f"{gate_map.get('label', gate_map.get('gate_id', ''))} | "
            f"{'pass' if gate_map.get('passed') else 'fail'} | "
            f"{gate_map.get('requirement', '')} | "
            f"{_format_metrics(metrics)} |"
        )
    flags = _sequence(report.get("side_effect_flags"))
    if flags:
        lines.extend(
            [
                "",
                "## Side-Effect Flags",
                "",
                "| Flag | Severity | Blocking | Notes |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for flag in flags:
            flag_map = _mapping(flag)
            lines.append(
                "| "
                f"{flag_map.get('flag_id', '')} | "
                f"{float(flag_map.get('severity', 0.0)):.2f} | "
                f"{'yes' if flag_map.get('blocking') else 'no'} | "
                f"{flag_map.get('notes', '')} |"
            )
    lines.extend(
        [
            "",
            "Promotion means only that a candidate cleared these deterministic "
            "compute-only scoring and telemetry gates. It does not establish "
            "real human effects, neural mechanisms, biological mechanisms, "
            "clinical relevance, or therapeutic benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def _passes_ck1_improvement(
    ck1_delta: float | None,
    ck1_success: float | None,
    thresholds: CK7GateThresholds,
) -> bool:
    if ck1_delta is None or ck1_delta < thresholds.min_ck1_mean_delta:
        return False
    return ck1_success is None or ck1_success >= thresholds.min_ck1_success_rate


def _passes_projection_telemetry(
    projection_delta: float | None,
    projection_success: float | None,
    delta_error: float | None,
    thresholds: CK7GateThresholds,
) -> bool:
    if projection_delta is None or delta_error is None:
        return False
    success_passes = (
        projection_success is None
        or projection_success >= thresholds.min_projection_success_rate
    )
    return (
        projection_delta >= thresholds.min_post_projection_delta
        and delta_error <= thresholds.max_mean_absolute_delta_error
        and success_passes
    )


def _passes_washout(
    ck1_delta: float | None,
    pseudo_delta: float | None,
    projection_delta: float | None,
    thresholds: CK7GateThresholds,
) -> bool:
    if ck1_delta is None or pseudo_delta is None or projection_delta is None:
        return False
    return (
        abs(ck1_delta) <= thresholds.max_washout_ck1_abs_delta
        and pseudo_delta <= thresholds.max_washout_pseudo_risk_delta
        and abs(projection_delta) <= thresholds.max_washout_projection_abs_delta
    )


def _gate(
    gate_id: str,
    label: str,
    passed: bool,
    metrics: Mapping[str, float | None],
    requirement: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "label": label,
        "passed": bool(passed),
        "requirement": requirement,
        "metrics": {
            key: round(value, 6) if value is not None else None
            for key, value in metrics.items()
        },
    }


def _score_metrics(
    score_report: Mapping[str, Any],
    candidate_recipe_id: str | None,
) -> Mapping[str, Any]:
    metrics = dict(_summary(score_report))
    recipe_row = _row_by_id(
        score_report.get("recipes"), "recipe_id", candidate_recipe_id
    )
    if recipe_row:
        metrics.update(recipe_row)
        _copy_metric(
            metrics,
            source_key="mean_ck1_delta_vs_baseline",
            target_key="candidate_minus_baseline_mean_ck1_delta",
        )
        _copy_metric(
            metrics,
            source_key="mean_pseudo_delta_vs_baseline",
            target_key="candidate_minus_baseline_mean_pseudo_risk_delta",
        )
    return metrics


def _telemetry_metrics(
    telemetry_report: Mapping[str, Any],
    candidate_recipe_id: str | None,
) -> Mapping[str, Any]:
    metrics = dict(_summary(telemetry_report))
    telemetry_row = _row_by_id(
        telemetry_report.get("rows"),
        "report",
        candidate_recipe_id,
    )
    if telemetry_row:
        metrics.update(telemetry_row)
        _copy_metric(
            metrics,
            source_key="projection_positive_minus_baseline_mean_delta",
            target_key="candidate_minus_baseline_post_projection_delta",
        )
    return metrics


def _copy_metric(
    metrics: dict[str, Any],
    *,
    source_key: str,
    target_key: str,
) -> None:
    if source_key in metrics:
        metrics[target_key] = metrics[source_key]


def _row_by_id(
    rows: object,
    id_key: str,
    expected_id: str | None,
) -> Mapping[str, Any]:
    if expected_id is None:
        return {}
    for row in _sequence(rows):
        row_map = _mapping(row)
        if str(row_map.get(id_key, "")) == expected_id:
            return row_map
    return {}


def _side_effect_flag_rows(
    flags: Sequence[object],
    thresholds: CK7GateThresholds,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for flag in flags:
        if isinstance(flag, Mapping):
            flag_id = str(flag.get("flag_id", flag.get("id", flag.get("label", ""))))
            severity = _float_or_default(flag.get("severity"), 1.0)
            notes = str(flag.get("notes", flag.get("note", "")))
        else:
            flag_id = str(flag)
            severity = 1.0
            notes = ""
        rows.append(
            {
                "flag_id": flag_id,
                "severity": round(severity, 6),
                "blocking": severity >= thresholds.blocking_side_effect_severity,
                "notes": notes,
            }
        )
    return rows


def _blocking_flags(
    flags: Sequence[Mapping[str, Any]],
    thresholds: CK7GateThresholds,
) -> list[Mapping[str, Any]]:
    return [
        flag
        for flag in flags
        if float(flag.get("severity", 0.0)) >= thresholds.blocking_side_effect_severity
    ]


def _metric(metrics: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in metrics:
            return _float_or_none(metrics[key])
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_or_default(value: object, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if value is None:
            parts.append(f"{key}=missing")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("summary", report)
    return _mapping(summary)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
