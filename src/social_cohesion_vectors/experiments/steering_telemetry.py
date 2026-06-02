"""Report helpers for in-generation steering projection telemetry."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.experiments.causal_steering import (
    score_steered_generations,
)


def shape_steering_telemetry_report(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize hidden-state projection telemetry and generated-text scores."""

    scored = score_steered_generations(traces)
    trace_rows = [_trace_row(trace) for trace in scored]
    by_strength: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_strength[float(row["strength"])].append(row)
        by_prompt[str(row["prompt_id"])].append(row)

    strengths = sorted(by_strength)
    negative_strength = min(strengths) if strengths else 0.0
    baseline_strength = min(strengths, key=abs) if strengths else 0.0
    positive_strength = max(strengths) if strengths else 0.0
    prompt_margins = _prompt_margins(
        by_prompt,
        negative_strength=negative_strength,
        baseline_strength=baseline_strength,
        positive_strength=positive_strength,
    )
    return {
        "experiment": "steering_hidden_projection_telemetry",
        "description": (
            "Records projection-before and projection-after values at the actual "
            "activation-steering hook during greedy generation."
        ),
        "summary": {
            "traces": len(trace_rows),
            "strengths": strengths,
            "prompt_count": len(by_prompt),
            "mean_absolute_delta_error": _mean(
                abs(row["mean_projection_delta"] - row["strength"])
                for row in trace_rows
                if row["event_count"] > 0
            ),
            "positive_minus_negative_post_projection_delta": _mean(
                row["post_projection_margin"] for row in prompt_margins
            ),
            "positive_minus_baseline_post_projection_delta": _mean(
                row["post_projection_positive_minus_baseline"] for row in prompt_margins
            ),
            "positive_minus_negative_score_delta": _mean(
                row["score_margin"] for row in prompt_margins
            ),
            "positive_minus_baseline_score_delta": _mean(
                row["score_positive_minus_baseline"] for row in prompt_margins
            ),
        },
        "strengths": [
            _strength_row(strength, rows)
            for strength, rows in sorted(by_strength.items())
        ],
        "prompts": prompt_margins,
        "traces": trace_rows,
    }


def write_steering_telemetry_reports(
    traces: Sequence[Mapping[str, Any]],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Write JSON and Markdown telemetry reports."""

    report = shape_steering_telemetry_report(traces)
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_telemetry_markdown(report), encoding="utf-8")
    return report


def render_telemetry_markdown(report: Mapping[str, Any]) -> str:
    """Render a steering telemetry report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Steering Hidden Projection Telemetry",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Traces: {int(summary.get('traces', 0))}",
        f"- Prompts: {int(summary.get('prompt_count', 0))}",
        f"- Strengths: {summary.get('strengths', [])}",
        "- Mean absolute hidden delta error: "
        f"{float(summary.get('mean_absolute_delta_error', 0.0)):.6f}",
        "- Positive-minus-negative post-hook projection delta: "
        f"{float(summary.get('positive_minus_negative_post_projection_delta', 0.0)):+.4f}",
        "- Positive-minus-baseline post-hook projection delta: "
        f"{float(summary.get('positive_minus_baseline_post_projection_delta', 0.0)):+.4f}",
        "- Positive-minus-negative text-score delta: "
        f"{float(summary.get('positive_minus_negative_score_delta', 0.0)):+.4f}",
        "",
        "## Strength Means",
        "",
        "| Strength | Runs | Events/run | Before proj | After proj | Delta | Score | First token |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in _sequence(report.get("strengths")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{float(row_map.get('strength', 0.0)):+.2f} | "
            f"{int(row_map.get('runs', 0))} | "
            f"{float(row_map.get('mean_events', 0.0)):.1f} | "
            f"{float(row_map.get('mean_projection_before', 0.0)):+.4f} | "
            f"{float(row_map.get('mean_projection_after', 0.0)):+.4f} | "
            f"{float(row_map.get('mean_projection_delta', 0.0)):+.4f} | "
            f"{float(row_map.get('mean_cohesion_score', 0.0)):.4f} | "
            f"{row_map.get('first_token_mode', '')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation note: a near-zero hidden delta error means the hook "
            "is applying the requested vector displacement at the targeted hidden "
            "state. It does not imply the downstream generation or score moves "
            "monotonically.",
            "",
        ]
    )
    return "\n".join(lines)


def _trace_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    events = [_mapping(event) for event in _sequence(trace.get("events"))]
    components = _mapping(trace.get("score_components"))
    return {
        "model_id": str(trace.get("model_id", "")),
        "layer": int(trace.get("layer", 0)),
        "hook_site": str(trace.get("hook_site", "")),
        "steering_position": str(trace.get("steering_position", "")),
        "steering_timing": str(trace.get("steering_timing", "")),
        "prompt_id": str(trace.get("prompt_id", "")),
        "mechanism": str(trace.get("mechanism", "")),
        "strength": float(trace.get("strength", 0.0)),
        "event_count": len(events),
        "mean_projection_before": _mean(
            float(event.get("mean_projection_before", 0.0)) for event in events
        ),
        "mean_projection_after": _mean(
            float(event.get("mean_projection_after", 0.0)) for event in events
        ),
        "mean_projection_delta": _mean(
            float(event.get("mean_projection_delta", 0.0)) for event in events
        ),
        "cohesion_score": float(trace.get("cohesion_score", 0.0)),
        "autonomy_safety": float(components.get("autonomy_safety", 0.0)),
        "generated_text": str(trace.get("generated_text", "")),
        "first_token": _first_token(trace),
    }


def _strength_row(strength: float, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "strength": strength,
        "runs": len(rows),
        "mean_events": _mean(float(row["event_count"]) for row in rows),
        "mean_projection_before": _mean(
            float(row["mean_projection_before"]) for row in rows
        ),
        "mean_projection_after": _mean(
            float(row["mean_projection_after"]) for row in rows
        ),
        "mean_projection_delta": _mean(
            float(row["mean_projection_delta"]) for row in rows
        ),
        "mean_cohesion_score": _mean(float(row["cohesion_score"]) for row in rows),
        "mean_autonomy_safety": _mean(float(row["autonomy_safety"]) for row in rows),
        "first_token_mode": _mode(str(row["first_token"]) for row in rows),
    }


def _prompt_margins(
    by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    negative_strength: float,
    baseline_strength: float,
    positive_strength: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in sorted(by_prompt.items()):
        by_strength = {float(row["strength"]): row for row in prompt_rows}
        if (
            negative_strength not in by_strength
            or baseline_strength not in by_strength
            or positive_strength not in by_strength
        ):
            continue
        negative = by_strength[negative_strength]
        baseline = by_strength[baseline_strength]
        positive = by_strength[positive_strength]
        rows.append(
            {
                "prompt_id": prompt_id,
                "mechanism": str(positive.get("mechanism", "")),
                "post_projection_margin": float(positive["mean_projection_after"])
                - float(negative["mean_projection_after"]),
                "post_projection_positive_minus_baseline": float(
                    positive["mean_projection_after"]
                )
                - float(baseline["mean_projection_after"]),
                "score_margin": float(positive["cohesion_score"])
                - float(negative["cohesion_score"]),
                "score_positive_minus_baseline": float(positive["cohesion_score"])
                - float(baseline["cohesion_score"]),
            }
        )
    return rows


def _first_token(trace: Mapping[str, Any]) -> str:
    steps = _sequence(trace.get("steps"))
    if not steps:
        return ""
    return str(_mapping(steps[0]).get("token_text", "")).replace("\n", "\\n")


def _mode(values: Iterable[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return round(float(np.mean(materialized)), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
