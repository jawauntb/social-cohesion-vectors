"""Projection checks for generated text from activation-steering sweeps."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

TextMode = Literal["response", "prompt_response"]

_SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<report>.+)::prompt=(?P<prompt_id>.+)::strength=(?P<strength>[+-]?(?:\d+\.?\d*|\.\d+))$"
)
_EPSILON = 1e-12


def projection_prompt_records_from_report_paths(
    report_paths: Sequence[str | Path],
    *,
    text_mode: TextMode = "response",
) -> list[dict[str, Any]]:
    """Build activation prompt records from one or more steering reports."""

    records: list[dict[str, Any]] = []
    for report_path in report_paths:
        path = Path(report_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        records.extend(
            projection_prompt_records(
                report.get("records", []),
                report_name=path.stem,
                text_mode=text_mode,
            )
        )
    return records


def projection_prompt_records(
    records: Iterable[Mapping[str, Any]],
    *,
    report_name: str,
    text_mode: TextMode = "response",
) -> list[dict[str, Any]]:
    """Convert steered generation records into activation-extraction prompts."""

    prompt_records: list[dict[str, Any]] = []
    for record in records:
        prompt_id = str(record.get("prompt_id", ""))
        if not prompt_id:
            continue
        strength = float(record.get("strength", 0.0))
        text = _projection_text(record, text_mode=text_mode)
        if not text.strip():
            continue
        prompt_records.append(
            {
                "sample_id": _sample_id(
                    report_name=report_name,
                    prompt_id=prompt_id,
                    strength=strength,
                ),
                "pair_id": f"{report_name}::{prompt_id}",
                "label": _strength_label(strength),
                "target_score": float(record.get("cohesion_score", math.nan)),
                "text": text,
            }
        )
    return prompt_records


def summarize_generation_projection(
    *,
    activation_npz: str | Path,
    direction_npz: str | Path,
) -> dict[str, Any]:
    """Project extracted generation activations onto a saved direction."""

    activations, metadata = _load_activation_npz(activation_npz)
    direction = _load_direction(direction_npz)
    if activations.shape[1] != direction.shape[0]:
        raise ValueError(
            f"Activation dimension {activations.shape[1]} does not match "
            f"direction dimension {direction.shape[0]}."
        )
    projections = activations @ direction
    rows = _report_rows(metadata=metadata, projections=projections)
    return {
        "experiment": "steered_generation_projection",
        "description": (
            "Re-embeds generated responses from activation-steering sweeps and "
            "projects them onto the original contrastive direction."
        ),
        "activation_npz": str(activation_npz),
        "direction_npz": str(direction_npz),
        "samples": int(activations.shape[0]),
        "activation_dim": int(activations.shape[1]),
        "rows": rows,
    }


def write_generation_projection_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write projection report JSON and Markdown files."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_generation_projection_markdown(report), encoding="utf-8"
    )


def render_generation_projection_markdown(report: Mapping[str, Any]) -> str:
    """Render the projection report as a compact Markdown table."""

    lines = [
        "# Steered Generation Projection Check",
        "",
        str(report.get("description", "")),
        "",
        f"- Samples: {int(report.get('samples', 0))}",
        f"- Activation dim: {int(report.get('activation_dim', 0))}",
        "",
        "| Report | Strengths | Projection win | Proj pos-base | Proj pos-neg | Score win | Score pos-base | Score pos-neg | Projection-score r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("rows")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('report', '')} | "
            f"{row_map.get('strengths', [])} | "
            f"{float(row_map.get('projection_success_rate', 0.0)):.3f} | "
            f"{float(row_map.get('projection_positive_minus_baseline_mean_delta', 0.0)):+.4f} | "
            f"{float(row_map.get('projection_positive_minus_negative_mean_delta', 0.0)):+.4f} | "
            f"{float(row_map.get('score_success_rate', 0.0)):.3f} | "
            f"{float(row_map.get('score_positive_minus_baseline_mean_delta', 0.0)):+.4f} | "
            f"{float(row_map.get('score_positive_minus_negative_mean_delta', 0.0)):+.4f} | "
            f"{float(row_map.get('projection_score_correlation', 0.0)):+.3f} |"
        )
    strength_rows = _all_strength_rows(report)
    if strength_rows:
        lines.extend(
            [
                "",
                "## Strength Means",
                "",
                "| Report | Strength | Samples | Mean projection | Mean score |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in strength_rows:
            lines.append(
                "| "
                f"{row['report']} | "
                f"{float(row['strength']):+.2f} | "
                f"{int(row['samples'])} | "
                f"{float(row['mean_projection']):+.4f} | "
                f"{float(row['mean_score']):.4f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation note: this checks whether steered generations move in "
            "the learned activation direction after re-embedding. It does not by "
            "itself establish behavioral or human-social validity.",
            "",
        ]
    )
    return "\n".join(lines)


def _all_strength_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _sequence(report.get("rows")):
        row_map = _mapping(row)
        report_name = str(row_map.get("report", ""))
        for strength_row in _sequence(row_map.get("by_strength")):
            strength_map = _mapping(strength_row)
            rows.append(
                {
                    "report": report_name,
                    "strength": float(strength_map.get("strength", 0.0)),
                    "samples": int(strength_map.get("samples", 0)),
                    "mean_projection": float(strength_map.get("mean_projection", 0.0)),
                    "mean_score": float(strength_map.get("mean_score", 0.0)),
                }
            )
    return rows


def _projection_text(record: Mapping[str, Any], *, text_mode: TextMode) -> str:
    response = str(record.get("generated_text", ""))
    if text_mode == "response":
        return response
    prompt = str(record.get("prompt", record.get("text", "")))
    return "\n\n".join(["Prompt:", prompt, "Model response:", response])


def _sample_id(*, report_name: str, prompt_id: str, strength: float) -> str:
    return f"{report_name}::prompt={prompt_id}::strength={strength:g}"


def _strength_label(strength: float) -> str:
    if strength > 0:
        return "positive"
    if strength < 0:
        return "negative"
    return "baseline"


def _load_activation_npz(path: str | Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(Path(path), allow_pickle=False) as data:
        activations = np.asarray(data["activations"], dtype=np.float64)
        sample_ids = np.asarray(data["sample_ids"], dtype=str)
        pair_ids = np.asarray(data["pair_ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=str)
        target_scores = np.asarray(data["target_scores"], dtype=np.float64)
    metadata = [
        _metadata_from_sample(
            sample_id=str(sample_id),
            pair_id=str(pair_id),
            label=str(label),
            target_score=float(target_score),
        )
        for sample_id, pair_id, label, target_score in zip(
            sample_ids,
            pair_ids,
            labels,
            target_scores,
            strict=True,
        )
    ]
    return activations, metadata


def _load_direction(path: str | Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        direction = np.asarray(data["direction"], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= _EPSILON:
        raise ValueError("Direction vector has zero norm.")
    return direction / norm


def _metadata_from_sample(
    *,
    sample_id: str,
    pair_id: str,
    label: str,
    target_score: float,
) -> dict[str, Any]:
    match = _SAMPLE_ID_PATTERN.match(sample_id)
    if match is None:
        raise ValueError(f"Unrecognized steered generation sample_id: {sample_id}")
    return {
        "sample_id": sample_id,
        "report": match.group("report"),
        "prompt_id": match.group("prompt_id"),
        "strength": float(match.group("strength")),
        "pair_id": pair_id,
        "label": label,
        "target_score": target_score,
    }


def _report_rows(
    *,
    metadata: Sequence[Mapping[str, Any]],
    projections: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for meta, projection in zip(metadata, projections, strict=True):
        grouped[str(meta["report"])].append((meta, float(projection)))
    rows = [
        _single_report_row(report_name, values)
        for report_name, values in sorted(grouped.items())
    ]
    rows.sort(
        key=lambda row: (
            -float(row["projection_success_rate"]),
            -float(row["projection_positive_minus_negative_mean_delta"]),
            str(row["report"]),
        )
    )
    return rows


def _single_report_row(
    report_name: str,
    values: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, Any]:
    strengths = sorted({float(meta["strength"]) for meta, _ in values})
    negative_strength = min(strengths) if strengths else 0.0
    baseline_strength = min(strengths, key=abs) if strengths else 0.0
    positive_strength = max(strengths) if strengths else 0.0
    pair_rows = _pair_rows(
        values,
        baseline_strength=baseline_strength,
        negative_strength=negative_strength,
        positive_strength=positive_strength,
    )
    return {
        "report": report_name,
        "strengths": strengths,
        "samples": len(values),
        "projection_success_rate": _success_rate(
            row["projection_margin"] for row in pair_rows
        ),
        "projection_positive_minus_baseline_mean_delta": _mean(
            row["projection_positive_minus_baseline"] for row in pair_rows
        ),
        "projection_positive_minus_negative_mean_delta": _mean(
            row["projection_margin"] for row in pair_rows
        ),
        "score_success_rate": _success_rate(row["score_margin"] for row in pair_rows),
        "score_positive_minus_baseline_mean_delta": _mean(
            row["score_positive_minus_baseline"] for row in pair_rows
        ),
        "score_positive_minus_negative_mean_delta": _mean(
            row["score_margin"] for row in pair_rows
        ),
        "projection_score_correlation": _correlation(
            [projection for _, projection in values],
            [float(meta["target_score"]) for meta, _ in values],
        ),
        "by_strength": _strength_rows(values),
    }


def _pair_rows(
    values: Sequence[tuple[Mapping[str, Any], float]],
    *,
    baseline_strength: float,
    negative_strength: float,
    positive_strength: float,
) -> list[dict[str, float]]:
    grouped: dict[str, dict[float, tuple[Mapping[str, Any], float]]] = defaultdict(dict)
    for meta, projection in values:
        grouped[str(meta["pair_id"])][float(meta["strength"])] = (meta, projection)
    rows: list[dict[str, float]] = []
    for strengths in grouped.values():
        if (
            baseline_strength not in strengths
            or negative_strength not in strengths
            or positive_strength not in strengths
        ):
            continue
        baseline_meta, baseline_projection = strengths[baseline_strength]
        negative_meta, negative_projection = strengths[negative_strength]
        positive_meta, positive_projection = strengths[positive_strength]
        rows.append(
            {
                "projection_margin": positive_projection - negative_projection,
                "projection_positive_minus_baseline": positive_projection
                - baseline_projection,
                "score_margin": float(positive_meta["target_score"])
                - float(negative_meta["target_score"]),
                "score_positive_minus_baseline": float(positive_meta["target_score"])
                - float(baseline_meta["target_score"]),
            }
        )
    return rows


def _strength_rows(
    values: Sequence[tuple[Mapping[str, Any], float]],
) -> list[dict[str, float | int]]:
    grouped: dict[float, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for meta, projection in values:
        grouped[float(meta["strength"])].append((meta, projection))
    return [
        {
            "strength": strength,
            "samples": len(rows),
            "mean_projection": _mean(projection for _, projection in rows),
            "mean_score": _mean(float(meta["target_score"]) for meta, _ in rows),
        }
        for strength, rows in sorted(grouped.items())
    ]


def _success_rate(margins: Iterable[float]) -> float:
    outcomes = [
        1.0 if margin > 0 else 0.5 if margin == 0 else 0.0 for margin in margins
    ]
    return round(sum(outcomes) / len(outcomes), 6) if outcomes else 0.0


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return round(float(np.mean(finite)), 6) if finite else 0.0


def _correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=True)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return 0.0
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(np.std(x)) <= _EPSILON or float(np.std(y)) <= _EPSILON:
        return 0.0
    return round(float(np.corrcoef(x, y)[0, 1]), 6)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
