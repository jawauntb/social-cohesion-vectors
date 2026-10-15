"""Summarize steering telemetry reports into one layer-comparison table."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = summarize_telemetry_reports(args.reports)
    write_telemetry_summary(
        summary,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    print(f"wrote {args.markdown_output} rows={len(summary['rows'])}")
    return 0


def summarize_telemetry_reports(paths: Sequence[Path]) -> dict[str, Any]:
    """Return sortable rows from telemetry report JSON files."""

    rows = [_row_from_report(path) for path in sorted(paths)]
    rows.sort(
        key=lambda row: (
            -int(bool(row["promoted"])),
            -float(row["positive_minus_negative_score_delta"]),
            -float(row["positive_minus_negative_post_projection_delta"]),
            float(row["mean_absolute_delta_error"]),
            str(row["report"]),
        )
    )
    return {
        "experiment": "steering_telemetry_grid_summary",
        "description": (
            "Compares hidden-state steering telemetry across matched directions, "
            "layers, hook settings, and strengths."
        ),
        "rows": rows,
    }


def write_telemetry_summary(
    summary: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown summary files."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(summary), encoding="utf-8")


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the telemetry grid summary."""

    rows = _sequence(summary.get("rows"))
    lines = [
        "# Steering Telemetry Grid Summary",
        "",
        str(summary.get("description", "")),
        "",
        "| Report | Promotion | Reasons | Model | Layer | Hook | Timing | Position | Strengths | Delta error | Post pos-neg | Post pos-base | Score pos-neg | Score pos-base |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('report', '')}` | "
            f"{row_map.get('promotion_status', 'unknown')} | "
            f"{_join_or_none(row_map.get('promotion_reasons'))} | "
            f"`{row_map.get('model_id', '')}` | "
            f"{int(row_map.get('layer', 0))} | "
            f"{row_map.get('hook_site', '')} | "
            f"{row_map.get('steering_timing', '')} | "
            f"{row_map.get('steering_position', '')} | "
            f"{row_map.get('strengths', [])} | "
            f"{float(row_map.get('mean_absolute_delta_error', 0.0)):.6f} | "
            f"{float(row_map.get('positive_minus_negative_post_projection_delta', 0.0)):+.3f} | "
            f"{float(row_map.get('positive_minus_baseline_post_projection_delta', 0.0)):+.3f} | "
            f"{float(row_map.get('positive_minus_negative_score_delta', 0.0)):+.3f} | "
            f"{float(row_map.get('positive_minus_baseline_score_delta', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation note: a strong causal steering candidate should show "
            "low delta error, positive post-hook projection movement, and "
            "positive text-score movement. Projection movement without behavior "
            "movement localizes the bottleneck downstream of vector injection.",
            "",
        ]
    )
    return "\n".join(lines)


def _row_from_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = _mapping(report.get("summary"))
    traces = _sequence(report.get("traces"))
    first = _mapping(traces[0]) if traces else {}
    return {
        "report": path.name,
        "model_id": first.get("model_id", ""),
        "layer": int(first.get("layer", _layer_from_filename(path))),
        "hook_site": first.get("hook_site", ""),
        "steering_position": first.get("steering_position", ""),
        "steering_timing": first.get("steering_timing", ""),
        "promotion_status": _mapping(report.get("promotion_gate")).get(
            "status",
            "unknown",
        ),
        "promoted": bool(_mapping(report.get("promotion_gate")).get("promoted", False)),
        "promotion_reasons": _mapping(report.get("promotion_gate")).get("reasons", []),
        "strengths": summary.get("strengths", []),
        "mean_absolute_delta_error": float(
            summary.get("mean_absolute_delta_error", 0.0)
        ),
        "positive_minus_negative_post_projection_delta": float(
            summary.get("positive_minus_negative_post_projection_delta", 0.0)
        ),
        "positive_minus_baseline_post_projection_delta": float(
            summary.get("positive_minus_baseline_post_projection_delta", 0.0)
        ),
        "positive_minus_negative_score_delta": float(
            summary.get("positive_minus_negative_score_delta", 0.0)
        ),
        "positive_minus_baseline_score_delta": float(
            summary.get("positive_minus_baseline_score_delta", 0.0)
        ),
    }


def _layer_from_filename(path: Path) -> int:
    match = re.search(r"layer(-?\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/reports/steering_telemetry_grid_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/reports/steering_telemetry_grid_summary.md"),
    )
    return parser.parse_args(argv)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _join_or_none(value: object) -> str:
    items = [str(item) for item in _sequence(value) if str(item)]
    return ", ".join(items) if items else "none"


if __name__ == "__main__":
    raise SystemExit(main())
