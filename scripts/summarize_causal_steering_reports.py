"""Summarize causal steering reports into one sortable table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [_row_from_report(path) for path in sorted(args.reports)]
    rows.sort(
        key=lambda row: (
            -int(bool(row["promoted"])),
            -float(row["positive_vs_negative_success_rate"]),
            -float(row["positive_minus_negative_mean_score_delta"]),
            str(row["report"]),
        )
    )
    payload = {"experiment": "causal_steering_sweep_summary", "rows": rows}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"wrote {args.markdown_output} rows={len(rows)}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/reports/causal_steering_sweep_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/reports/causal_steering_sweep_summary.md"),
    )
    return parser.parse_args(argv)


def _row_from_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["summary"]
    records = report.get("records", [])
    first = records[0] if records else {}
    return {
        "report": path.name,
        "model_id": first.get("model_id", ""),
        "layer": first.get("layer", ""),
        "hook_site": first.get("hook_site", ""),
        "steering_position": first.get("steering_position", ""),
        "steering_timing": first.get("steering_timing", ""),
        "promotion_status": _mapping(report.get("promotion_gate")).get(
            "status",
            "unknown",
        ),
        "promoted": bool(_mapping(report.get("promotion_gate")).get("promoted", False)),
        "promotion_reasons": _mapping(report.get("promotion_gate")).get("reasons", []),
        "strengths": summary["strengths"],
        "positive_vs_negative_success_rate": summary[
            "positive_vs_negative_success_rate"
        ],
        "autonomy_positive_vs_negative_success_rate": summary[
            "autonomy_positive_vs_negative_success_rate"
        ],
        "positive_minus_baseline_mean_score_delta": summary[
            "positive_minus_baseline_mean_score_delta"
        ],
        "positive_minus_negative_mean_score_delta": summary[
            "positive_minus_negative_mean_score_delta"
        ],
    }


def _render_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Causal Steering Sweep Summary",
        "",
        "| Report | Promotion | Reasons | Hook | Timing | Position | Strengths | Cohesion win | Autonomy win | Pos-baseline | Pos-neg |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['report']} | "
            f"{row['promotion_status']} | "
            f"{_join_or_none(row['promotion_reasons'])} | "
            f"{row['hook_site']} | "
            f"{row['steering_timing']} | "
            f"{row['steering_position']} | "
            f"{row['strengths']} | "
            f"{float(row['positive_vs_negative_success_rate']):.3f} | "
            f"{float(row['autonomy_positive_vs_negative_success_rate']):.3f} | "
            f"{float(row['positive_minus_baseline_mean_score_delta']):+.3f} | "
            f"{float(row['positive_minus_negative_mean_score_delta']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation note: these are local-rubric smoke results on held-out "
            "decision prompts, not human validation. A strong steering claim needs "
            "monotonic behavioral shifts plus anti-compliance controls.",
            "",
        ]
    )
    return "\n".join(lines)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _join_or_none(value: object) -> str:
    if not isinstance(value, list | tuple):
        return "none"
    items = [str(item) for item in value if str(item)]
    return ", ".join(items) if items else "none"


if __name__ == "__main__":
    raise SystemExit(main())
