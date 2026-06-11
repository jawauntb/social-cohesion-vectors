"""Summarize constructed bridge preservation across evaluation slices."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATION_KEYS = ("on_source", "on_target", "on_fresh_source", "on_fresh_target")


@dataclass(frozen=True)
class BridgePreservationReportInput:
    """Named fresh bridge diagnostic report path."""

    model_id: str
    path: Path


def run_bridge_preservation_summary_from_files(
    *,
    bridge_reports: Sequence[BridgePreservationReportInput],
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Load bridge diagnostics and summarize constructed-bridge preservation."""

    reports = [
        (item.model_id, _load_json(item.path), str(item.path))
        for item in bridge_reports
    ]
    model_rows = [
        _model_rows(model_id=model_id, report=report, report_path=report_path)
        for model_id, report, report_path in reports
    ]
    evaluation_rows = [
        row
        for model_id, report, _report_path in reports
        for row in _evaluation_rows(model_id=model_id, report=report)
    ]
    failure_rows = [
        row
        for model_id, report, _report_path in reports
        for row in _failure_rows(model_id=model_id, report=report)
    ]
    summary = _summary(
        model_rows=model_rows,
        evaluation_rows=evaluation_rows,
        failure_rows=failure_rows,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "bridge_preservation_summary",
        "description": (
            "Summarizes whether constructed bridge directions preserve "
            "separation across source, target, fresh source, and fresh target "
            "evaluation slices."
        ),
        "inputs": {
            "bridge_reports": [
                {"model_id": item.model_id, "path": str(item.path)}
                for item in bridge_reports
            ],
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
            "evaluation_keys": list(EVALUATION_KEYS),
        },
        "summary": summary,
        "model_rows": model_rows,
        "evaluation_rows": evaluation_rows,
        "failure_rows": failure_rows,
        "interpretation_guardrail": (
            "Bridge preservation summaries are post-hoc text-benchmark "
            "activation diagnostics. They do not establish causal steering, "
            "human behavioral, neural, clinical, deployment, or real-world "
            "social-effect claims."
        ),
    }


def save_bridge_preservation_summary(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write a bridge-preservation report as JSON and Markdown."""

    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_bridge_preservation_summary_markdown(report),
        encoding="utf-8",
    )


def render_bridge_preservation_summary_markdown(report: Mapping[str, Any]) -> str:
    """Render a bridge-preservation summary as Markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Bridge Preservation Summary",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Models: {int(summary.get('models', 0))}",
        f"- Ready for preservation claims: "
        f"{bool(summary.get('ready_for_preservation_claims', False))}",
        f"- Constructed direction rows: "
        f"{int(summary.get('constructed_direction_rows', 0))}",
        f"- Evaluation rows: {int(summary.get('evaluation_rows', 0))}",
        f"- Failing models: {int(summary.get('failing_models', 0))}",
        f"- Failed pair evaluations: "
        f"{int(summary.get('failed_pair_evaluations', 0))}",
        f"- Worst margin: {float(summary.get('worst_margin', 0.0)):+.3f}",
        f"- Worst evaluation: `{summary.get('worst_evaluation', '')}`",
        "",
        "## Model Rows",
        "",
        "| Model | Readiness | Constructed dirs | Worst margin | Failed pairs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("model_rows")):
        item = _mapping(row)
        lines.append(
            "| "
            f"`{item.get('model_id', '')}` | "
            f"`{item.get('readiness', '')}` | "
            f"{int(item.get('constructed_direction_count', 0))} | "
            f"{float(item.get('worst_margin', 0.0)):+.3f} | "
            f"{int(item.get('failed_pair_evaluations', 0))} |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Rows",
            "",
            "| Model | Evaluation | Min accuracy | Min margin | Failed pairs |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("evaluation_rows")):
        item = _mapping(row)
        lines.append(
            "| "
            f"`{item.get('model_id', '')}` | "
            f"`{item.get('evaluation', '')}` | "
            f"{float(item.get('min_accuracy', 0.0)):.3f} | "
            f"{float(item.get('min_margin', 0.0)):+.3f} | "
            f"{int(item.get('failed_pair_evaluations', 0))} |"
        )

    lines.extend(
        [
            "",
            "## Top Failed Pairs",
            "",
            "| Model | Evaluation | Direction | Pair | Margin |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    sorted_failures = sorted(
        (_mapping(row) for row in _sequence(report.get("failure_rows"))),
        key=lambda row: float(row.get("margin", 0.0)),
    )
    for row in sorted_failures[:30]:
        lines.append(
            "| "
            f"`{row.get('model_id', '')}` | "
            f"`{row.get('evaluation', '')}` | "
            f"`{row.get('direction_id', '')}` | "
            f"`{row.get('pair_id', '')}` | "
            f"{float(row.get('margin', 0.0)):+.3f} |"
        )

    lines.extend(
        ["", "## Interpretation Guardrail", "", str(report.get("interpretation_guardrail", ""))]
    )
    return "\n".join(lines) + "\n"


def _model_rows(
    *,
    model_id: str,
    report: Mapping[str, Any],
    report_path: str,
) -> dict[str, Any]:
    evaluations = [
        row for row in _evaluation_rows(model_id=model_id, report=report)
    ]
    failures = [row for row in _failure_rows(model_id=model_id, report=report)]
    margins = [float(row["min_margin"]) for row in evaluations]
    summary = _mapping(report.get("summary"))
    return {
        "model_id": model_id,
        "report_path": report_path,
        "readiness": str(summary.get("readiness", "")),
        "constructed_direction_count": int(
            summary.get("constructed_direction_count", 0)
        ),
        "worst_margin": min(margins, default=0.0),
        "failed_pair_evaluations": len(failures),
    }


def _evaluation_rows(
    *,
    model_id: str,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    constructed_rows = _constructed_direction_rows(report)
    rows: list[dict[str, Any]] = []
    for eval_key in EVALUATION_KEYS:
        evaluations = [
            _mapping(direction.get(eval_key))
            for direction in constructed_rows
            if direction.get(eval_key) is not None
        ]
        accuracies = [float(row.get("pairwise_accuracy", 0.0)) for row in evaluations]
        margins = [float(row.get("min_margin", 0.0)) for row in evaluations]
        rows.append(
            {
                "model_id": model_id,
                "evaluation": eval_key.removeprefix("on_"),
                "direction_count": len(evaluations),
                "min_accuracy": min(accuracies, default=0.0),
                "min_margin": min(margins, default=0.0),
                "failed_pair_evaluations": sum(
                    int(row.get("failed_pair_count", 0)) for row in evaluations
                ),
            }
        )
    return rows


def _failure_rows(
    *,
    model_id: str,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in _constructed_direction_rows(report):
        direction_id = str(direction.get("direction_id", ""))
        bridge_family = direction_id.split(":", 1)[0] if direction_id else ""
        for eval_key in EVALUATION_KEYS:
            evaluation = eval_key.removeprefix("on_")
            eval_row = _mapping(direction.get(eval_key))
            for failure in _sequence(eval_row.get("failed_pairs")):
                failure_row = _mapping(failure)
                rows.append(
                    {
                        "model_id": model_id,
                        "direction_id": direction_id,
                        "bridge_family": bridge_family,
                        "evaluation": evaluation,
                        "pair_id": str(failure_row.get("pair_id", "")),
                        "margin": float(failure_row.get("margin", 0.0)),
                    }
                )
    return rows


def _summary(
    *,
    model_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    failed_models = {
        str(row.get("model_id", "")) for row in failure_rows if row.get("model_id")
    }
    margins = [float(row.get("min_margin", 0.0)) for row in evaluation_rows]
    worst_margin = min(margins, default=0.0)
    worst_evaluation = _worst_evaluation(evaluation_rows)
    accuracy_ready = all(
        float(row.get("min_accuracy", 0.0)) >= min_pairwise_accuracy
        for row in evaluation_rows
    )
    margin_ready = all(
        float(row.get("min_margin", 0.0)) > min_margin for row in evaluation_rows
    )
    failure_ready = not failure_rows
    return {
        "models": len(model_rows),
        "constructed_direction_rows": sum(
            int(row.get("constructed_direction_count", 0)) for row in model_rows
        ),
        "evaluation_rows": len(evaluation_rows),
        "failing_models": len(failed_models),
        "failed_pair_evaluations": len(failure_rows),
        "worst_margin": worst_margin,
        "worst_evaluation": worst_evaluation,
        "ready_for_preservation_claims": (
            bool(model_rows) and accuracy_ready and margin_ready and failure_ready
        ),
        "failures_by_evaluation": dict(
            sorted(
                Counter(str(row.get("evaluation", "")) for row in failure_rows).items()
            )
        ),
    }


def _constructed_direction_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(row)
        for row in _sequence(report.get("direction_evaluations"))
        if _mapping(row).get("direction_family") == "constructed_bridge"
    ]


def _worst_evaluation(evaluation_rows: Sequence[Mapping[str, Any]]) -> str:
    if not evaluation_rows:
        return ""
    row = min(evaluation_rows, key=lambda item: float(item.get("min_margin", 0.0)))
    return str(row.get("evaluation", ""))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
