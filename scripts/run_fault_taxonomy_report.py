"""Build pseudo-cohesion fault-taxonomy and SAE grouping reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fault_taxonomy import (
    summarize_pseudo_report_by_fault_class,
    summarize_sae_report_by_fault_class,
    taxonomy_summary,
)
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    default_examples,
    run_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sae_reports = [_load_json(path) for path in args.sae_report]
    report = build_report(sae_reports=sae_reports)
    write_reports(
        report,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(
        "wrote fault taxonomy report: "
        f"{args.output_json} and {args.output_markdown}; "
        f"sae_reports={len(sae_reports)}"
    )
    return 0


def build_report(
    *,
    sae_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a JSON-ready fault taxonomy report."""

    examples = default_examples()
    contrast_ids = sorted({example.contrast_id for example in examples})
    pseudo_report = run_experiment(examples)
    return {
        "experiment": "pseudo_cohesion_fault_taxonomy",
        "description": (
            "Symbolic fault labels, role/asymmetry metadata, and grouped "
            "scorer/SAE outcomes for pseudo-cohesion hard negatives."
        ),
        "taxonomy": taxonomy_summary(contrast_ids),
        "pseudo_cohesion": {
            "summary": pseudo_report["summary"],
            "fault_class_outcomes": summarize_pseudo_report_by_fault_class(
                pseudo_report
            ),
            "scorer_failures": pseudo_report["failure_cases"]["current_scorer"],
            "lexical_failures": pseudo_report["failure_cases"]["lexical_only"],
        },
        "sae_reports": [
            summarize_sae_report_by_fault_class(report) for report in sae_reports
        ],
    }


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise fault taxonomy report as markdown."""

    taxonomy = _mapping(report.get("taxonomy"))
    pseudo = _mapping(report.get("pseudo_cohesion"))
    summary = _mapping(pseudo.get("summary"))
    lines = [
        "# Pseudo-Cohesion Fault Taxonomy",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Annotated contrasts: {int(taxonomy.get('annotated_contrasts', 0))}",
        f"- Requested contrasts: {int(taxonomy.get('requested_contrasts', 0))}",
        f"- Missing contrasts: {len(_sequence(taxonomy.get('missing_contrasts')))}",
        f"- Current scorer failures: {int(summary.get('scorer_failure_count', 0))}",
        f"- Lexical-only failures: {int(summary.get('lexical_failure_count', 0))}",
        "",
        "## Fault Class Coverage",
        "",
        "| Fault class | Contrasts |",
        "| --- | ---: |",
    ]
    for fault_class, count in sorted(
        _mapping(taxonomy.get("fault_class_counts")).items()
    ):
        lines.append(f"| {_cell(str(fault_class))} | {int(count)} |")

    lines.extend(["", "## Guardrail Coverage", "", "| Guardrail | Contrasts |", "| --- | ---: |"])
    for guardrail, count in sorted(
        _mapping(taxonomy.get("guardrail_failure_counts")).items()
    ):
        lines.append(f"| {_cell(str(guardrail))} | {int(count)} |")

    lines.extend(_render_pseudo_outcomes(pseudo))
    lines.extend(_render_sae_reports(_sequence(report.get("sae_reports"))))
    lines.extend(_render_annotations(taxonomy))
    lines.append("")
    return "\n".join(lines)


def _render_pseudo_outcomes(pseudo: Mapping[str, Any]) -> list[str]:
    outcomes = _mapping(pseudo.get("fault_class_outcomes"))
    lines = [
        "",
        "## Scorer Outcomes By Fault",
        "",
        (
            "| Fault class | Contrasts | Scorer prefers genuine | "
            "Mean scorer margin | Lexical prefers genuine | Mean lexical margin |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw_row in _sequence(outcomes.get("rows")):
        row = _mapping(raw_row)
        lines.append(
            "| "
            f"{_cell(str(row.get('fault_class', '')))} | "
            f"{int(row.get('contrasts', 0))} | "
            f"{int(row.get('scorer_prefers_genuine', 0))} | "
            f"{float(row.get('mean_scorer_margin_genuine_minus_pseudo', 0.0)):+.4f} | "
            f"{int(row.get('lexical_prefers_genuine', 0))} | "
            f"{float(row.get('mean_lexical_margin_genuine_minus_pseudo', 0.0)):+.4f} |"
        )
    return lines


def _render_sae_reports(sae_reports: Sequence[object]) -> list[str]:
    lines: list[str] = []
    for index, raw_report in enumerate(sae_reports, start=1):
        report = _mapping(raw_report)
        lines.extend(
            [
                "",
                f"## SAE Fault Grouping {index}",
                "",
                f"- Model: `{report.get('model_id', '')}`",
                f"- SAE release: `{report.get('release', '')}`",
                f"- SAE id: `{report.get('sae_id', '')}`",
                f"- Source prompts: `{report.get('source', '')}`",
            ]
        )
        lines.extend(_render_feature_deltas(report))
        lines.extend(_render_transfer_failures(report))
    return lines


def _render_feature_deltas(report: Mapping[str, Any]) -> list[str]:
    rows = sorted(
        (_mapping(row) for row in _sequence(report.get("feature_fault_deltas"))),
        key=lambda row: abs(float(row.get("mean_delta_genuine_minus_pseudo", 0.0))),
        reverse=True,
    )
    lines = [
        "",
        "### Largest Mean Feature Deltas By Fault",
        "",
        "| Feature | Fault class | Pairs | Genuine-pseudo | Pseudo-genuine |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows[:24]:
        lines.append(
            "| "
            f"{int(row.get('feature', 0))} | "
            f"{_cell(str(row.get('fault_class', '')))} | "
            f"{int(row.get('pairs', 0))} | "
            f"{float(row.get('mean_delta_genuine_minus_pseudo', 0.0)):+.4f} | "
            f"{float(row.get('mean_delta_pseudo_minus_genuine', 0.0)):+.4f} |"
        )
    return lines


def _render_transfer_failures(report: Mapping[str, Any]) -> list[str]:
    rows = sorted(
        (_mapping(row) for row in _sequence(report.get("transfer_failures"))),
        key=lambda row: int(row.get("failures", 0)),
        reverse=True,
    )
    lines = [
        "",
        "### Transfer Failures By Fault",
        "",
        "| Aggregation | Features | Fault class | Failures |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows[:32]:
        lines.append(
            "| "
            f"{_cell(str(row.get('activation_metric', '')))} | "
            f"{_cell(str(row.get('features', '')))} | "
            f"{_cell(str(row.get('fault_class', '')))} | "
            f"{int(row.get('failures', 0))} |"
        )
    return lines


def _render_annotations(taxonomy: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Contrast Annotations",
        "",
        "| Contrast | Fault classes | Guardrails | Pressure target | Cost bearer |",
        "| --- | --- | --- | --- | --- |",
    ]
    for raw_annotation in _sequence(taxonomy.get("annotations")):
        annotation = _mapping(raw_annotation)
        role = _mapping(annotation.get("role_asymmetry"))
        lines.append(
            "| "
            f"{_cell(str(annotation.get('contrast_id', '')))} | "
            f"{_cell(_join(annotation.get('fault_classes')))} | "
            f"{_cell(_join(annotation.get('guardrail_failures')))} | "
            f"{_cell(str(role.get('pressure_target', '')))} | "
            f"{_cell(str(role.get('cost_bearer', '')))} |"
        )
    return lines


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected object JSON report at {path}"
        raise ValueError(msg)
    return payload


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    reports_dir = get_config().paths.reports
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-report",
        type=Path,
        action="append",
        default=[],
        help="Optional SAE token inspection JSON report to group by fault class.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=reports_dir / "pseudo_cohesion_fault_taxonomy.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=reports_dir / "pseudo_cohesion_fault_taxonomy.md",
    )
    return parser.parse_args(argv)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else []


def _join(value: object) -> str:
    items = [str(item) for item in _sequence(value)]
    return ", ".join(items) if items else "n/a"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
