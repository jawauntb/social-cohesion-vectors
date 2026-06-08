"""Future-option coverage audit for slack-preservation benchmarks."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.fault_generation import FUTURE_OPTION_ORDER
from social_cohesion_vectors.schemas import PairwiseExample


def run_slack_preservation_audit_from_file(
    pairs_path: str | Path,
    *,
    option_metadata_key: str = "slack_options_tested",
    group_metadata_key: str = "primary_fault_class",
    min_pairs_per_option: int = 1,
    min_slack_margin: float = 0.0,
) -> dict[str, Any]:
    """Load pairwise examples and audit future-option slack coverage."""

    return run_slack_preservation_audit(
        pairs=load_pairwise_examples_jsonl(pairs_path),
        option_metadata_key=option_metadata_key,
        group_metadata_key=group_metadata_key,
        min_pairs_per_option=min_pairs_per_option,
        min_slack_margin=min_slack_margin,
        input_path=str(pairs_path),
    )


def run_slack_preservation_audit(
    *,
    pairs: Sequence[PairwiseExample],
    option_metadata_key: str = "slack_options_tested",
    group_metadata_key: str = "primary_fault_class",
    min_pairs_per_option: int = 1,
    min_slack_margin: float = 0.0,
    input_path: str | None = None,
) -> dict[str, Any]:
    """Audit whether generated pairs preserve future-option coverage."""

    pair_rows = [
        _pair_row(
            pair,
            option_metadata_key=option_metadata_key,
            group_metadata_key=group_metadata_key,
            min_slack_margin=min_slack_margin,
        )
        for pair in pairs
    ]
    option_rows = _option_rows(pair_rows)
    required_options = list(FUTURE_OPTION_ORDER)
    readiness = _readiness(
        pairs=len(pairs),
        pair_rows=pair_rows,
        option_rows=option_rows,
        required_options=required_options,
        min_pairs_per_option=min_pairs_per_option,
        min_slack_margin=min_slack_margin,
    )
    margins = [float(row["slack_margin"]) for row in pair_rows]
    return {
        "experiment": "slack_preservation_audit",
        "description": (
            "Audits whether pairwise generated benchmarks explicitly cover "
            "future-option paths and whether genuine examples preserve more "
            "slack than pseudo-cohesion examples."
        ),
        "inputs": {
            "pairs_path": input_path,
            "pairs": len(pairs),
            "option_metadata_key": option_metadata_key,
            "group_metadata_key": group_metadata_key,
            "required_options": required_options,
            "min_pairs_per_option": min_pairs_per_option,
            "min_slack_margin": min_slack_margin,
        },
        "summary": {
            "pairs": len(pairs),
            "pairs_with_options": sum(bool(row["options"]) for row in pair_rows),
            "missing_option_pairs": sum(not row["options"] for row in pair_rows),
            "missing_slack_margin_pairs": sum(
                not row["has_slack_margin"] for row in pair_rows
            ),
            "options_covered": sum(
                int(row["pairs"]) > 0
                for row in option_rows
                if row["option"] in required_options
            ),
            "required_options": len(required_options),
            "slack_prefers_genuine": sum(
                bool(row["slack_prefers_genuine"]) for row in pair_rows
            ),
            "slack_pairwise_accuracy": _ratio(
                sum(bool(row["slack_prefers_genuine"]) for row in pair_rows),
                len(pair_rows),
            ),
            "mean_slack_preservation_margin": _mean(margins),
            "min_slack_preservation_margin": round(min(margins), 6) if margins else 0.0,
            "activation_readiness": readiness["status"],
            "ready_for_activation": readiness["ready"],
        },
        "readiness": readiness,
        "options": option_rows,
        "pairs": pair_rows,
    }


def save_slack_preservation_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown slack-preservation audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_slack_preservation_markdown(report),
        encoding="utf-8",
    )


def render_slack_preservation_markdown(report: Mapping[str, Any]) -> str:
    """Render a slack-preservation coverage audit as markdown."""

    summary = _mapping(report.get("summary"))
    readiness = _mapping(report.get("readiness"))
    lines = [
        "# Slack Preservation Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Pairs with future-option metadata: "
        f"{int(summary.get('pairs_with_options', 0))}",
        f"- Required options covered: "
        f"{int(summary.get('options_covered', 0))}/"
        f"{int(summary.get('required_options', 0))}",
        f"- Slack prefers genuine: {int(summary.get('slack_prefers_genuine', 0))}",
        f"- Slack pairwise accuracy: "
        f"{float(summary.get('slack_pairwise_accuracy', 0.0)):.3f}",
        f"- Mean slack-preservation margin: "
        f"{float(summary.get('mean_slack_preservation_margin', 0.0)):+.3f}",
        f"- Min slack-preservation margin: "
        f"{float(summary.get('min_slack_preservation_margin', 0.0)):+.3f}",
        f"- Activation readiness: `{summary.get('activation_readiness', 'not_ready')}`",
        f"- Ready for activation: {bool(summary.get('ready_for_activation', False))}",
        "",
        "## Readiness Gates",
        "",
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
    failed_options = _sequence(readiness.get("failed_options"))
    if failed_options:
        lines.extend(
            [
                "",
                "Not ready for activation: future-option coverage or slack "
                f"margins fail for {', '.join(str(item) for item in failed_options)}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Future Options",
            "",
            "| Future option | Pairs | Slack prefers genuine | Mean margin | Min margin | Fault classes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _sequence(report.get("options")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('option', '')}` | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{int(row_map.get('slack_prefers_genuine', 0))} | "
            f"{float(row_map.get('mean_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('min_margin', 0.0)):+.3f} | "
            f"{', '.join(f'`{value}`' for value in _sequence(row_map.get('groups')))} |"
        )
    return "\n".join(lines) + "\n"


def _pair_row(
    pair: PairwiseExample,
    *,
    option_metadata_key: str,
    group_metadata_key: str,
    min_slack_margin: float,
) -> dict[str, Any]:
    options = _metadata_values(pair.metadata.get(option_metadata_key))
    positive_slack = _float_metadata(pair.metadata.get("positive_slack_preservation"))
    negative_slack = _float_metadata(pair.metadata.get("negative_slack_preservation"))
    has_slack_margin = _has_slack_margin_metadata(pair)
    slack_margin = _slack_margin(pair, positive_slack, negative_slack)
    return {
        "pair_id": pair.pair_id,
        "group": str(pair.metadata.get(group_metadata_key, "ungrouped")),
        "options": options,
        "has_slack_margin": has_slack_margin,
        "positive_slack_preservation": positive_slack,
        "negative_slack_preservation": negative_slack,
        "slack_margin": slack_margin,
        "slack_prefers_genuine": slack_margin > min_slack_margin,
    }


def _option_rows(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        for option in _sequence(row.get("options")):
            grouped[str(option)].append(row)
    ordered_options = [*FUTURE_OPTION_ORDER]
    ordered_options.extend(
        option for option in sorted(grouped) if option not in FUTURE_OPTION_ORDER
    )
    rows: list[dict[str, Any]] = []
    for option in ordered_options:
        option_rows = grouped.get(option, [])
        margins = [float(row.get("slack_margin", 0.0)) for row in option_rows]
        groups = sorted({str(row.get("group", "")) for row in option_rows})
        rows.append(
            {
                "option": option,
                "pairs": len(option_rows),
                "slack_prefers_genuine": sum(
                    bool(row.get("slack_prefers_genuine", False)) for row in option_rows
                ),
                "mean_margin": _mean(margins),
                "min_margin": round(min(margins), 6) if margins else 0.0,
                "groups": groups,
            }
        )
    return rows


def _readiness(
    *,
    pairs: int,
    pair_rows: Sequence[Mapping[str, Any]],
    option_rows: Sequence[Mapping[str, Any]],
    required_options: Sequence[str],
    min_pairs_per_option: int,
    min_slack_margin: float,
) -> dict[str, Any]:
    option_by_id = {str(row.get("option", "")): row for row in option_rows}
    missing_options = [
        option
        for option in required_options
        if int(_mapping(option_by_id.get(option)).get("pairs", 0)) == 0
    ]
    thin_options = [
        option
        for option in required_options
        if int(_mapping(option_by_id.get(option)).get("pairs", 0))
        < min_pairs_per_option
    ]
    low_margin_options = [
        option
        for option in required_options
        if float(_mapping(option_by_id.get(option)).get("min_margin", 0.0))
        <= min_slack_margin
    ]
    missing_option_pairs = sum(not row.get("options") for row in pair_rows)
    missing_slack_margin_pairs = sum(
        not row.get("has_slack_margin", False) for row in pair_rows
    )
    failed_options = sorted(
        set(missing_options) | set(thin_options) | set(low_margin_options)
    )
    min_option_pairs = min(
        (
            int(_mapping(option_by_id.get(option)).get("pairs", 0))
            for option in required_options
        ),
        default=0,
    )
    min_option_margin = min(
        (
            float(_mapping(option_by_id.get(option)).get("min_margin", 0.0))
            for option in required_options
        ),
        default=0.0,
    )
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(pairs),
            "threshold": 1.0,
            "passed": pairs > 0,
        },
        {
            "gate_id": "complete_future_option_metadata",
            "value": float(pairs - missing_option_pairs),
            "threshold": float(pairs),
            "passed": missing_option_pairs == 0,
        },
        {
            "gate_id": "complete_slack_margin_metadata",
            "value": float(pairs - missing_slack_margin_pairs),
            "threshold": float(pairs),
            "passed": missing_slack_margin_pairs == 0,
        },
        {
            "gate_id": "required_future_option_coverage",
            "value": float(len(required_options) - len(missing_options)),
            "threshold": float(len(required_options)),
            "passed": not missing_options,
        },
        {
            "gate_id": "min_pairs_per_future_option",
            "value": float(min_option_pairs),
            "threshold": float(min_pairs_per_option),
            "passed": not thin_options,
        },
        {
            "gate_id": "positive_slack_margin_per_future_option",
            "value": min_option_margin,
            "threshold": min_slack_margin,
            "passed": not low_margin_options,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "slack_preservation_ready"
        if ready
        else "not_ready_for_slack_preservation_claims",
        "ready": ready,
        "missing_option_pairs": missing_option_pairs,
        "missing_slack_margin_pairs": missing_slack_margin_pairs,
        "missing_options": missing_options,
        "thin_options": thin_options,
        "low_margin_options": low_margin_options,
        "failed_options": failed_options,
        "gates": gates,
    }


def _slack_margin(
    pair: PairwiseExample,
    positive_slack: float,
    negative_slack: float,
) -> float:
    raw_margin = pair.metadata.get("slack_preservation_margin")
    if raw_margin is not None:
        return _float_metadata(raw_margin)
    if positive_slack != 0.0 or negative_slack != 0.0:
        return round(positive_slack - negative_slack, 6)
    return round(pair.positive_score - pair.negative_score, 6)


def _has_slack_margin_metadata(pair: PairwiseExample) -> bool:
    if pair.metadata.get("slack_preservation_margin") is not None:
        return True
    return (
        pair.metadata.get("positive_slack_preservation") is not None
        and pair.metadata.get("negative_slack_preservation") is not None
    )


def _metadata_values(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    values = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    return list(dict.fromkeys(values))


def _float_metadata(raw_value: object) -> float:
    if raw_value is None:
        return 0.0
    try:
        return float(str(raw_value))
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
