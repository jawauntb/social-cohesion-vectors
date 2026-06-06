"""Lexical cue leakage reports for pairwise benchmarks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.baselines import NEGATIVE_WORDS, POSITIVE_WORDS
from social_cohesion_vectors.schemas import PairwiseExample


def run_lexical_leakage_report_from_file(
    pairs_path: str | Path,
    *,
    group_metadata_key: str = "primary_fault_class",
    max_cue_solved_rate: float = 0.10,
) -> dict[str, Any]:
    """Load pairwise examples and summarize lexical cue leakage."""

    return run_lexical_leakage_report(
        pairs=load_pairwise_examples_jsonl(pairs_path),
        group_metadata_key=group_metadata_key,
        input_path=str(pairs_path),
        max_cue_solved_rate=max_cue_solved_rate,
    )


def run_lexical_leakage_report(
    *,
    pairs: Sequence[PairwiseExample],
    group_metadata_key: str = "primary_fault_class",
    input_path: str | None = None,
    max_cue_solved_rate: float = 0.10,
) -> dict[str, Any]:
    """Summarize how often simple cue counts already separate pair labels."""

    pair_rows = [_pair_row(pair, group_metadata_key=group_metadata_key) for pair in pairs]
    solved = sum(1 for row in pair_rows if float(row["cue_margin"]) > 0.0)
    tied = sum(1 for row in pair_rows if float(row["cue_margin"]) == 0.0)
    inverted = sum(1 for row in pair_rows if float(row["cue_margin"]) < 0.0)
    cue_solved_rate = round(solved / len(pairs), 6) if pairs else 0.0
    groups = _group_rows(pair_rows)
    readiness = _activation_readiness(
        pairs=len(pairs),
        cue_solved_rate=cue_solved_rate,
        groups=groups,
        max_cue_solved_rate=max_cue_solved_rate,
    )
    return {
        "experiment": "lexical_leakage_report",
        "description": (
            "Counts simple prosocial and adversarial cue terms in pairwise "
            "positive/negative texts to detect surface leakage before activation "
            "or SAE results are trusted."
        ),
        "inputs": {
            "pairs_path": input_path,
            "pairs": len(pairs),
            "group_metadata_key": group_metadata_key,
        },
        "summary": {
            "pairs": len(pairs),
            "cue_solved_pairs": solved,
            "cue_tied_pairs": tied,
            "cue_inverted_pairs": inverted,
            "cue_solved_rate": cue_solved_rate,
            "mean_cue_margin": _mean(float(row["cue_margin"]) for row in pair_rows),
            "activation_readiness": readiness["status"],
            "ready_for_activation": readiness["ready"],
        },
        "readiness": readiness,
        "groups": groups,
        "pairs": pair_rows,
    }


def save_lexical_leakage_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown lexical leakage reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_lexical_leakage_markdown(report), encoding="utf-8")


def render_lexical_leakage_markdown(report: Mapping[str, Any]) -> str:
    """Render lexical leakage results as markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Lexical Leakage Report",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Cue-solved pairs: {int(summary.get('cue_solved_pairs', 0))}",
        f"- Cue-tied pairs: {int(summary.get('cue_tied_pairs', 0))}",
        f"- Cue-inverted pairs: {int(summary.get('cue_inverted_pairs', 0))}",
        f"- Cue-solved rate: {float(summary.get('cue_solved_rate', 0.0)):.3f}",
        f"- Mean cue margin: {float(summary.get('mean_cue_margin', 0.0)):.3f}",
        f"- Activation readiness: `{summary.get('activation_readiness', 'not_ready')}`",
        f"- Ready for activation: {bool(summary.get('ready_for_activation', False))}",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    readiness = _mapping(report.get("readiness"))
    for gate in _sequence(readiness.get("gates")):
        gate_map = _mapping(gate)
        lines.append(
            "| "
            f"{gate_map.get('gate_id', '')} | "
            f"{float(gate_map.get('value', 0.0)):.3f} | "
            f"{float(gate_map.get('threshold', 0.0)):.3f} | "
            f"{bool(gate_map.get('passed', False))} |"
        )
    failed_groups = _sequence(readiness.get("failed_groups"))
    if failed_groups:
        lines.extend(
            [
                "",
                "Not ready for activation: lexical cues solve one or more "
                f"groups ({', '.join(str(group) for group in failed_groups)}).",
            ]
        )
    lines.extend(
        [
            "",
            "## Groups",
            "",
            "| Group | Pairs | Cue-solved | Cue-solved rate | Mean cue margin |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('group', '')} | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{int(row_map.get('cue_solved_pairs', 0))} | "
            f"{float(row_map.get('cue_solved_rate', 0.0)):.3f} | "
            f"{float(row_map.get('mean_cue_margin', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Leakiest Pairs",
            "",
            "| Pair | Group | Positive cue | Negative cue | Margin |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    pair_rows = sorted(
        (_mapping(row) for row in _sequence(report.get("pairs"))),
        key=lambda row: float(row.get("cue_margin", 0.0)),
        reverse=True,
    )
    for row in pair_rows[:20]:
        lines.append(
            "| "
            f"{row.get('pair_id', '')} | "
            f"{row.get('group', '')} | "
            f"{float(row.get('positive_cue_score', 0.0)):.3f} | "
            f"{float(row.get('negative_cue_score', 0.0)):.3f} | "
            f"{float(row.get('cue_margin', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def lexical_cue_score(text: str) -> float:
    """Return a simple positive-minus-negative lexical cue score."""

    return float(_term_count(text, POSITIVE_WORDS) - _term_count(text, NEGATIVE_WORDS))


def _pair_row(
    pair: PairwiseExample,
    *,
    group_metadata_key: str,
) -> dict[str, Any]:
    positive = lexical_cue_score(pair.positive_text)
    negative = lexical_cue_score(pair.negative_text)
    return {
        "pair_id": pair.pair_id,
        "group": str(pair.metadata.get(group_metadata_key, "ungrouped")),
        "positive_cue_score": positive,
        "negative_cue_score": negative,
        "cue_margin": round(positive - negative, 6),
    }


def _group_rows(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["group"])].append(row)
    rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        solved = sum(1 for row in group_rows if float(row["cue_margin"]) > 0.0)
        rows.append(
            {
                "group": group,
                "pairs": len(group_rows),
                "cue_solved_pairs": solved,
                "cue_solved_rate": round(solved / len(group_rows), 6)
                if group_rows
                else 0.0,
                "mean_cue_margin": _mean(float(row["cue_margin"]) for row in group_rows),
            }
        )
    return rows


def _activation_readiness(
    *,
    pairs: int,
    cue_solved_rate: float,
    groups: Sequence[Mapping[str, Any]],
    max_cue_solved_rate: float,
) -> dict[str, Any]:
    failed_groups = [
        str(row.get("group", ""))
        for row in groups
        if float(row.get("cue_solved_rate", 0.0)) > max_cue_solved_rate
    ]
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(pairs),
            "threshold": 1.0,
            "passed": pairs > 0,
        },
        {
            "gate_id": "overall_cue_solved_rate",
            "value": cue_solved_rate,
            "threshold": max_cue_solved_rate,
            "passed": cue_solved_rate <= max_cue_solved_rate,
        },
        {
            "gate_id": "group_cue_solved_rate",
            "value": max(
                (float(row.get("cue_solved_rate", 0.0)) for row in groups),
                default=0.0,
            ),
            "threshold": max_cue_solved_rate,
            "passed": not failed_groups,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "activation_ready" if ready else "not_ready_for_activation",
        "ready": ready,
        "max_cue_solved_rate": max_cue_solved_rate,
        "failed_groups": failed_groups,
        "gates": gates,
    }


def _term_count(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(term, lowered)) for term in terms)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
