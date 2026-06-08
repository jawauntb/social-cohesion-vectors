"""Candidate tournaments for generated fault-class authorship batches."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.component_audit import (
    run_component_margin_audit,
)
from social_cohesion_vectors.experiments.fault_generation import (
    FaultPromptRecord,
    fault_examples_from_prompt_outputs,
    pairwise_examples_from_generated_fault_examples,
    scored_runs_from_generated_fault_examples,
)
from social_cohesion_vectors.experiments.lexical_leakage import (
    run_lexical_leakage_report,
)


@dataclass(frozen=True)
class CandidateOutputSet:
    """A named batch of candidate outputs for the same prompt records."""

    candidate_id: str
    output_records: Sequence[Mapping[str, Any]]
    source_path: str | None = None


@dataclass(frozen=True)
class FaultAuthorshipTournamentResult:
    """Tournament report plus selected raw output rows."""

    report: dict[str, Any]
    selected_output_records: list[dict[str, Any]]


def run_fault_authorship_tournament(
    *,
    records: Sequence[FaultPromptRecord],
    candidates: Sequence[CandidateOutputSet],
    provider: str,
    model: str,
    target_word_count_min: int = 55,
    target_word_count_max: int = 75,
) -> FaultAuthorshipTournamentResult:
    """Evaluate candidate generations and select one candidate per prompt pair."""

    record_index = {record.prompt_id: record for record in records}
    candidate_evaluations = [
        _evaluate_candidate(
            records=records,
            candidate=candidate,
            provider=provider,
            model=model,
            target_word_count_min=target_word_count_min,
            target_word_count_max=target_word_count_max,
        )
        for candidate in candidates
    ]
    rows_by_base_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in candidate_evaluations:
        for row in evaluation["rows"]:
            rows_by_base_key[str(row["base_pair_key"])].append(row)

    expected_pair_keys = _expected_pair_keys(records)
    selected_rows = [
        _select_candidate_row(rows_by_base_key[base_key])
        for base_key in expected_pair_keys
        if rows_by_base_key.get(base_key)
    ]
    selected_rows = [row for row in selected_rows if row is not None]
    selected_output_records = _selected_output_records(
        selected_rows=selected_rows,
        candidates=candidates,
        record_index=record_index,
    )
    selection_summary = _selection_summary(
        selected_rows=selected_rows,
        expected_pair_count=len(expected_pair_keys),
    )
    report = {
        "experiment": "fault_authorship_candidate_tournament",
        "description": (
            "Selects generated pseudo/genuine hard-negative pairs from multiple "
            "authorship candidate batches using scorer, slack, lexical leakage, "
            "length, and formatting gates."
        ),
        "inputs": {
            "provider": provider,
            "model": model,
            "prompt_records": len(records),
            "expected_pairs": len(expected_pair_keys),
            "candidate_sets": len(candidates),
            "target_word_count_min": target_word_count_min,
            "target_word_count_max": target_word_count_max,
        },
        "summary": selection_summary,
        "candidate_summaries": [
            evaluation["summary"] for evaluation in candidate_evaluations
        ],
        "selected_pairs": selected_rows,
        "candidate_pairs": [
            row for evaluation in candidate_evaluations for row in evaluation["rows"]
        ],
        "missing_pair_keys": [
            base_key
            for base_key in expected_pair_keys
            if base_key not in {str(row["base_pair_key"]) for row in selected_rows}
        ],
    }
    return FaultAuthorshipTournamentResult(
        report=report,
        selected_output_records=selected_output_records,
    )


def save_fault_authorship_tournament_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown tournament reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fault_authorship_tournament_markdown(report),
        encoding="utf-8",
    )


def render_fault_authorship_tournament_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise fault authorship tournament report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Fault Authorship Candidate Tournament",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Expected pairs: {int(summary.get('expected_pairs', 0))}",
        f"- Selected pairs: {int(summary.get('selected_pairs', 0))}",
        f"- Missing pairs: {int(summary.get('missing_pairs', 0))}",
        f"- All required gates passed: "
        f"{int(summary.get('all_required_gates_passed', 0))}",
        f"- Score gate pass rate: {float(summary.get('score_gate_rate', 0.0)):.3f}",
        f"- Slack gate pass rate: {float(summary.get('slack_gate_rate', 0.0)):.3f}",
        f"- Lexical gate pass rate: "
        f"{float(summary.get('lexical_gate_rate', 0.0)):.3f}",
        f"- Core gate triad passed: "
        f"{int(summary.get('core_required_gates_passed', 0))}",
        f"- Mean selected score margin: "
        f"{float(summary.get('mean_score_margin', 0.0)):+.3f}",
        f"- Mean selected slack margin: "
        f"{float(summary.get('mean_slack_preservation_margin', 0.0)):+.3f}",
        f"- Mean selected cue margin: "
        f"{float(summary.get('mean_cue_margin', 0.0)):+.3f}",
        f"- Tournament status: `{summary.get('status', 'unknown')}`",
        "",
        "## Candidate Sets",
        "",
        "| Candidate | Pairs | Score pass | Slack pass | Lexical pass | Core gates | "
        "All gates | "
        "Mean score margin | Mean slack margin | Mean cue margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence_of_mappings(report.get("candidate_summaries")):
        lines.append(
            "| "
            f"{row.get('candidate_id', '')} | "
            f"{int(row.get('pairs', 0))} | "
            f"{int(row.get('score_gate_passed', 0))} | "
            f"{int(row.get('slack_gate_passed', 0))} | "
            f"{int(row.get('lexical_gate_passed', 0))} | "
            f"{int(row.get('core_required_gates_passed', 0))} | "
            f"{int(row.get('all_required_gates_passed', 0))} | "
            f"{float(row.get('mean_score_margin', 0.0)):+.3f} | "
            f"{float(row.get('mean_slack_preservation_margin', 0.0)):+.3f} | "
            f"{float(row.get('mean_cue_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Pairs",
            "",
            "| Pair | Fault | Winner | Score margin | Slack margin | Cue margin | Gates |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _sequence_of_mappings(report.get("selected_pairs")):
        lines.append(
            "| "
            f"`{row.get('base_pair_key', '')}` | "
            f"{row.get('primary_fault_class', '')} | "
            f"{row.get('candidate_id', '')} | "
            f"{float(row.get('score_margin', 0.0)):+.3f} | "
            f"{float(row.get('slack_preservation_margin', 0.0)):+.3f} | "
            f"{float(row.get('cue_margin', 0.0)):+.3f} | "
            f"{int(row.get('gate_pass_count', 0))}/5 |"
        )
    missing = _sequence(report.get("missing_pair_keys"))
    if missing:
        lines.extend(["", "## Missing Pair Keys", ""])
        lines.extend(f"- `{base_key}`" for base_key in missing)
    return "\n".join(lines) + "\n"


def _evaluate_candidate(
    *,
    records: Sequence[FaultPromptRecord],
    candidate: CandidateOutputSet,
    provider: str,
    model: str,
    target_word_count_min: int,
    target_word_count_max: int,
) -> dict[str, Any]:
    outputs = _valid_outputs_by_prompt_id(candidate.output_records)
    examples = fault_examples_from_prompt_outputs(
        records,
        outputs,
        provider=provider,
        model=model,
    )
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        source=f"generated_fault_class_{provider}_{candidate.candidate_id}",
        provider=provider,
    )
    component_report = run_component_margin_audit(
        scored_runs=scored_runs,
        pairs=pairs,
    )
    lexical_report = run_lexical_leakage_report(pairs=pairs)
    component_by_pair = {
        str(row.get("pair_id", "")): row for row in component_report["pairs"]
    }
    lexical_by_pair = {
        str(row.get("pair_id", "")): row for row in lexical_report["pairs"]
    }
    rows = [
        _candidate_pair_row(
            pair,
            candidate_id=candidate.candidate_id,
            provider=provider,
            component_row=_mapping(component_by_pair.get(pair.pair_id)),
            lexical_row=_mapping(lexical_by_pair.get(pair.pair_id)),
            target_word_count_min=target_word_count_min,
            target_word_count_max=target_word_count_max,
        )
        for pair in pairs
    ]
    return {
        "candidate": candidate,
        "rows": rows,
        "summary": _candidate_summary(
            candidate=candidate,
            rows=rows,
            raw_outputs=len(candidate.output_records),
            valid_outputs=len(outputs),
        ),
    }


def _candidate_pair_row(
    pair: Any,
    *,
    candidate_id: str,
    provider: str,
    component_row: Mapping[str, Any],
    lexical_row: Mapping[str, Any],
    target_word_count_min: int,
    target_word_count_max: int,
) -> dict[str, Any]:
    base_contrast = str(pair.metadata.get("base_contrast_id", pair.scenario_id))
    variant = _stable_variant(
        str(pair.metadata.get("generated_variant", "unknown")),
        provider=provider,
    )
    positive_words = _word_count(pair.positive_text)
    negative_words = _word_count(pair.negative_text)
    score_margin = float(component_row.get("score_margin", 0.0))
    slack_margin = float(component_row.get("slack_preservation_margin", 0.0))
    cue_margin = float(lexical_row.get("cue_margin", 0.0))
    format_issue_count = _format_issue_count(pair.positive_text) + _format_issue_count(
        pair.negative_text
    )
    length_fit = _word_count_in_range(
        positive_words,
        target_word_count_min=target_word_count_min,
        target_word_count_max=target_word_count_max,
    ) and _word_count_in_range(
        negative_words,
        target_word_count_min=target_word_count_min,
        target_word_count_max=target_word_count_max,
    )
    format_gate = format_issue_count == 0
    score_gate = score_margin > 0.0
    slack_gate = slack_margin > 0.0
    lexical_gate = cue_margin <= 0.0
    gate_passes = {
        "score_prefers_genuine": score_gate,
        "slack_prefers_genuine": slack_gate,
        "lexical_not_solved_by_genuine_cues": lexical_gate,
        "length_in_target_range": length_fit,
        "formatting_clean": format_gate,
    }
    complexity_cost = (
        _word_count_distance(positive_words, target_word_count_min, target_word_count_max)
        + _word_count_distance(
            negative_words,
            target_word_count_min,
            target_word_count_max,
        )
        + (10 * format_issue_count)
    )
    selection_score = (
        (100.0 if score_gate else 0.0)
        + (100.0 if slack_gate else 0.0)
        + (50.0 if lexical_gate else 0.0)
        + (20.0 if length_fit else 0.0)
        + (5.0 if format_gate else 0.0)
        + score_margin
        + slack_margin
        - (0.25 * max(cue_margin, 0.0))
        - (0.01 * complexity_cost)
    )
    return {
        "candidate_id": candidate_id,
        "pair_id": pair.pair_id,
        "base_pair_key": f"{base_contrast}__{variant}",
        "base_contrast_id": base_contrast,
        "variant": variant,
        "primary_fault_class": str(pair.metadata.get("primary_fault_class", "")),
        "positive_prompt_id": (
            f"{base_contrast}__{variant}__genuine_cohesion"
        ),
        "negative_prompt_id": (
            f"{base_contrast}__{variant}__pseudo_cohesion"
        ),
        "positive_run_id": pair.positive_run_id,
        "negative_run_id": pair.negative_run_id,
        "positive_words": positive_words,
        "negative_words": negative_words,
        "score_margin": round(score_margin, 6),
        "slack_preservation_margin": round(slack_margin, 6),
        "positive_cue_score": float(lexical_row.get("positive_cue_score", 0.0)),
        "negative_cue_score": float(lexical_row.get("negative_cue_score", 0.0)),
        "cue_margin": round(cue_margin, 6),
        "format_issue_count": format_issue_count,
        "complexity_cost": round(complexity_cost, 6),
        "gate_passes": gate_passes,
        "gate_pass_count": sum(1 for passed in gate_passes.values() if passed),
        "core_required_gates_pass": score_gate and slack_gate and lexical_gate,
        "all_required_gates_pass": all(gate_passes.values()),
        "selection_score": round(selection_score, 6),
        "selection_tuple": _selection_tuple(
            score_gate=score_gate,
            slack_gate=slack_gate,
            lexical_gate=lexical_gate,
            length_fit=length_fit,
            format_gate=format_gate,
            slack_margin=slack_margin,
            score_margin=score_margin,
            cue_margin=cue_margin,
            complexity_cost=complexity_cost,
        ),
    }


def _selection_tuple(
    *,
    score_gate: bool,
    slack_gate: bool,
    lexical_gate: bool,
    length_fit: bool,
    format_gate: bool,
    slack_margin: float,
    score_margin: float,
    cue_margin: float,
    complexity_cost: float,
) -> list[float | int]:
    return [
        int(score_gate and slack_gate and lexical_gate and length_fit and format_gate),
        int(score_gate and slack_gate and lexical_gate),
        int(score_gate and slack_gate),
        int(score_gate),
        int(lexical_gate),
        int(length_fit),
        int(format_gate),
        round(slack_margin, 6),
        round(score_margin, 6),
        round(-max(cue_margin, 0.0), 6),
        round(-complexity_cost, 6),
    ]


def _select_candidate_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            tuple(row.get("selection_tuple", [])),
            float(row.get("selection_score", 0.0)),
            str(row.get("candidate_id", "")),
        ),
    )


def _selected_output_records(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateOutputSet],
    record_index: Mapping[str, FaultPromptRecord],
) -> list[dict[str, Any]]:
    raw_by_candidate = {
        candidate.candidate_id: {
            str(row.get("prompt_id", "")): row for row in candidate.output_records
        }
        for candidate in candidates
    }
    output_records: list[dict[str, Any]] = []
    for row in selected_rows:
        candidate_id = str(row["candidate_id"])
        for prompt_key, label in (
            ("negative_prompt_id", "pseudo_cohesion"),
            ("positive_prompt_id", "genuine_cohesion"),
        ):
            prompt_id = str(row[prompt_key])
            record = record_index[prompt_id]
            raw_output = raw_by_candidate[candidate_id][prompt_id]
            output_records.append(
                {
                    "prompt_id": prompt_id,
                    "base_contrast_id": record.base_contrast_id,
                    "variant": record.variant,
                    "label": label,
                    "primary_fault_class": record.primary_fault_class,
                    "prompt_contract_version": str(
                        record.metadata.get("prompt_contract_version", "")
                    ),
                    "target_word_count_min": record.metadata.get(
                        "target_word_count_min",
                        "",
                    ),
                    "target_word_count_max": record.metadata.get(
                        "target_word_count_max",
                        "",
                    ),
                    "future_options_tested": str(
                        record.metadata.get("future_options_tested", "")
                    ),
                    "future_option_contract": str(
                        record.metadata.get("future_option_contract", "")
                    ),
                    "provider": str(raw_output.get("provider", "")),
                    "model": str(raw_output.get("model", "")),
                    "status": str(raw_output.get("status", "ok")),
                    "valid": bool(raw_output.get("valid", True)),
                    "error_type": str(raw_output.get("error_type", "")),
                    "error_detail": str(raw_output.get("error_detail", "")),
                    "text": str(raw_output.get("text", "")).strip(),
                    "text_length": len(str(raw_output.get("text", "")).strip()),
                    "selected_from_candidate": candidate_id,
                    "tournament_base_pair_key": str(row["base_pair_key"]),
                    "selection_score": float(row.get("selection_score", 0.0)),
                    "selection_gate_pass_count": int(row.get("gate_pass_count", 0)),
                    "selection_all_required_gates_pass": bool(
                        row.get("all_required_gates_pass", False)
                    ),
                }
            )
    return output_records


def _candidate_summary(
    *,
    candidate: CandidateOutputSet,
    rows: Sequence[Mapping[str, Any]],
    raw_outputs: int,
    valid_outputs: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_path": candidate.source_path,
        "raw_outputs": raw_outputs,
        "valid_outputs": valid_outputs,
        "pairs": len(rows),
        "score_gate_passed": sum(
            1
            for row in rows
            if bool(_mapping(row.get("gate_passes")).get("score_prefers_genuine"))
        ),
        "slack_gate_passed": sum(
            1
            for row in rows
            if bool(_mapping(row.get("gate_passes")).get("slack_prefers_genuine"))
        ),
        "lexical_gate_passed": sum(
            1
            for row in rows
            if bool(
                _mapping(row.get("gate_passes")).get(
                    "lexical_not_solved_by_genuine_cues"
                )
            )
        ),
        "length_gate_passed": sum(
            1
            for row in rows
            if bool(_mapping(row.get("gate_passes")).get("length_in_target_range"))
        ),
        "format_gate_passed": sum(
            1
            for row in rows
            if bool(_mapping(row.get("gate_passes")).get("formatting_clean"))
        ),
        "all_required_gates_passed": sum(
            1 for row in rows if bool(row.get("all_required_gates_pass"))
        ),
        "core_required_gates_passed": sum(
            1 for row in rows if bool(row.get("core_required_gates_pass"))
        ),
        "mean_score_margin": _mean(float(row.get("score_margin", 0.0)) for row in rows),
        "mean_slack_preservation_margin": _mean(
            float(row.get("slack_preservation_margin", 0.0)) for row in rows
        ),
        "mean_cue_margin": _mean(float(row.get("cue_margin", 0.0)) for row in rows),
    }


def _selection_summary(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    expected_pair_count: int,
) -> dict[str, Any]:
    selected_count = len(selected_rows)
    score_passed = _gate_count(selected_rows, "score_prefers_genuine")
    slack_passed = _gate_count(selected_rows, "slack_prefers_genuine")
    lexical_passed = _gate_count(selected_rows, "lexical_not_solved_by_genuine_cues")
    all_gates_passed = sum(
        1 for row in selected_rows if bool(row.get("all_required_gates_pass"))
    )
    core_gates_passed = sum(
        1 for row in selected_rows if bool(row.get("core_required_gates_pass"))
    )
    return {
        "expected_pairs": expected_pair_count,
        "selected_pairs": selected_count,
        "missing_pairs": max(expected_pair_count - selected_count, 0),
        "all_required_gates_passed": all_gates_passed,
        "core_required_gates_passed": core_gates_passed,
        "score_gate_passed": score_passed,
        "slack_gate_passed": slack_passed,
        "lexical_gate_passed": lexical_passed,
        "score_gate_rate": _rate(score_passed, selected_count),
        "slack_gate_rate": _rate(slack_passed, selected_count),
        "lexical_gate_rate": _rate(lexical_passed, selected_count),
        "all_required_gate_rate": _rate(all_gates_passed, selected_count),
        "core_required_gate_rate": _rate(core_gates_passed, selected_count),
        "mean_score_margin": _mean(
            float(row.get("score_margin", 0.0)) for row in selected_rows
        ),
        "mean_slack_preservation_margin": _mean(
            float(row.get("slack_preservation_margin", 0.0))
            for row in selected_rows
        ),
        "mean_cue_margin": _mean(
            float(row.get("cue_margin", 0.0)) for row in selected_rows
        ),
        "status": "selected_dataset_ready_for_audits"
        if selected_count == expected_pair_count and all_gates_passed == selected_count
        else "selected_dataset_needs_review",
    }


def _gate_count(rows: Sequence[Mapping[str, Any]], gate: str) -> int:
    return sum(1 for row in rows if bool(_mapping(row.get("gate_passes")).get(gate)))


def _expected_pair_keys(records: Sequence[FaultPromptRecord]) -> list[str]:
    by_key: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_key[f"{record.base_contrast_id}__{record.variant}"].add(record.label)
    return [
        key
        for key in sorted(by_key)
        if {"pseudo_cohesion", "genuine_cohesion"}.issubset(by_key[key])
    ]


def _valid_outputs_by_prompt_id(
    output_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for record in output_records:
        prompt_id = str(record.get("prompt_id", ""))
        text = str(record.get("text", "")).strip()
        if not prompt_id or not text:
            continue
        if "valid" in record and not bool(record.get("valid", False)):
            continue
        outputs[prompt_id] = text
    return outputs


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def _word_count_in_range(
    words: int,
    *,
    target_word_count_min: int,
    target_word_count_max: int,
) -> bool:
    return target_word_count_min <= words <= target_word_count_max


def _word_count_distance(words: int, minimum: int, maximum: int) -> int:
    if minimum <= words <= maximum:
        return 0
    if words < minimum:
        return minimum - words
    return words - maximum


def _format_issue_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = sum(1 for line in lines if re.match(r"^([-*]|\d+[.)])\s+", line))
    heading_lines = sum(1 for line in lines if line.endswith(":"))
    return bullet_lines + heading_lines


def _stable_variant(raw_variant: str, *, provider: str) -> str:
    suffix = f"_{_slug(provider)}"
    if raw_variant.endswith(suffix):
        return raw_variant[: -len(suffix)]
    return raw_variant


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: Sequence[float] | Any) -> float:
    concrete = list(values)
    if not concrete:
        return 0.0
    return round(sum(concrete) / len(concrete), 6)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else []


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _sequence(value)]
