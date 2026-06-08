"""Term-level diagnostics for fixed lexical baselines."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.baselines import NEGATIVE_WORDS, POSITIVE_WORDS
from social_cohesion_vectors.experiments.lexical_leakage import lexical_cue_score
from social_cohesion_vectors.schemas import PairwiseExample


def run_lexical_baseline_diagnostic_from_file(
    pairs_path: str | Path,
    *,
    group_metadata_key: str = "primary_fault_class",
) -> dict[str, Any]:
    """Load pairwise examples and report term-specific lexical separability."""

    return run_lexical_baseline_diagnostic(
        pairs=load_pairwise_examples_jsonl(pairs_path),
        group_metadata_key=group_metadata_key,
        input_path=str(pairs_path),
    )


def run_lexical_baseline_diagnostic(
    *,
    pairs: Sequence[PairwiseExample],
    group_metadata_key: str = "primary_fault_class",
    input_path: str | None = None,
) -> dict[str, Any]:
    """Explain why a fixed lexicon may transfer after aggregate cue balancing."""

    term_rows = _term_rows(pairs)
    pair_rows = _pair_rows(pairs, group_metadata_key=group_metadata_key)
    group_rows = _group_rows(pairs, group_metadata_key=group_metadata_key)
    top_terms = sorted(
        term_rows,
        key=lambda row: (
            float(row["best_pairwise_accuracy"]),
            float(row["mean_abs_pair_margin"]),
            str(row["term"]),
        ),
        reverse=True,
    )[:20]
    mean_cue_margin = _mean(float(row["cue_margin"]) for row in pair_rows)
    best_single_feature_accuracy = max(
        (float(row["best_pairwise_accuracy"]) for row in term_rows),
        default=0.0,
    )
    return {
        "experiment": "lexical_baseline_diagnostic",
        "description": (
            "Breaks the fixed lexical baseline into per-term label-aligned "
            "margins. This catches term-specific lexical structure that can "
            "survive aggregate positive-minus-negative cue balancing."
        ),
        "inputs": {
            "pairs_path": input_path,
            "pairs": len(pairs),
            "group_metadata_key": group_metadata_key,
            "positive_terms": list(POSITIVE_WORDS),
            "negative_terms": list(NEGATIVE_WORDS),
        },
        "summary": {
            "pairs": len(pairs),
            "terms": len(term_rows),
            "label_aligned_terms": sum(
                float(row["label_aligned_margin"]) > 0.0 for row in term_rows
            ),
            "label_inverted_terms": sum(
                float(row["label_aligned_margin"]) < 0.0 for row in term_rows
            ),
            "tied_terms": sum(
                float(row["label_aligned_margin"]) == 0.0 for row in term_rows
            ),
            "mean_pair_cue_margin": mean_cue_margin,
            "best_single_feature_accuracy": round(best_single_feature_accuracy, 6),
            "aggregate_balanced_term_polarized": (
                abs(mean_cue_margin) <= 1e-9 and best_single_feature_accuracy > 0.5
            ),
        },
        "top_terms": top_terms,
        "terms": term_rows,
        "groups": group_rows,
        "pairs": pair_rows,
    }


def save_lexical_baseline_diagnostic(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown lexical-baseline diagnostic reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_lexical_baseline_diagnostic_markdown(report),
        encoding="utf-8",
    )


def render_lexical_baseline_diagnostic_markdown(report: Mapping[str, Any]) -> str:
    """Render term-level lexical-baseline diagnostics as markdown."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Lexical Baseline Diagnostic",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Terms: {int(summary.get('terms', 0))}",
        f"- Label-aligned terms: {int(summary.get('label_aligned_terms', 0))}",
        f"- Label-inverted terms: {int(summary.get('label_inverted_terms', 0))}",
        f"- Tied terms: {int(summary.get('tied_terms', 0))}",
        f"- Mean pair cue margin: "
        f"{float(summary.get('mean_pair_cue_margin', 0.0)):+.3f}",
        f"- Best single-feature pairwise accuracy: "
        f"{float(summary.get('best_single_feature_accuracy', 0.0)):.3f}",
        f"- Aggregate balanced but term-polarized: "
        f"{bool(summary.get('aggregate_balanced_term_polarized', False))}",
        "",
        "## Top Single Features",
        "",
        "| Feature | Lexicon | Best direction | Pairwise accuracy | Mean abs margin | Positive total | Negative total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("top_terms")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('term', '')}` | "
            f"`{row_map.get('lexicon', '')}` | "
            f"`{row_map.get('best_direction', '')}` | "
            f"{float(row_map.get('best_pairwise_accuracy', 0.0)):.3f} | "
            f"{float(row_map.get('mean_abs_pair_margin', 0.0)):.3f} | "
            f"{float(row_map.get('positive_feature_sum', 0.0)):.3f} | "
            f"{float(row_map.get('negative_feature_sum', 0.0)):.3f} | "
        )
    lines.extend(
        [
            "",
            "## Groups",
            "",
            "| Group | Pairs | Mean cue margin | Top label-aligned term | Top margin |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in _sequence(report.get("groups")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('group', '')}` | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{float(row_map.get('mean_cue_margin', 0.0)):+.3f} | "
            f"`{row_map.get('top_term', '')}` | "
            f"{float(row_map.get('top_term_margin', 0.0)):+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _term_rows(pairs: Sequence[PairwiseExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lexicon, terms in _lexical_features():
        for term in terms:
            positive_values = [_feature_value(pair.positive_text, term) for pair in pairs]
            negative_values = [_feature_value(pair.negative_text, term) for pair in pairs]
            diffs = [
                positive - negative
                for positive, negative in zip(positive_values, negative_values, strict=True)
            ]
            positive_total = round(sum(positive_values), 6)
            negative_total = round(sum(negative_values), 6)
            raw_margin = round(positive_total - negative_total, 6)
            label_aligned_margin = (
                raw_margin if lexicon == "positive" else -raw_margin
            )
            positive_higher = sum(diff > 0.0 for diff in diffs)
            negative_higher = sum(diff < 0.0 for diff in diffs)
            tied = sum(diff == 0.0 for diff in diffs)
            best_wins = max(positive_higher, negative_higher)
            best_direction = (
                "positive_higher"
                if positive_higher >= negative_higher
                else "negative_higher"
            )
            rows.append(
                {
                    "term": term,
                    "lexicon": lexicon,
                    "positive_feature_sum": positive_total,
                    "negative_feature_sum": negative_total,
                    "raw_positive_minus_negative_margin": raw_margin,
                    "label_aligned_margin": label_aligned_margin,
                    "abs_label_aligned_margin": abs(label_aligned_margin),
                    "positive_higher_pairs": positive_higher,
                    "negative_higher_pairs": negative_higher,
                    "tied_pairs": tied,
                    "best_direction": best_direction,
                    "best_pairwise_accuracy": _pairwise_accuracy(
                        best_wins=best_wins,
                        ties=tied,
                        pairs=len(pairs),
                    ),
                    "mean_abs_pair_margin": _mean(abs(diff) for diff in diffs),
                }
            )
    return rows


def _pair_rows(
    pairs: Sequence[PairwiseExample],
    *,
    group_metadata_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        positive_score = lexical_cue_score(pair.positive_text)
        negative_score = lexical_cue_score(pair.negative_text)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "group": str(pair.metadata.get(group_metadata_key, "ungrouped")),
                "positive_cue_score": positive_score,
                "negative_cue_score": negative_score,
                "cue_margin": round(positive_score - negative_score, 6),
            }
        )
    return rows


def _group_rows(
    pairs: Sequence[PairwiseExample],
    *,
    group_metadata_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[PairwiseExample]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair.metadata.get(group_metadata_key, "ungrouped"))].append(pair)
    rows: list[dict[str, Any]] = []
    for group, group_pairs in sorted(grouped.items()):
        pair_rows = _pair_rows(group_pairs, group_metadata_key=group_metadata_key)
        term_rows = _term_rows(group_pairs)
        top_term = max(
            term_rows,
            key=lambda row: float(row["abs_label_aligned_margin"]),
            default={},
        )
        rows.append(
            {
                "group": group,
                "pairs": len(group_pairs),
                "mean_cue_margin": _mean(
                    float(row["cue_margin"]) for row in pair_rows
                ),
                "top_term": str(top_term.get("term", "")),
                "top_term_margin": float(top_term.get("label_aligned_margin", 0.0)),
            }
        )
    return rows


def _term_count(text: str, term: str) -> int:
    return len(re.findall(term, text.lower()))


def _feature_value(text: str, term: str) -> float:
    if term == "__log_token_count__":
        return math.log1p(float(len(re.findall(r"[a-zA-Z]+", text))))
    return float(_term_count(text, term))


def _lexical_features() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("positive", POSITIVE_WORDS),
        ("negative", NEGATIVE_WORDS),
        ("length", ("__log_token_count__",)),
    )


def _pairwise_accuracy(*, best_wins: int, ties: int, pairs: int) -> float:
    if pairs == 0:
        return 0.0
    return round((best_wins + (0.5 * ties)) / pairs, 6)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
