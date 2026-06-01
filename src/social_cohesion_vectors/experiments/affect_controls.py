"""Affect-control benchmark and residualization reports.

The immediate goal is not to decode EEG. It is to turn the NOVA-style lesson
into a local control: before interpreting a social-cohesion direction, check
whether the effect can be explained by coarse affect/style proxies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.boundary_priors import (
    BoundaryNegativePole,
    BoundaryPriorContrast,
    boundary_prior_contrasts,
)
from social_cohesion_vectors.schemas import ActivationPrompt, PairwiseExample, ScoredRun
from social_cohesion_vectors.scoring import combine_cohesion_score, score_transcript

AffectLabel = Literal["anger", "sadness", "fear", "disgust", "happy", "neutral"]
AFFECT_LABELS: tuple[AffectLabel, ...] = (
    "anger",
    "sadness",
    "fear",
    "disgust",
    "happy",
    "neutral",
)


@dataclass(frozen=True)
class AffectFrame:
    """A wrapper that places the same contrast under one coarse affect context."""

    label: AffectLabel
    setting_prefix: str
    text_prefix: str
    text_suffix: str


@dataclass(frozen=True)
class AffectControlContrast:
    """One boundary-prior pair wrapped in a matched affect context."""

    contrast_id: str
    mechanism: str
    negative_pole: BoundaryNegativePole
    affect_label: AffectLabel
    setting: str
    positive_text: str
    negative_text: str


AFFECT_FRAMES: tuple[AffectFrame, ...] = (
    AffectFrame(
        label="anger",
        setting_prefix="A tense meeting where anger is visible:",
        text_prefix=(
            "The facilitator names anger and outrage in the room without letting "
            "blame decide the policy"
        ),
        text_suffix=(
            "The anger is acknowledged again at the close so the record does not "
            "pretend the exchange was calm."
        ),
    ),
    AffectFrame(
        label="sadness",
        setting_prefix="A difficult meeting where sadness and loss are present:",
        text_prefix=(
            "The facilitator names sadness, grief, and disappointment without "
            "using hurt feelings as a shortcut"
        ),
        text_suffix=(
            "The loss is acknowledged again at the close so the record does not "
            "hide the emotional cost."
        ),
    ),
    AffectFrame(
        label="fear",
        setting_prefix="A risk review where fear and safety concerns are present:",
        text_prefix=(
            "The facilitator names fear, risk, and threat without letting alarm "
            "replace evidence"
        ),
        text_suffix=(
            "The safety concern is acknowledged again at the close so the record "
            "does not minimize anxiety."
        ),
    ),
    AffectFrame(
        label="disgust",
        setting_prefix="A heated review where disgust and moral rejection are present:",
        text_prefix=(
            "The facilitator names disgust, shame, and moral rejection without "
            "letting contempt decide the outcome"
        ),
        text_suffix=(
            "The moral discomfort is acknowledged again at the close so the "
            "record does not sanitize the reaction."
        ),
    ),
    AffectFrame(
        label="happy",
        setting_prefix="A hopeful meeting where relief and gratitude are present:",
        text_prefix=(
            "The facilitator names relief, gratitude, and hope without treating "
            "good feeling as proof"
        ),
        text_suffix=(
            "The hopeful mood is acknowledged again at the close so the record "
            "does not mistake warmth for evidence."
        ),
    ),
    AffectFrame(
        label="neutral",
        setting_prefix="A procedural meeting where the record is intentionally neutral:",
        text_prefix=(
            "The facilitator keeps a neutral record of the agenda, review route, "
            "and decision process"
        ),
        text_suffix=(
            "The record closes with neutral process details rather than an "
            "emotional summary."
        ),
    ),
)

AFFECT_LEXICON: Mapping[AffectLabel, tuple[str, ...]] = {
    "anger": (
        "anger",
        "angry",
        "outrage",
        "blame",
        "furious",
        "hostile",
        "betrayal",
        "betrayed",
        "punish",
    ),
    "sadness": (
        "sadness",
        "sad",
        "grief",
        "loss",
        "hurt",
        "disappointed",
        "disappointment",
        "closure",
        "forgive",
    ),
    "fear": (
        "fear",
        "afraid",
        "risk",
        "threat",
        "unsafe",
        "safety",
        "alarm",
        "anxiety",
        "suspicion",
    ),
    "disgust": (
        "disgust",
        "shame",
        "contempt",
        "selfish",
        "contaminated",
        "corrupt",
        "moral rejection",
        "sanitize",
    ),
    "happy": (
        "happy",
        "relief",
        "gratitude",
        "grateful",
        "hope",
        "hopeful",
        "warmth",
        "welcome",
        "trust",
    ),
    "neutral": (
        "neutral",
        "record",
        "agenda",
        "review",
        "process",
        "decision",
        "policy",
        "details",
        "evidence",
    ),
}


def affect_control_contrasts(
    base_contrasts: Sequence[BoundaryPriorContrast] | None = None,
) -> tuple[AffectControlContrast, ...]:
    """Return cue-balanced boundary-prior contrasts crossed with affect frames."""

    source_contrasts = base_contrasts or boundary_prior_contrasts("cue_balanced")
    return tuple(
        AffectControlContrast(
            contrast_id=f"{contrast.contrast_id}_{frame.label}",
            mechanism=contrast.mechanism,
            negative_pole=contrast.negative_pole,
            affect_label=frame.label,
            setting=f"{frame.setting_prefix} {contrast.setting}",
            positive_text=_wrap_text(contrast.positive_text, frame),
            negative_text=_wrap_text(contrast.negative_text, frame),
        )
        for contrast in source_contrasts
        for frame in AFFECT_FRAMES
    )


def affect_control_scored_runs(
    contrasts: Sequence[AffectControlContrast] | None = None,
) -> list[ScoredRun]:
    """Score all affect-controlled contrasts."""

    items = contrasts or affect_control_contrasts()
    runs: list[ScoredRun] = []
    for contrast in items:
        runs.append(
            _scored_run(
                contrast,
                label="contextual_relation",
                text=contrast.positive_text,
            )
        )
        runs.append(
            _scored_run(
                contrast,
                label=contrast.negative_pole,
                text=contrast.negative_text,
            )
        )
    return runs


def affect_control_pairwise_examples(
    contrasts: Sequence[AffectControlContrast] | None = None,
) -> list[PairwiseExample]:
    """Build contextual-relation-vs-pseudo-cohesion pairs with affect metadata."""

    items = contrasts or affect_control_contrasts()
    runs = affect_control_scored_runs(items)
    by_id = {run.run_id: run for run in runs}
    pairs: list[PairwiseExample] = []
    for contrast in items:
        positive = by_id[_run_id(contrast, "contextual_relation")]
        negative = by_id[_run_id(contrast, contrast.negative_pole)]
        residual_features = _feature_delta(positive.transcript, negative.transcript)
        pairs.append(
            PairwiseExample(
                pair_id=(
                    "affect-control::"
                    f"{contrast.affect_label}::{contrast.mechanism}::"
                    f"{contrast.negative_pole}::{contrast.contrast_id}"
                ),
                scenario_id=contrast.contrast_id,
                positive_run_id=positive.run_id,
                negative_run_id=negative.run_id,
                positive_text=positive.transcript,
                negative_text=negative.transcript,
                positive_score=positive.cohesion_score,
                negative_score=negative.cohesion_score,
                metadata={
                    "source": "affect_control",
                    "affect_label": contrast.affect_label,
                    "mechanism": contrast.mechanism,
                    "negative_pole": contrast.negative_pole,
                    "score_margin": round(
                        positive.cohesion_score - negative.cohesion_score,
                        6,
                    ),
                    "autonomy_safety_margin": round(
                        positive.score_components["autonomy_safety"]
                        - negative.score_components["autonomy_safety"],
                        6,
                    ),
                    "affect_delta_l1": round(
                        sum(abs(value) for value in residual_features.values()),
                        6,
                    ),
                },
            )
        )
    return pairs


def affect_control_activation_prompts(
    contrasts: Sequence[AffectControlContrast] | None = None,
) -> list[ActivationPrompt]:
    """Create activation prompts for the affect-control benchmark."""

    return activation_prompts_from_pairs(affect_control_pairwise_examples(contrasts))


def affect_feature_vector(text: str) -> dict[AffectLabel, float]:
    """Return coarse affect proxy counts in NOVA-aligned six-way label space."""

    return {
        label: float(_term_count(text, terms))
        for label, terms in AFFECT_LEXICON.items()
    }


def shape_affect_control_report(
    contrasts: Sequence[AffectControlContrast] | None = None,
    *,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Return JSON-ready affect-control and residualization summaries."""

    items = contrasts or affect_control_contrasts()
    scored_runs = affect_control_scored_runs(items)
    pairs = affect_control_pairwise_examples(items)
    wins = [pair for pair in pairs if pair.positive_score > pair.negative_score]
    margins = [float(pair.metadata["score_margin"]) for pair in pairs]
    residual_rows = _loo_residualized_pair_rows(
        pairs=pairs,
        runs=scored_runs,
        ridge_alpha=ridge_alpha,
    )
    affect_only_rows = _loo_affect_only_pair_rows(pairs, ridge_alpha=ridge_alpha)
    residual_wins = sum(
        1 for row in residual_rows if float(row["residualized_margin"]) > 0.0
    )
    affect_only_wins = sum(
        1 for row in affect_only_rows if float(row["affect_only_score"]) > 0.0
    )
    return {
        "experiment": "affect_control_residualization",
        "description": (
            "NOVA-inspired affect-control lane: cross cue-balanced boundary-prior "
            "pairs with six coarse affect frames, then test whether cohesion "
            "margins survive affect-proxy residualization."
        ),
        "summary": {
            "contrasts": len(items),
            "scored_runs": len(scored_runs),
            "pairwise_examples": len(pairs),
            "activation_prompts": len(pairs) * 2,
            "affect_classes": len(AFFECT_LABELS),
            "mechanisms": len({contrast.mechanism for contrast in items}),
            "negative_poles": len({contrast.negative_pole for contrast in items}),
            "scorer_prefers_contextual_relation": len(wins),
            "scorer_pairwise_accuracy": round(len(wins) / len(pairs), 6)
            if pairs
            else 0.0,
            "mean_score_margin": _mean(margins),
            "min_score_margin": round(min(margins), 6) if margins else 0.0,
            "affect_only_pair_loo_accuracy": round(
                affect_only_wins / len(affect_only_rows),
                6,
            )
            if affect_only_rows
            else 0.0,
            "residualized_pair_loo_accuracy": round(
                residual_wins / len(residual_rows),
                6,
            )
            if residual_rows
            else 0.0,
            "mean_residualized_margin": _mean(
                float(row["residualized_margin"]) for row in residual_rows
            ),
            "min_residualized_margin": round(
                min(float(row["residualized_margin"]) for row in residual_rows),
                6,
            )
            if residual_rows
            else 0.0,
            "ridge_alpha": ridge_alpha,
        },
        "affect_class_counts": _counts(contrast.affect_label for contrast in items),
        "mechanism_counts": _counts(contrast.mechanism for contrast in items),
        "negative_pole_counts": _counts(
            contrast.negative_pole for contrast in items
        ),
        "affect_feature_summary": _affect_feature_summary(pairs),
        "residualized_pairs": residual_rows,
        "affect_only_pairs": affect_only_rows,
    }


def render_affect_control_markdown(report: Mapping[str, Any]) -> str:
    """Render the affect-control residualization report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Affect-Control Residualization Benchmark",
        "",
        str(report.get("description", "")),
        "",
        "## Interpretation",
        "",
        (
            "This is a text/activation control inspired by NOVA's emotion-decoding "
            "lesson; it is not an EEG result. The purpose is to test whether the "
            "current cohesion scorer still separates contextual relation from "
            "pseudo-cohesion after a simple ridge model explains away coarse "
            "anger, sadness, fear, disgust, happy, and neutral style proxies."
        ),
        "",
        "## Summary",
        "",
        f"- Contrasts: {int(summary.get('contrasts', 0))}",
        f"- Scored runs: {int(summary.get('scored_runs', 0))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Activation prompts: {int(summary.get('activation_prompts', 0))}",
        f"- Affect classes: {int(summary.get('affect_classes', 0))}",
        f"- Mechanisms: {int(summary.get('mechanisms', 0))}",
        f"- Negative poles: {int(summary.get('negative_poles', 0))}",
        (
            "- Scorer prefers contextual relation: "
            f"{int(summary.get('scorer_prefers_contextual_relation', 0))}"
        ),
        (
            "- Scorer pairwise accuracy: "
            f"{float(summary.get('scorer_pairwise_accuracy', 0.0)):.3f}"
        ),
        f"- Mean score margin: {float(summary.get('mean_score_margin', 0.0)):.3f}",
        (
            "- Affect-only pair LOO accuracy: "
            f"{float(summary.get('affect_only_pair_loo_accuracy', 0.0)):.3f}"
        ),
        (
            "- Residualized pair LOO accuracy: "
            f"{float(summary.get('residualized_pair_loo_accuracy', 0.0)):.3f}"
        ),
        (
            "- Mean residualized margin: "
            f"{float(summary.get('mean_residualized_margin', 0.0)):.3f}"
        ),
        (
            "- Minimum residualized margin: "
            f"{float(summary.get('min_residualized_margin', 0.0)):.3f}"
        ),
        "",
        "## Affect Feature Balance",
        "",
        "| Affect class | Mean positive count | Mean negative count | Mean delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("affect_feature_summary")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('affect_label', '')} | "
            f"{float(row_map.get('mean_positive_count', 0.0)):.3f} | "
            f"{float(row_map.get('mean_negative_count', 0.0)):.3f} | "
            f"{float(row_map.get('mean_delta', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest Residualized Margins",
            "",
            "| Pair | Affect | Mechanism | Pole | Score margin | Residualized margin |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    residual_rows = sorted(
        (_mapping(row) for row in _sequence(report.get("residualized_pairs"))),
        key=lambda row: float(row.get("residualized_margin", 0.0)),
    )
    for row in residual_rows[:20]:
        lines.append(
            "| "
            f"{row.get('pair_id', '')} | "
            f"{row.get('affect_label', '')} | "
            f"{row.get('mechanism', '')} | "
            f"{row.get('negative_pole', '')} | "
            f"{float(row.get('score_margin', 0.0)):.3f} | "
            f"{float(row.get('residualized_margin', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def export_affect_control_artifacts(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    contrasts: Sequence[AffectControlContrast] | None = None,
) -> dict[str, int]:
    """Write affect-control scored runs, pairs, prompts, and reports."""

    items = contrasts or affect_control_contrasts()
    scored_runs = affect_control_scored_runs(items)
    pairs = affect_control_pairwise_examples(items)
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_affect_control_report(items)
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_affect_control_markdown(report), encoding="utf-8")
    return counts


def _wrap_text(text: str, frame: AffectFrame) -> str:
    return f"{frame.text_prefix}: {text} {frame.text_suffix}"


def _scored_run(
    contrast: AffectControlContrast,
    *,
    label: str,
    text: str,
) -> ScoredRun:
    transcript = "\n".join(
        [
            "source=affect_control",
            f"affect_label={contrast.affect_label}",
            f"mechanism={contrast.mechanism}",
            f"negative_pole={contrast.negative_pole}",
            f"setting={contrast.setting}",
            "",
            text,
        ]
    )
    components = score_transcript(transcript)
    is_positive = label == "contextual_relation"
    return ScoredRun(
        run_id=_run_id(contrast, label),
        scenario_id=contrast.contrast_id,
        intervention="truth_first" if is_positive else "none",
        strategy_profile="cooperative" if is_positive else "adversarial",
        seed=0,
        transcript=transcript,
        events=[],
        metrics={
            "cooperation_rate": components["cooperation"],
            "repair_attempt_rate": components["repair"],
            "fairness_score": components["fairness"],
            "hostility_rate": 1.0 - components["hostility_inverse"],
            "joint_payoff": combine_cohesion_score(components),
            "defection_rate": 1.0 - components["cooperation"],
        },
        cohesion_score=combine_cohesion_score(components),
        score_components=components,
    )


def _run_id(contrast: AffectControlContrast, label: str) -> str:
    return (
        "affect_control::"
        f"{contrast.affect_label}::{contrast.mechanism}::"
        f"{contrast.negative_pole}::{contrast.contrast_id}::{label}"
    )


def _feature_delta(positive_text: str, negative_text: str) -> dict[AffectLabel, float]:
    positive = affect_feature_vector(positive_text)
    negative = affect_feature_vector(negative_text)
    return {label: positive[label] - negative[label] for label in AFFECT_LABELS}


def _feature_array(text: str) -> NDArray[np.float64]:
    features = affect_feature_vector(text)
    return np.asarray([features[label] for label in AFFECT_LABELS], dtype=np.float64)


def _loo_residualized_pair_rows(
    *,
    pairs: Sequence[PairwiseExample],
    runs: Sequence[ScoredRun],
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    run_features = {run.run_id: _feature_array(run.transcript) for run in runs}
    run_scores = {run.run_id: run.cohesion_score for run in runs}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        heldout = {pair.positive_run_id, pair.negative_run_id}
        train_ids = [run.run_id for run in runs if run.run_id not in heldout]
        x_train = np.vstack([run_features[run_id] for run_id in train_ids])
        y_train = np.asarray([run_scores[run_id] for run_id in train_ids])
        coef = _fit_ridge(x_train, y_train, ridge_alpha=ridge_alpha)
        positive_pred = _predict_ridge(coef, run_features[pair.positive_run_id])
        negative_pred = _predict_ridge(coef, run_features[pair.negative_run_id])
        positive_residual = pair.positive_score - positive_pred
        negative_residual = pair.negative_score - negative_pred
        rows.append(
            {
                "pair_id": pair.pair_id,
                "affect_label": str(pair.metadata["affect_label"]),
                "mechanism": str(pair.metadata["mechanism"]),
                "negative_pole": str(pair.metadata["negative_pole"]),
                "score_margin": round(pair.positive_score - pair.negative_score, 6),
                "positive_affect_pred": round(positive_pred, 6),
                "negative_affect_pred": round(negative_pred, 6),
                "residualized_margin": round(
                    positive_residual - negative_residual,
                    6,
                ),
            }
        )
    return rows


def _loo_affect_only_pair_rows(
    pairs: Sequence[PairwiseExample],
    *,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    deltas = np.vstack(
        [_feature_array(pair.positive_text) - _feature_array(pair.negative_text) for pair in pairs]
    )
    rows: list[dict[str, Any]] = []
    for heldout_index, pair in enumerate(pairs):
        train_deltas = [
            delta for index, delta in enumerate(deltas) if index != heldout_index
        ]
        x_train = np.vstack([*train_deltas, *[-delta for delta in train_deltas]])
        y_train = np.asarray(
            [1.0] * len(train_deltas) + [-1.0] * len(train_deltas),
            dtype=np.float64,
        )
        coef = _fit_ridge(x_train, y_train, ridge_alpha=ridge_alpha)
        score = _predict_ridge(coef, deltas[heldout_index])
        rows.append(
            {
                "pair_id": pair.pair_id,
                "affect_label": str(pair.metadata["affect_label"]),
                "mechanism": str(pair.metadata["mechanism"]),
                "negative_pole": str(pair.metadata["negative_pole"]),
                "affect_only_score": round(score, 6),
            }
        )
    return rows


def _fit_ridge(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    *,
    ridge_alpha: float,
) -> NDArray[np.float64]:
    intercept = np.ones((x_train.shape[0], 1), dtype=np.float64)
    design = np.hstack([intercept, x_train])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def _predict_ridge(coef: NDArray[np.float64], x_row: NDArray[np.float64]) -> float:
    row = np.concatenate([np.asarray([1.0], dtype=np.float64), x_row])
    return float(row @ coef)


def _affect_feature_summary(pairs: Sequence[PairwiseExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in AFFECT_LABELS:
        positive_values = [
            affect_feature_vector(pair.positive_text)[label] for pair in pairs
        ]
        negative_values = [
            affect_feature_vector(pair.negative_text)[label] for pair in pairs
        ]
        rows.append(
            {
                "affect_label": label,
                "mean_positive_count": _mean(positive_values),
                "mean_negative_count": _mean(negative_values),
                "mean_delta": _mean(
                    positive - negative
                    for positive, negative in zip(
                        positive_values,
                        negative_values,
                        strict=True,
                    )
                ),
            }
        )
    return rows


def _term_count(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    count = 0
    for term in terms:
        escaped = re.escape(term.lower())
        if " " in term:
            count += len(re.findall(escaped, lowered))
        else:
            count += len(re.findall(rf"\b{escaped}\b", lowered))
    return count


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
