"""Tiny baselines for transition-record prediction benchmarks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

PredictionBaseline = Literal["global_mean", "recipe_mean", "nearest_context"]


@dataclass(frozen=True)
class TransitionPrediction:
    """One baseline prediction for a held-out transition record."""

    transition_id: str
    baseline: PredictionBaseline
    predicted_ck1_delta: float
    predicted_effect_class: str
    support: int
    source_transition_id: str | None = None


@dataclass(frozen=True)
class TransitionPredictionEvaluation:
    """Aggregate leave-one-out quality for a transition baseline."""

    baseline: PredictionBaseline
    n_records: int
    mean_absolute_ck1_error: float
    effect_class_accuracy: float
    predictions: list[TransitionPrediction]


def predict_transition(
    record: Mapping[str, Any],
    *,
    training_records: Sequence[Mapping[str, Any]],
    baseline: PredictionBaseline = "recipe_mean",
) -> TransitionPrediction:
    """Predict CK-1 delta and effect class from cheap transition baselines."""

    if baseline == "global_mean":
        return _mean_prediction(
            record,
            training_records=training_records,
            baseline=baseline,
        )
    if baseline == "recipe_mean":
        recipe_id = _perturbation_recipe_id(record)
        recipe_records = [
            item
            for item in training_records
            if _perturbation_recipe_id(item) == recipe_id
        ]
        return _mean_prediction(
            record,
            training_records=recipe_records or training_records,
            baseline=baseline,
        )
    if baseline == "nearest_context":
        return _nearest_context_prediction(record, training_records=training_records)
    raise ValueError(f"Unknown transition prediction baseline: {baseline}")


def evaluate_transition_baseline(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline: PredictionBaseline = "recipe_mean",
) -> TransitionPredictionEvaluation:
    """Evaluate a baseline with deterministic leave-one-transition-out splits."""

    predictions: list[TransitionPrediction] = []
    absolute_errors: list[float] = []
    effect_hits = 0
    for index, record in enumerate(records):
        training_records = [item for offset, item in enumerate(records) if offset != index]
        prediction = predict_transition(
            record,
            training_records=training_records,
            baseline=baseline,
        )
        predictions.append(prediction)
        absolute_errors.append(
            abs(prediction.predicted_ck1_delta - observed_ck1_delta(record))
        )
        if prediction.predicted_effect_class == observed_effect_class(record):
            effect_hits += 1

    return TransitionPredictionEvaluation(
        baseline=baseline,
        n_records=len(records),
        mean_absolute_ck1_error=_mean(absolute_errors),
        effect_class_accuracy=round(effect_hits / len(records), 6) if records else 0.0,
        predictions=predictions,
    )


def transition_feature_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the benchmark-only features used by nearest-context matching."""

    baseline_state = _mapping(record.get("baseline_state"))
    perturbation = _mapping(record.get("perturbation"))
    dose = _mapping(record.get("dose"))
    return (
        baseline_state.get("phase", ""),
        baseline_state.get("mechanism", ""),
        baseline_state.get("recipe_id", ""),
        perturbation.get("recipe_id", ""),
        tuple(_component_ids(perturbation.get("components"))),
        round(float(dose.get("absolute_strength_sum", 0.0)), 6),
        tuple(_timing_schedules(record.get("timing"))),
    )


def observed_ck1_delta(record: Mapping[str, Any]) -> float:
    """Read the observed CK-1 score delta from a transition record."""

    observed = _mapping(record.get("observed_transition"))
    return round(float(observed.get("ck1_score_delta", 0.0)), 6)


def observed_effect_class(record: Mapping[str, Any]) -> str:
    """Read the observed transition effect class."""

    return str(record.get("effect_class", "neutral_transition"))


def _mean_prediction(
    record: Mapping[str, Any],
    *,
    training_records: Sequence[Mapping[str, Any]],
    baseline: PredictionBaseline,
) -> TransitionPrediction:
    return TransitionPrediction(
        transition_id=str(record.get("transition_id", "")),
        baseline=baseline,
        predicted_ck1_delta=_mean(
            [observed_ck1_delta(item) for item in training_records]
        ),
        predicted_effect_class=_majority_effect_class(training_records),
        support=len(training_records),
    )


def _nearest_context_prediction(
    record: Mapping[str, Any],
    *,
    training_records: Sequence[Mapping[str, Any]],
) -> TransitionPrediction:
    if not training_records:
        return _mean_prediction(record, training_records=[], baseline="nearest_context")

    heldout_key = transition_feature_key(record)
    nearest = min(
        training_records,
        key=lambda item: (
            _feature_distance(heldout_key, transition_feature_key(item)),
            str(item.get("transition_id", "")),
        ),
    )
    return TransitionPrediction(
        transition_id=str(record.get("transition_id", "")),
        baseline="nearest_context",
        predicted_ck1_delta=observed_ck1_delta(nearest),
        predicted_effect_class=observed_effect_class(nearest),
        support=1,
        source_transition_id=str(nearest.get("transition_id", "")),
    )


def _feature_distance(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    distance = 0.0
    for left_value, right_value in zip(left[:4], right[:4], strict=True):
        distance += 0.0 if left_value == right_value else 1.0
    distance += _jaccard_distance(set(left[4]), set(right[4]))
    distance += abs(float(left[5]) - float(right[5]))
    distance += _jaccard_distance(set(left[6]), set(right[6]))
    return distance


def _jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / len(left | right))


def _majority_effect_class(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "neutral_transition"
    counts = Counter(observed_effect_class(record) for record in records)
    return min(counts, key=lambda label: (-counts[label], label))


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _perturbation_recipe_id(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("perturbation")).get("recipe_id", ""))


def _component_ids(value: object) -> list[str]:
    return [
        str(component.get("component_id", ""))
        for component in _sequence_of_mappings(value)
    ]


def _timing_schedules(value: object) -> list[str]:
    return [
        str(item.get("steering_schedule", ""))
        for item in _sequence_of_mappings(value)
    ]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]
