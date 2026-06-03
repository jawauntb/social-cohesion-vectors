from __future__ import annotations

from social_cohesion_vectors.experiments.transition_prediction import (
    evaluate_transition_baseline,
    observed_ck1_delta,
    predict_transition,
    transition_feature_key,
)


def test_recipe_mean_predicts_delta_and_majority_effect_class() -> None:
    heldout = _transition("heldout", "repair", "ck1", 0.0, "neutral_transition")
    training = [
        _transition("a", "repair", "ck1", 0.2, "beneficial_transition"),
        _transition("b", "consent", "ck1", 0.4, "beneficial_transition"),
        _transition("c", "repair", "guardrail", -0.1, "adverse_transition"),
    ]

    prediction = predict_transition(
        heldout,
        training_records=training,
        baseline="recipe_mean",
    )

    assert prediction.transition_id == "heldout"
    assert prediction.baseline == "recipe_mean"
    assert prediction.predicted_ck1_delta == 0.3
    assert prediction.predicted_effect_class == "beneficial_transition"
    assert prediction.support == 2


def test_recipe_mean_falls_back_to_global_mean_for_unseen_recipe() -> None:
    heldout = _transition("heldout", "repair", "new_recipe", 0.0, "neutral_transition")
    training = [
        _transition("a", "repair", "ck1", 0.2, "beneficial_transition"),
        _transition("b", "consent", "guardrail", -0.1, "adverse_transition"),
    ]

    prediction = predict_transition(
        heldout,
        training_records=training,
        baseline="recipe_mean",
    )

    assert prediction.predicted_ck1_delta == 0.05
    assert prediction.predicted_effect_class == "adverse_transition"
    assert prediction.support == 2


def test_nearest_context_uses_baseline_and_perturbation_metadata() -> None:
    heldout = _transition(
        "heldout",
        "repair",
        "ck1",
        0.0,
        "neutral_transition",
        mechanism="boundary",
        component_id="ck1",
        strength=0.5,
        schedule="first-4",
    )
    near = _transition(
        "near",
        "repair",
        "ck1",
        0.22,
        "beneficial_transition",
        mechanism="boundary",
        component_id="ck1",
        strength=0.5,
        schedule="first-4",
    )
    far = _transition(
        "far",
        "consent",
        "guardrail",
        -0.3,
        "adverse_transition",
        mechanism="privacy",
        component_id="privacy_exit",
        strength=0.2,
        schedule="constant",
    )

    prediction = predict_transition(
        heldout,
        training_records=[far, near],
        baseline="nearest_context",
    )

    assert transition_feature_key(heldout) == transition_feature_key(near)
    assert prediction.predicted_ck1_delta == 0.22
    assert prediction.predicted_effect_class == "beneficial_transition"
    assert prediction.source_transition_id == "near"
    assert prediction.support == 1


def test_nearest_context_uses_component_overlap_before_tie_break() -> None:
    heldout = _transition(
        "heldout",
        "repair",
        "bundle",
        0.0,
        "neutral_transition",
        component_ids=["truth", "privacy"],
    )
    overlap = _transition(
        "overlap",
        "repair",
        "bundle",
        0.2,
        "beneficial_transition",
        component_ids=["truth", "privacy", "respect"],
    )
    no_overlap = _transition(
        "aaaa",
        "repair",
        "bundle",
        -0.2,
        "adverse_transition",
        component_ids=["surveillance", "coercion"],
    )

    prediction = predict_transition(
        heldout,
        training_records=[no_overlap, overlap],
        baseline="nearest_context",
    )

    assert prediction.source_transition_id == "overlap"
    assert prediction.predicted_ck1_delta == 0.2


def test_evaluate_transition_baseline_runs_leave_one_out() -> None:
    records = [
        _transition("a", "repair", "ck1", 0.2, "beneficial_transition"),
        _transition("b", "repair", "ck1", 0.4, "beneficial_transition"),
        _transition("c", "repair", "guardrail", -0.1, "adverse_transition"),
    ]

    evaluation = evaluate_transition_baseline(records, baseline="recipe_mean")

    assert evaluation.baseline == "recipe_mean"
    assert evaluation.n_records == 3
    assert evaluation.effect_class_accuracy == 0.666667
    assert evaluation.mean_absolute_ck1_error == 0.266667
    assert [prediction.transition_id for prediction in evaluation.predictions] == [
        "a",
        "b",
        "c",
    ]
    assert [observed_ck1_delta(record) for record in records] == [0.2, 0.4, -0.1]


def _transition(
    transition_id: str,
    phase: str,
    recipe_id: str,
    ck1_delta: float,
    effect_class: str,
    *,
    mechanism: str = "boundary",
    component_id: str | None = None,
    component_ids: list[str] | None = None,
    strength: float = 0.5,
    schedule: str = "constant",
) -> dict[str, object]:
    components = [
        {
            "component_id": component,
            "strength": strength,
        }
        for component in (component_ids or [component_id or recipe_id])
    ]
    return {
        "transition_id": transition_id,
        "baseline_state": {
            "phase": phase,
            "mechanism": mechanism,
            "recipe_id": "baseline",
        },
        "perturbation": {
            "recipe_id": recipe_id,
            "components": components,
        },
        "dose": {"absolute_strength_sum": abs(strength)},
        "timing": [{"steering_schedule": schedule}],
        "effect_class": effect_class,
        "observed_transition": {"ck1_score_delta": ck1_delta},
    }
