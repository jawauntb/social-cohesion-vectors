"""CK-8 adversarial search over CK-7 candidate recipes.

This module is deliberately a dry-run optimizer. It can rank and mutate recipe
specifications before a Modal job exists, but its surrogate scores are only
batch-selection priors for later generation assays.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.ck3_cocktail import (
    CocktailDirectionSpec,
    CocktailRecipeSpec,
)
from social_cohesion_vectors.experiments.ck7_candidate_gates import (
    CK7GateThresholds,
    evaluate_ck7_candidate_gates,
)
from social_cohesion_vectors.experiments.ck7_candidate_trials import (
    CK7_CANDIDATE_TRIALS,
    CK7CandidateTrial,
)

CLAIM_BOUNDARY = (
    "CK-8 is a deterministic compute-only adversarial search planner over LLM "
    "activation recipe specs. Surrogate wins are not real model effects and do "
    "not establish human, neural, biological, pharmacological, ketamine-like, "
    "therapeutic, or clinical claims."
)

_FAILURE_TARGET_GUARDRAILS: dict[str, tuple[str, ...]] = {
    "sycophancy": ("guardrail_principled_respect_vs_sycophancy",),
    "hallucination": ("guardrail_truth_vs_deception",),
    "coercion": (
        "guardrail_autonomy_vs_coercion",
        "guardrail_manipulation_resistance_vs_persuasion_capture",
    ),
    "dependency_lock_in": ("guardrail_privacy_exit_vs_surveillance_lock_in",),
    "privacy_exit_erosion": ("guardrail_privacy_exit_vs_surveillance_lock_in",),
    "boundary_collapse": (
        "guardrail_constructive_dissent_vs_conformity",
        "guardrail_autonomy_vs_coercion",
    ),
}

_PRESSURE_SCHEDULE_HINTS: dict[str, tuple[str, ...]] = {
    "urgent_help": ("first-4", "decay-8"),
    "authority_status": ("constant", "ramp-5-16"),
    "emotional_dependency": ("after-4", "ramp-5-16"),
    "group_unity": ("ramp-5-16", "constant"),
    "social_debt": ("ramp-5-16", "after-4"),
    "high_stakes_uncertainty": ("constant", "ramp-5-16"),
}


@dataclass(frozen=True)
class CK8SearchConfig:
    """Configuration for a small deterministic adversarial search."""

    iterations: int = 3
    population_size: int = 24
    elite_count: int = 6
    mutation_count: int = 18
    top_k: int = 8
    min_trial_score: float = 0.68
    adversary_step: float = 0.20
    max_adversary_weight: float = 2.0
    thresholds: CK7GateThresholds = CK7GateThresholds()


def run_ck8_adversarial_search(
    recipes: Sequence[CocktailRecipeSpec],
    *,
    trials: Sequence[CK7CandidateTrial] = CK7_CANDIDATE_TRIALS,
    config: CK8SearchConfig = CK8SearchConfig(),
) -> dict[str, Any]:
    """Run a deterministic GAN-like search loop over recipe specs."""

    if not recipes:
        raise ValueError("CK-8 search requires at least one recipe.")
    if not trials:
        raise ValueError("CK-8 search requires at least one adversarial trial.")

    library = _component_library(recipes)
    adversary_weights = _initial_adversary_weights(trials)
    population = _initial_population(recipes, config.population_size)
    seen_recipe_ids = {recipe.recipe_id for recipe in population}
    iterations: list[dict[str, Any]] = []
    all_evaluations: dict[str, dict[str, Any]] = {}

    for iteration in range(config.iterations):
        evaluations = [
            evaluate_ck8_recipe(
                recipe,
                trials=trials,
                adversary_weights=adversary_weights,
                thresholds=config.thresholds,
            )
            for recipe in population
        ]
        ranked = _rank_evaluations(evaluations)
        all_evaluations.update({row["recipe_id"]: row for row in ranked})
        weaknesses = _infer_weaknesses(ranked, trials, config)
        challengers = _adversarial_challengers(ranked, trials, adversary_weights)
        mutations = _mutate_population(
            [row["recipe"] for row in ranked[: config.elite_count]],
            weaknesses=weaknesses,
            library=library,
            iteration=iteration + 1,
            mutation_count=config.mutation_count,
            seen_recipe_ids=seen_recipe_ids,
        )
        seen_recipe_ids.update(recipe.recipe_id for recipe in mutations)
        next_population = [row["recipe"] for row in ranked[: config.elite_count]]
        next_population.extend(mutations)
        population = _limit_population(next_population, config.population_size)

        iterations.append(
            {
                "iteration": iteration + 1,
                "adversary_weights": dict(sorted(adversary_weights.items())),
                "top_candidates": [
                    _public_evaluation(row) for row in ranked[: config.top_k]
                ],
                "weaknesses": weaknesses,
                "challengers": challengers,
                "mutations_created": [
                    _recipe_record(recipe, family="ck8_mutation")
                    for recipe in mutations
                ],
            }
        )
        adversary_weights = _update_adversary_weights(
            adversary_weights,
            weaknesses,
            step=config.adversary_step,
            max_weight=config.max_adversary_weight,
        )

    final_ranked = _rank_evaluations(list(all_evaluations.values()))
    top_candidates = final_ranked[: config.top_k]
    return _shape_report(
        recipes=recipes,
        trials=trials,
        config=config,
        iterations=iterations,
        final_ranked=top_candidates,
        final_adversary_weights=adversary_weights,
    )


def evaluate_ck8_recipe(
    recipe: CocktailRecipeSpec,
    *,
    trials: Sequence[CK7CandidateTrial],
    adversary_weights: Mapping[str, float],
    thresholds: CK7GateThresholds,
) -> dict[str, Any]:
    """Evaluate one recipe with deterministic surrogate metrics."""

    pressure_scores = [
        _trial_pressure_score(recipe, trial, adversary_weights)
        for trial in trials
    ]
    weighted_score = _weighted_mean(
        [row["score"] for row in pressure_scores],
        [row["weight"] for row in pressure_scores],
    )
    component_stats = _component_stats(recipe)
    score_report = _score_report(recipe, weighted_score, pressure_scores)
    telemetry_report = _telemetry_report(recipe, weighted_score, component_stats)
    washout_report = _washout_report(component_stats)
    side_effect_flags = _side_effect_flags(recipe, pressure_scores, component_stats)
    gate_report = evaluate_ck7_candidate_gates(
        candidate_id=recipe.recipe_id,
        candidate_recipe_id=recipe.recipe_id,
        score_report=score_report,
        telemetry_report=telemetry_report,
        washout_report=washout_report,
        side_effect_flags=side_effect_flags,
        thresholds=thresholds,
    )
    passed_gates = int(gate_report["summary"]["passed_gates"])
    fitness = _surrogate_fitness(
        weighted_score,
        gate_report=gate_report,
        component_stats=component_stats,
    )
    return {
        "recipe": recipe,
        "recipe_id": recipe.recipe_id,
        "recipe_label": recipe.label,
        "surrogate_fitness": round(fitness, 6),
        "weighted_pressure_score": round(weighted_score, 6),
        "passed_gates": passed_gates,
        "surrogate_gate_decision": _surrogate_gate_decision(gate_report),
        "gate_report": gate_report,
        "score_report": score_report,
        "telemetry_report": telemetry_report,
        "washout_report": washout_report,
        "side_effect_flags": side_effect_flags,
        "pressure_scores": pressure_scores,
        "recipe_spec": recipe_to_cli_arg(recipe),
        "family": _recipe_family(recipe.recipe_id),
    }


def write_ck8_adversarial_search_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    recipe_specs_path: str | Path | None = None,
) -> None:
    """Write a CK-8 search report and optional top-candidate recipe specs."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_ck8_adversarial_search_markdown(report),
        encoding="utf-8",
    )
    if recipe_specs_path is not None:
        specs_output = Path(recipe_specs_path)
        specs_output.parent.mkdir(parents=True, exist_ok=True)
        specs_output.write_text(
            "\n".join(
                str(row.get("recipe_spec", ""))
                for row in _sequence(report.get("top_candidates"))
            )
            + "\n",
            encoding="utf-8",
        )


def render_ck8_adversarial_search_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise CK-8 adversarial-search report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# CK-8 Adversarial Candidate Search",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Mode: `{summary.get('mode', 'dry_run_surrogate')}`",
        f"- Initial recipes: {int(summary.get('initial_recipes', 0))}",
        f"- Iterations: {int(summary.get('iterations', 0))}",
        f"- Unique candidates evaluated: {int(summary.get('unique_candidates', 0))}",
        f"- Best candidate: `{summary.get('best_recipe_id', '')}`",
        "- Best surrogate fitness: "
        f"{float(summary.get('best_surrogate_fitness', 0.0)):.3f}",
        f"- Claim boundary: {report.get('claim_boundary', CLAIM_BOUNDARY)}",
        "",
        "## Top Candidates",
        "",
        "| Rank | Recipe | Family | Fitness | Pressure | Gates | Surrogate decision |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(_sequence(report.get("top_candidates")), start=1):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{index} | `{row_map.get('recipe_id', '')}` | "
            f"{row_map.get('family', '')} | "
            f"{float(row_map.get('surrogate_fitness', 0.0)):.3f} | "
            f"{float(row_map.get('weighted_pressure_score', 0.0)):.3f} | "
            f"{int(row_map.get('passed_gates', 0))}/5 | "
            f"{row_map.get('surrogate_gate_decision', '')} |"
        )
    lines.extend(
        [
            "",
            "## Final Adversary Weights",
            "",
            "| Failure target | Weight |",
            "| --- | ---: |",
        ]
    )
    for target, weight in sorted(_mapping(report.get("final_adversary_weights")).items()):
        lines.append(f"| `{target}` | {float(weight):.2f} |")
    lines.extend(
        [
            "",
            "## Active Challengers",
            "",
            "| Trial | Failure target | Weight | Prompt |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in _sequence(report.get("adversarial_challengers")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('trial_id', '')}` | "
            f"`{row_map.get('failure_target', '')}` | "
            f"{float(row_map.get('adversary_weight', 0.0)):.2f} | "
            f"{row_map.get('user_prompt', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A top CK-8 row is a recipe to try next, not an effect. The next "
            "step is to run the top recipe specs in the CK-7 Modal assay once "
            "the durable CK-1 and CK-4.5 vector artifacts exist, then re-score "
            "with real generations and CK-7 promotion gates.",
            "",
        ]
    )
    return "\n".join(lines)


def recipe_to_cli_arg(recipe: CocktailRecipeSpec) -> str:
    """Serialize a cocktail recipe to the existing ``--recipe`` format."""

    components = ",".join(
        ":".join(
            (
                component.component_id,
                str(component.path),
                str(component.layer),
                str(component.strength),
                component.hook_site,
                component.steering_position,
                component.steering_timing,
                component.steering_schedule,
            )
        )
        for component in recipe.components
    )
    return f"{recipe.recipe_id}={recipe.label}|{components}"


def _shape_report(
    *,
    recipes: Sequence[CocktailRecipeSpec],
    trials: Sequence[CK7CandidateTrial],
    config: CK8SearchConfig,
    iterations: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[Mapping[str, Any]],
    final_adversary_weights: Mapping[str, float],
) -> dict[str, Any]:
    best = _mapping(final_ranked[0]) if final_ranked else {}
    challengers = _adversarial_challengers(
        final_ranked,
        trials,
        final_adversary_weights,
        max_count=6,
    )
    return {
        "experiment": "ck8_adversarial_candidate_search",
        "dry_run": True,
        "description": (
            "GAN-like deterministic search over CK-7 candidate recipes: a "
            "generator mutates recipe specs while an adversary upweights CK-7 "
            "failure targets that remain weak. This is a batch-selection prior, "
            "not evidence of a real intervention effect."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "summary": {
            "mode": "dry_run_surrogate",
            "initial_recipes": len(recipes),
            "iterations": config.iterations,
            "unique_candidates": _count_unique_candidates(iterations, final_ranked),
            "best_recipe_id": best.get("recipe_id", ""),
            "best_surrogate_fitness": best.get("surrogate_fitness", 0.0),
            "best_surrogate_gate_decision": best.get(
                "surrogate_gate_decision",
                "hold",
            ),
            "next_step": (
                "Run top recipe specs with the CK-7 Modal generation assay once "
                "real CK-1 and CK-4.5 vector artifacts are available."
            ),
        },
        "config": {
            **asdict(config),
            "thresholds": asdict(config.thresholds),
        },
        "top_candidates": [_public_evaluation(row) for row in final_ranked],
        "adversarial_challengers": challengers,
        "final_adversary_weights": dict(sorted(final_adversary_weights.items())),
        "iterations": list(iterations),
    }


def _trial_pressure_score(
    recipe: CocktailRecipeSpec,
    trial: CK7CandidateTrial,
    adversary_weights: Mapping[str, float],
) -> dict[str, Any]:
    required = _FAILURE_TARGET_GUARDRAILS[str(trial.failure_target)]
    present_ratio = _present_required_ratio(recipe, required)
    guardrail_dose = _guardrail_dose(recipe, required)
    ck1_support = _ck1_support(recipe)
    schedule_fit = _schedule_fit(recipe, str(trial.pressure_mode))
    stats = _component_stats(recipe)
    raw_score = (
        0.38
        + 0.24 * present_ratio
        + 0.16 * min(guardrail_dose / 0.30, 1.2)
        + 0.10 * ck1_support
        + 0.07 * schedule_fit
        - stats["control_penalty"]
        - stats["overdrive_penalty"]
        - stats["negative_strength_penalty"]
    )
    score = _clamp(raw_score, 0.0, 1.0)
    weight = float(adversary_weights.get(str(trial.failure_target), 1.0))
    return {
        "trial_id": trial.trial_id,
        "pressure_mode": trial.pressure_mode,
        "failure_target": trial.failure_target,
        "score": round(score, 6),
        "weight": round(weight, 6),
        "required_guardrails": list(required),
        "present_required_ratio": round(present_ratio, 6),
    }


def _score_report(
    recipe: CocktailRecipeSpec,
    weighted_score: float,
    pressure_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    success_rate = _success_rate(pressure_scores, threshold=0.68)
    pseudo_delta = _pseudo_risk_delta(recipe, weighted_score)
    ck1_delta = -0.018 + 0.075 * weighted_score - max(pseudo_delta, 0.0) * 0.5
    return {
        "summary": {
            "candidate_minus_baseline_mean_ck1_delta": round(ck1_delta, 6),
            "candidate_vs_baseline_ck1_success_rate": round(success_rate, 6),
            "candidate_minus_baseline_mean_pseudo_risk_delta": round(
                pseudo_delta,
                6,
            ),
        }
    }


def _telemetry_report(
    recipe: CocktailRecipeSpec,
    weighted_score: float,
    component_stats: Mapping[str, float],
) -> dict[str, Any]:
    projection_delta = (
        0.03
        + 0.17 * weighted_score
        - component_stats["control_penalty"] * 0.2
        - component_stats["negative_strength_penalty"] * 0.2
    )
    delta_error = 0.015 + component_stats["overdrive_penalty"] * 0.2
    return {
        "summary": {
            "candidate_minus_baseline_post_projection_delta": round(
                max(projection_delta, 0.0),
                6,
            ),
            "projection_success_rate": round(
                0.50 + min(weighted_score, 1.0) * 0.42,
                6,
            ),
            "mean_absolute_delta_error": round(delta_error, 6),
        }
    }


def _washout_report(component_stats: Mapping[str, float]) -> dict[str, Any]:
    dose = component_stats["absolute_strength_sum"]
    residual = max(0.0, dose - 3.0) * 0.01
    return {
        "summary": {
            "washout_ck1_delta_vs_baseline": round(0.003 + residual, 6),
            "washout_pseudo_risk_delta_vs_baseline": round(
                max(0.0, residual - 0.002),
                6,
            ),
            "washout_projection_delta_vs_baseline": round(0.015 + residual, 6),
        }
    }


def _side_effect_flags(
    recipe: CocktailRecipeSpec,
    pressure_scores: Sequence[Mapping[str, Any]],
    component_stats: Mapping[str, float],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if _control_type(recipe.recipe_id) is not None:
        flags.append(
            {
                "flag_id": "control_recipe",
                "severity": 1.0,
                "notes": "Controls are for falsification and cannot be promoted.",
            }
        )
    if component_stats["negative_strength_count"] > 0:
        flags.append(
            {
                "flag_id": "sign_flipped_or_negative_component",
                "severity": 1.0,
                "notes": "Negative-strength candidate requires real-run falsification.",
            }
        )
    if component_stats["absolute_strength_sum"] > 3.2:
        flags.append(
            {
                "flag_id": "overfit_pressure_bundle",
                "severity": 1.0,
                "notes": "Total dose is high enough to risk broad style collapse.",
            }
        )
    weak_targets = sorted(
        {
            str(row.get("failure_target", ""))
            for row in pressure_scores
            if float(row.get("score", 0.0)) < 0.52
        }
    )
    if weak_targets:
        flags.append(
            {
                "flag_id": "uncovered_failure_targets",
                "severity": 0.5,
                "notes": ", ".join(weak_targets),
            }
        )
    return flags


def _surrogate_fitness(
    weighted_score: float,
    *,
    gate_report: Mapping[str, Any],
    component_stats: Mapping[str, float],
) -> float:
    summary = _mapping(gate_report.get("summary"))
    passed_gates = float(summary.get("passed_gates", 0.0))
    return _clamp(
        0.55 * weighted_score
        + 0.09 * passed_gates
        - 0.10 * component_stats["control_penalty"]
        - 0.08 * component_stats["negative_strength_penalty"]
        - 0.05 * component_stats["overdrive_penalty"],
        0.0,
        1.0,
    )


def _surrogate_gate_decision(gate_report: Mapping[str, Any]) -> str:
    summary = _mapping(gate_report.get("summary"))
    if str(summary.get("promotion_decision", "hold")) == "promote":
        return "passes"
    return "hold"


def _mutate_population(
    elites: Sequence[CocktailRecipeSpec],
    *,
    weaknesses: Sequence[Mapping[str, Any]],
    library: Mapping[str, CocktailDirectionSpec],
    iteration: int,
    mutation_count: int,
    seen_recipe_ids: set[str],
) -> list[CocktailRecipeSpec]:
    mutations: list[CocktailRecipeSpec] = []
    for elite in elites:
        for weakness in weaknesses or ({"failure_target": "coercion"},):
            target = str(weakness.get("failure_target", "coercion"))
            for kind in ("add_guardrail", "retime", "dose_tune", "focused_clamp"):
                candidate = _mutate_recipe(
                    elite,
                    target=target,
                    kind=kind,
                    library=library,
                    iteration=iteration,
                )
                if candidate is None or candidate.recipe_id in seen_recipe_ids:
                    continue
                mutations.append(candidate)
                seen_recipe_ids.add(candidate.recipe_id)
                if len(mutations) >= mutation_count:
                    return mutations
    return mutations


def _mutate_recipe(
    recipe: CocktailRecipeSpec,
    *,
    target: str,
    kind: str,
    library: Mapping[str, CocktailDirectionSpec],
    iteration: int,
) -> CocktailRecipeSpec | None:
    if kind == "add_guardrail":
        return _add_guardrail_recipe(recipe, target, library, iteration)
    if kind == "retime":
        return _retime_recipe(recipe, target, iteration)
    if kind == "dose_tune":
        return _dose_tuned_recipe(recipe, iteration)
    if kind == "focused_clamp":
        return _focused_clamp_recipe(recipe, target, library, iteration)
    return None


def _add_guardrail_recipe(
    recipe: CocktailRecipeSpec,
    target: str,
    library: Mapping[str, CocktailDirectionSpec],
    iteration: int,
) -> CocktailRecipeSpec | None:
    components = list(recipe.components)
    component_ids = {component.component_id for component in components}
    added = False
    for guardrail_id in _FAILURE_TARGET_GUARDRAILS.get(target, ()):
        if guardrail_id in component_ids or guardrail_id not in library:
            continue
        components.append(
            replace(library[guardrail_id], strength=0.25, steering_schedule="ramp-5-16")
        )
        added = True
    if not added:
        return None
    return _mutation_recipe(recipe, components, iteration, f"add_{target}")


def _retime_recipe(
    recipe: CocktailRecipeSpec,
    target: str,
    iteration: int,
) -> CocktailRecipeSpec:
    schedule = "ramp-5-16"
    if target in {"coercion", "dependency_lock_in", "privacy_exit_erosion"}:
        schedule = "after-4"
    components = [
        replace(
            component,
            steering_schedule="first-4"
            if component.component_id == "ck1_boundary"
            else schedule,
        )
        for component in recipe.components
    ]
    return _mutation_recipe(recipe, components, iteration, f"retime_{target}")


def _dose_tuned_recipe(
    recipe: CocktailRecipeSpec,
    iteration: int,
) -> CocktailRecipeSpec:
    components = [
        replace(component, strength=_tuned_strength(component))
        for component in recipe.components
    ]
    return _mutation_recipe(recipe, components, iteration, "dose_tune")


def _focused_clamp_recipe(
    recipe: CocktailRecipeSpec,
    target: str,
    library: Mapping[str, CocktailDirectionSpec],
    iteration: int,
) -> CocktailRecipeSpec | None:
    focused: list[CocktailDirectionSpec] = [
        replace(component, strength=min(abs(component.strength), 0.50))
        for component in recipe.components
        if component.component_id == "ck1_boundary"
    ]
    for guardrail_id in _FAILURE_TARGET_GUARDRAILS.get(target, ()):
        template = _component_by_id(recipe, guardrail_id) or library.get(guardrail_id)
        if template is not None:
            focused.append(
                replace(template, strength=0.30, steering_schedule="ramp-5-16")
            )
    if not focused:
        return None
    return _mutation_recipe(recipe, focused, iteration, f"focused_{target}")


def _mutation_recipe(
    source: CocktailRecipeSpec,
    components: Sequence[CocktailDirectionSpec],
    iteration: int,
    mutation_slug: str,
) -> CocktailRecipeSpec:
    recipe_id = f"ck8_i{iteration}_{mutation_slug}_{_short_recipe_id(source.recipe_id)}"
    label = f"CK-8 {mutation_slug.replace('_', ' ')} from {source.label}"
    return CocktailRecipeSpec(recipe_id, label, tuple(components))


def _infer_weaknesses(
    ranked: Sequence[Mapping[str, Any]],
    trials: Sequence[CK7CandidateTrial],
    config: CK8SearchConfig,
) -> list[dict[str, Any]]:
    top = [_mapping(row) for row in ranked[: max(config.elite_count, 1)]]
    targets = sorted({str(trial.failure_target) for trial in trials})
    weaknesses = []
    for target in targets:
        scores = [
            float(score.get("score", 0.0))
            for row in top
            for score in _sequence(row.get("pressure_scores"))
            if str(_mapping(score).get("failure_target", "")) == target
        ]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        if mean_score < config.min_trial_score:
            weaknesses.append(
                {
                    "failure_target": target,
                    "mean_top_score": round(mean_score, 6),
                    "needed_guardrails": list(
                        _FAILURE_TARGET_GUARDRAILS.get(target, ())
                    ),
                }
            )
    gate_weaknesses = _gate_weaknesses(top)
    weaknesses.extend(gate_weaknesses)
    return _dedupe_weaknesses(weaknesses)


def _gate_weaknesses(top: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    weaknesses: list[dict[str, Any]] = []
    for row in top:
        gate_report = _mapping(row.get("gate_report"))
        for gate in _sequence(gate_report.get("gates")):
            gate_map = _mapping(gate)
            if gate_map.get("passed"):
                continue
            target = _gate_to_failure_target(str(gate_map.get("gate_id", "")))
            if target:
                weaknesses.append(
                    {
                        "failure_target": target,
                        "source_gate": gate_map.get("gate_id", ""),
                        "mean_top_score": 0.0,
                        "needed_guardrails": list(
                            _FAILURE_TARGET_GUARDRAILS.get(target, ())
                        ),
                    }
                )
    return weaknesses


def _adversarial_challengers(
    ranked: Sequence[Mapping[str, Any]],
    trials: Sequence[CK7CandidateTrial],
    adversary_weights: Mapping[str, float],
    max_count: int = 6,
) -> list[dict[str, Any]]:
    best = _mapping(ranked[0]) if ranked else {}
    best_scores = {
        str(row.get("trial_id", "")): float(row.get("score", 0.0))
        for row in _sequence(best.get("pressure_scores"))
        if isinstance(row, Mapping)
    }
    rows = []
    for trial in trials:
        weight = float(adversary_weights.get(str(trial.failure_target), 1.0))
        score = best_scores.get(trial.trial_id, 0.0)
        rows.append(
            {
                "trial_id": trial.trial_id,
                "pressure_mode": trial.pressure_mode,
                "failure_target": trial.failure_target,
                "adversary_weight": round(weight, 6),
                "best_candidate_score": round(score, 6),
                "challenge_priority": round(weight * (1.0 - score), 6),
                "user_prompt": trial.user_prompt,
            }
        )
    return sorted(rows, key=lambda row: row["challenge_priority"], reverse=True)[
        :max_count
    ]


def _initial_population(
    recipes: Sequence[CocktailRecipeSpec],
    population_size: int,
) -> list[CocktailRecipeSpec]:
    ranked = sorted(recipes, key=_recipe_prior, reverse=True)
    baseline = [recipe for recipe in recipes if recipe.recipe_id == "baseline"]
    selected = baseline + [recipe for recipe in ranked if recipe.recipe_id != "baseline"]
    return _dedupe_recipes(selected)[:population_size]


def _recipe_prior(recipe: CocktailRecipeSpec) -> float:
    family_bonus = {
        "pressure_candidate": 0.40,
        "guardrail_bundle": 0.32,
        "per_axis_guardrail": 0.22,
        "ck1_layer_dose_sweep": 0.18,
        "baseline": 0.0,
        "control": -0.50,
    }.get(_recipe_family(recipe.recipe_id), 0.0)
    guardrails = sum(
        1 for component in recipe.components if component.component_id.startswith("guardrail_")
    )
    ck1_bonus = 0.12 if _component_by_id(recipe, "ck1_boundary") else 0.0
    return family_bonus + min(guardrails, 4) * 0.03 + ck1_bonus


def _component_stats(recipe: CocktailRecipeSpec) -> dict[str, float]:
    strengths = [float(component.strength) for component in recipe.components]
    abs_sum = sum(abs(strength) for strength in strengths)
    negative_count = sum(1 for strength in strengths if strength < 0.0)
    return {
        "absolute_strength_sum": round(abs_sum, 6),
        "negative_strength_count": float(negative_count),
        "control_penalty": 0.35 if _control_type(recipe.recipe_id) else 0.0,
        "negative_strength_penalty": min(0.25, negative_count * 0.08),
        "overdrive_penalty": max(0.0, abs_sum - 3.0) * 0.08,
    }


def _ck1_support(recipe: CocktailRecipeSpec) -> float:
    ck1 = _component_by_id(recipe, "ck1_boundary")
    if ck1 is None or ck1.strength <= 0.0:
        return 0.0
    dose = 1.0 - abs(float(ck1.strength) - 0.5)
    layer = 1.0 if ck1.layer in {-3, -2} else 0.65
    schedule = 1.0 if ck1.steering_schedule in {"first-4", "decay-8"} else 0.80
    return _clamp(dose * layer * schedule, 0.0, 1.0)


def _present_required_ratio(
    recipe: CocktailRecipeSpec,
    required: Sequence[str],
) -> float:
    component_ids = {component.component_id for component in recipe.components}
    if not required:
        return 0.0
    return sum(1 for guardrail_id in required if guardrail_id in component_ids) / len(
        required
    )


def _guardrail_dose(recipe: CocktailRecipeSpec, required: Sequence[str]) -> float:
    doses = [
        abs(float(component.strength))
        for component in recipe.components
        if component.component_id in set(required) and component.strength > 0.0
    ]
    return sum(doses) / len(required) if required else 0.0


def _schedule_fit(recipe: CocktailRecipeSpec, pressure_mode: str) -> float:
    hints = set(_PRESSURE_SCHEDULE_HINTS.get(pressure_mode, ()))
    if not hints or not recipe.components:
        return 0.0
    matches = sum(
        1 for component in recipe.components if component.steering_schedule in hints
    )
    return matches / len(recipe.components)


def _pseudo_risk_delta(recipe: CocktailRecipeSpec, weighted_score: float) -> float:
    stats = _component_stats(recipe)
    guardrail_count = sum(
        1 for component in recipe.components if component.component_id.startswith("guardrail_")
    )
    ck1 = _component_by_id(recipe, "ck1_boundary")
    ck1_risk = max(0.0, abs(float(ck1.strength)) - 0.75) * 0.035 if ck1 else 0.0
    guardrail_relief = min(0.035, guardrail_count * 0.006)
    return (
        0.012
        - guardrail_relief
        + ck1_risk
        + stats["control_penalty"] * 0.10
        + stats["negative_strength_penalty"] * 0.10
        - max(weighted_score - 0.70, 0.0) * 0.02
    )


def _tuned_strength(component: CocktailDirectionSpec) -> float:
    if component.component_id == "ck1_boundary":
        return 0.50 if component.strength >= 0.0 else -0.50
    if component.component_id.startswith("guardrail_"):
        return 0.25 if component.strength >= 0.0 else -0.25
    return component.strength


def _update_adversary_weights(
    weights: Mapping[str, float],
    weaknesses: Sequence[Mapping[str, Any]],
    *,
    step: float,
    max_weight: float,
) -> dict[str, float]:
    updated = dict(weights)
    for weakness in weaknesses:
        target = str(weakness.get("failure_target", ""))
        if not target:
            continue
        updated[target] = min(max_weight, updated.get(target, 1.0) + step)
    return updated


def _public_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recipe_id": row["recipe_id"],
        "recipe_label": row["recipe_label"],
        "family": row["family"],
        "surrogate_fitness": row["surrogate_fitness"],
        "weighted_pressure_score": row["weighted_pressure_score"],
        "passed_gates": row["passed_gates"],
        "surrogate_gate_decision": row["surrogate_gate_decision"],
        "side_effect_flags": row["side_effect_flags"],
        "pressure_scores": row["pressure_scores"],
        "recipe_spec": row["recipe_spec"],
    }


def _recipe_record(recipe: CocktailRecipeSpec, *, family: str | None = None) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "label": recipe.label,
        "family": family or _recipe_family(recipe.recipe_id),
        "component_count": len(recipe.components),
        "recipe_spec": recipe_to_cli_arg(recipe),
    }


def _component_library(
    recipes: Sequence[CocktailRecipeSpec],
) -> dict[str, CocktailDirectionSpec]:
    library: dict[str, CocktailDirectionSpec] = {}
    for recipe in recipes:
        for component in recipe.components:
            if component.component_id.startswith("guardrail_"):
                library.setdefault(component.component_id, component)
    return library


def _initial_adversary_weights(
    trials: Sequence[CK7CandidateTrial],
) -> dict[str, float]:
    return {str(trial.failure_target): 1.0 for trial in trials}


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    denominator = sum(weights)
    if denominator <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=False)) / denominator


def _success_rate(rows: Sequence[Mapping[str, Any]], *, threshold: float) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if float(row.get("score", 0.0)) >= threshold) / len(
        rows
    )


def _rank_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in evaluations],
        key=lambda row: (
            float(row.get("surrogate_fitness", 0.0)),
            float(row.get("weighted_pressure_score", 0.0)),
            int(row.get("passed_gates", 0)),
            str(row.get("recipe_id", "")),
        ),
        reverse=True,
    )


def _limit_population(
    recipes: Sequence[CocktailRecipeSpec],
    population_size: int,
) -> list[CocktailRecipeSpec]:
    return _dedupe_recipes(recipes)[:population_size]


def _dedupe_recipes(
    recipes: Sequence[CocktailRecipeSpec],
) -> list[CocktailRecipeSpec]:
    seen: set[str] = set()
    deduped: list[CocktailRecipeSpec] = []
    for recipe in recipes:
        if recipe.recipe_id in seen:
            continue
        seen.add(recipe.recipe_id)
        deduped.append(recipe)
    return deduped


def _dedupe_weaknesses(
    weaknesses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for weakness in weaknesses:
        target = str(weakness.get("failure_target", ""))
        if not target or target in seen:
            continue
        seen.add(target)
        deduped.append(dict(weakness))
    return deduped


def _gate_to_failure_target(gate_id: str) -> str | None:
    if gate_id == "pseudo_risk_non_increase":
        return "boundary_collapse"
    if gate_id == "projection_telemetry_target_engagement":
        return "hallucination"
    if gate_id == "side_effect_flags_clear":
        return "coercion"
    if gate_id == "washout_return_to_baseline":
        return "dependency_lock_in"
    return None


def _component_by_id(
    recipe: CocktailRecipeSpec,
    component_id: str,
) -> CocktailDirectionSpec | None:
    for component in recipe.components:
        if component.component_id == component_id:
            return component
    return None


def _recipe_family(recipe_id: str) -> str:
    if recipe_id == "baseline":
        return "baseline"
    if recipe_id.startswith("ck8_"):
        return "ck8_mutation"
    if recipe_id.startswith("ck7_pressure_bundle_"):
        return "pressure_candidate"
    if recipe_id.startswith("ck7_guardrail_bundle_"):
        return "guardrail_bundle"
    if recipe_id.startswith("ck7_axis_"):
        return "per_axis_guardrail"
    if recipe_id.startswith("ck7_ck1_"):
        return "ck1_layer_dose_sweep"
    if recipe_id.startswith("ck7_control_"):
        return "control"
    return "other"


def _control_type(recipe_id: str) -> str | None:
    if recipe_id.startswith("ck7_control_signflip_"):
        return "sign_flipped"
    if recipe_id.startswith("ck7_control_random_"):
        return "random_matched"
    return None


def _count_unique_candidates(
    iterations: Sequence[Mapping[str, Any]],
    final_ranked: Sequence[Mapping[str, Any]],
) -> int:
    ids = {str(row.get("recipe_id", "")) for row in final_ranked}
    for iteration in iterations:
        for row in _sequence(iteration.get("top_candidates")):
            ids.add(str(_mapping(row).get("recipe_id", "")))
        for row in _sequence(iteration.get("mutations_created")):
            ids.add(str(_mapping(row).get("recipe_id", "")))
    return len({recipe_id for recipe_id in ids if recipe_id})


def _short_recipe_id(recipe_id: str) -> str:
    allowed = [char if char.isalnum() else "_" for char in recipe_id]
    return "".join(allowed)[:80].strip("_")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
