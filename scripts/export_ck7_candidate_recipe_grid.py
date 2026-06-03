"""Export CK-7 pressure candidate recipe specs without running Modal."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.ck3_cocktail import (
    CocktailDirectionSpec,
    CocktailRecipeSpec,
)
from social_cohesion_vectors.experiments.trait_axes import (
    TraitAxis,
    canonical_trait_axes,
)

try:
    from scripts.export_ck45_guardrail_vector_grid import (
        DEFAULT_CK45_GUARDRAIL_AXES,
        build_guardrail_artifact_specs,
        parse_artifact_overrides,
        recipe_to_cli_arg,
    )
    from scripts.run_ck4_scheduled_modal_cocktail import validate_schedule
except ModuleNotFoundError:
    _CK45_SPEC = importlib.util.spec_from_file_location(
        "_ck45_guardrail_vector_grid",
        Path(__file__).with_name("export_ck45_guardrail_vector_grid.py"),
    )
    if _CK45_SPEC is None or _CK45_SPEC.loader is None:
        raise ImportError("Could not load CK-4.5 guardrail grid exporter.")
    _CK45_MODULE = importlib.util.module_from_spec(_CK45_SPEC)
    sys.modules[_CK45_SPEC.name] = _CK45_MODULE
    _CK45_SPEC.loader.exec_module(_CK45_MODULE)

    _CK4_SPEC = importlib.util.spec_from_file_location(
        "_ck4_scheduled_modal_cocktail",
        Path(__file__).with_name("run_ck4_scheduled_modal_cocktail.py"),
    )
    if _CK4_SPEC is None or _CK4_SPEC.loader is None:
        raise ImportError("Could not load CK-4 scheduled cocktail helpers.")
    _CK4_MODULE = importlib.util.module_from_spec(_CK4_SPEC)
    sys.modules[_CK4_SPEC.name] = _CK4_MODULE
    _CK4_SPEC.loader.exec_module(_CK4_MODULE)

    DEFAULT_CK45_GUARDRAIL_AXES = _CK45_MODULE.DEFAULT_CK45_GUARDRAIL_AXES
    build_guardrail_artifact_specs = _CK45_MODULE.build_guardrail_artifact_specs
    parse_artifact_overrides = _CK45_MODULE.parse_artifact_overrides
    recipe_to_cli_arg = _CK45_MODULE.recipe_to_cli_arg
    validate_schedule = _CK4_MODULE.validate_schedule

DEFAULT_CK7_PRESSURE_AXES = (
    "repair_vs_harm_denial",
    "reciprocity_vs_extraction",
    "truth_vs_deception",
    "autonomy_vs_coercion",
    "principled_respect_vs_sycophancy",
    "constructive_dissent_vs_conformity",
    "manipulation_resistance_vs_persuasion_capture",
    "privacy_exit_vs_surveillance_lock_in",
)
DEFAULT_CK7_CK1_LAYERS = (-4, -3, -2, -1)
DEFAULT_CK7_CK1_STRENGTHS = (0.25, 0.5, 1.0)
DEFAULT_CK7_GUARDRAIL_STRENGTHS = (0.15, 0.25)
TARGET_HIDDEN_SIZE = 896


@dataclass(frozen=True)
class CK7ScheduleSpec:
    """A named CK-7 timing pair for CK-1 and guardrail components."""

    schedule_id: str
    label: str
    ck1_schedule: str
    guardrail_schedule: str


class GuardrailArtifact(Protocol):
    """Structural type for CK-4.5 guardrail artifact specs."""

    @property
    def axis_id(self) -> str:
        """Trait-axis id for this guardrail artifact."""
        ...

    @property
    def path(self) -> Path:
        """Local direction artifact path."""
        ...

    @property
    def layer(self) -> int:
        """Activation layer for the guardrail artifact."""
        ...

    @property
    def strength(self) -> float:
        """Default guardrail strength."""
        ...


DEFAULT_CK7_SCHEDULES: tuple[CK7ScheduleSpec, ...] = (
    CK7ScheduleSpec(
        schedule_id="steady_constant",
        label="steady constant",
        ck1_schedule="constant",
        guardrail_schedule="constant",
    ),
    CK7ScheduleSpec(
        schedule_id="frontload_then_clamp",
        label="front-load CK-1 then clamp guardrails",
        ck1_schedule="first-4",
        guardrail_schedule="after-4",
    ),
    CK7ScheduleSpec(
        schedule_id="decay_then_ramp",
        label="CK-1 decay then guardrail ramp",
        ck1_schedule="decay-8",
        guardrail_schedule="ramp-5-16",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    axes = select_ck7_axes(args.axis)
    schedules = parse_schedule_specs(args.schedule)
    ck1_layers = tuple(args.ck1_layer or DEFAULT_CK7_CK1_LAYERS)
    ck1_strengths = tuple(args.ck1_strength or DEFAULT_CK7_CK1_STRENGTHS)
    guardrail_strengths = tuple(
        args.guardrail_strength or DEFAULT_CK7_GUARDRAIL_STRENGTHS
    )
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=args.direction_root,
        direction_template=args.direction_template,
        artifact_overrides=parse_artifact_overrides(args.artifact),
        layer=args.guardrail_layer,
        strength=guardrail_strengths[0],
    )
    recipes = build_ck7_candidate_recipes(
        artifacts,
        ck1_direction=args.ck1_direction,
        ck1_layers=ck1_layers,
        ck1_strengths=ck1_strengths,
        guardrail_strengths=guardrail_strengths,
        schedules=schedules,
        include_controls=not args.no_controls,
        random_control_root=args.random_control_root,
        random_seed=args.random_seed,
    )
    payload = write_ck7_candidate_grid(
        axes,
        artifacts,
        recipes,
        ck1_direction=args.ck1_direction,
        ck1_layers=ck1_layers,
        ck1_strengths=ck1_strengths,
        guardrail_strengths=guardrail_strengths,
        schedules=schedules,
        include_controls=not args.no_controls,
        random_control_root=args.random_control_root,
        random_seed=args.random_seed,
        json_output=args.output,
        recipe_specs_output=args.recipe_specs_output,
    )
    print(
        "CK-7 candidate recipe grid dry run: "
        f"axes={len(axes)} recipes={len(recipes)} controls={payload['controls_enabled']} "
        f"json={args.output}"
    )
    for recipe in _sequence(payload.get("recipes")):
        recipe_map = _mapping(recipe)
        print(f"--recipe {recipe_map['recipe_spec']}")
    return 0


def select_ck7_axes(axis_ids: Sequence[str] | None = None) -> tuple[TraitAxis, ...]:
    """Select CK-7 axes for boundary-preserving prosociality under pressure."""

    requested = tuple(axis_ids or DEFAULT_CK7_PRESSURE_AXES)
    axes_by_id = {axis.axis_id: axis for axis in canonical_trait_axes()}
    missing = sorted(set(requested).difference(axes_by_id))
    if missing:
        raise ValueError(f"Unknown trait axis id(s): {', '.join(missing)}")
    return tuple(axes_by_id[axis_id] for axis_id in requested)


def parse_schedule_specs(
    values: Sequence[str] | None = None,
) -> tuple[CK7ScheduleSpec, ...]:
    """Parse optional ``schedule_id=ck1_schedule:guardrail_schedule`` pairs."""

    if not values:
        return DEFAULT_CK7_SCHEDULES
    schedules: list[CK7ScheduleSpec] = []
    seen: set[str] = set()
    for value in values:
        schedule_id, ck1_schedule, guardrail_schedule = _parse_schedule_value(value)
        if schedule_id in seen:
            raise ValueError(f"Duplicate CK-7 schedule id: {schedule_id}")
        seen.add(schedule_id)
        schedules.append(
            CK7ScheduleSpec(
                schedule_id=schedule_id,
                label=schedule_id.replace("_", " "),
                ck1_schedule=ck1_schedule,
                guardrail_schedule=guardrail_schedule,
            )
        )
    return tuple(schedules)


def build_ck7_candidate_recipes(
    artifacts: Sequence[GuardrailArtifact],
    *,
    ck1_direction: Path,
    ck1_layers: Sequence[int] = DEFAULT_CK7_CK1_LAYERS,
    ck1_strengths: Sequence[float] = DEFAULT_CK7_CK1_STRENGTHS,
    guardrail_strengths: Sequence[float] = DEFAULT_CK7_GUARDRAIL_STRENGTHS,
    schedules: Sequence[CK7ScheduleSpec] = DEFAULT_CK7_SCHEDULES,
    include_controls: bool = True,
    random_control_root: Path | None = None,
    random_seed: int = 7,
) -> list[CocktailRecipeSpec]:
    """Build CK-7 baseline, candidate, and dry-run control recipes."""

    recipes = [CocktailRecipeSpec("baseline", "Baseline", ())]
    recipes.extend(
        _axis_recipes(
            artifacts,
            schedules=schedules,
            guardrail_strengths=guardrail_strengths,
        )
    )
    recipes.extend(
        _bundle_recipes(
            artifacts,
            schedules=schedules,
            guardrail_strengths=guardrail_strengths,
        )
    )
    recipes.extend(
        _ck1_sweep_recipes(
            ck1_direction,
            ck1_layers=ck1_layers,
            ck1_strengths=ck1_strengths,
            schedules=schedules,
        )
    )
    pressure_recipes = _pressure_bundle_recipes(
        artifacts,
        ck1_direction=ck1_direction,
        ck1_layers=ck1_layers,
        ck1_strengths=ck1_strengths,
        guardrail_strengths=guardrail_strengths,
        schedules=schedules,
    )
    recipes.extend(pressure_recipes)
    if include_controls:
        control_root = random_control_root or ck1_direction.parent / "ck7_controls"
        recipes.extend(
            _control_recipes(
                artifacts,
                ck1_direction=ck1_direction,
                ck1_layers=ck1_layers,
                ck1_strengths=ck1_strengths,
                guardrail_strengths=guardrail_strengths,
                schedules=schedules,
                random_control_root=control_root,
                random_seed=random_seed,
            )
        )
    return recipes


def write_ck7_candidate_grid(
    axes: Sequence[TraitAxis],
    artifacts: Sequence[GuardrailArtifact],
    recipes: Sequence[CocktailRecipeSpec],
    *,
    ck1_direction: Path,
    ck1_layers: Sequence[int],
    ck1_strengths: Sequence[float],
    guardrail_strengths: Sequence[float],
    schedules: Sequence[CK7ScheduleSpec],
    include_controls: bool,
    random_control_root: Path,
    random_seed: int,
    json_output: Path,
    recipe_specs_output: Path | None = None,
) -> dict[str, object]:
    """Write dry-run CK-7 candidate recipe specs."""

    payload: dict[str, object] = {
        "experiment": "ck7_candidate_recipe_grid",
        "dry_run": True,
        "description": (
            "Boundary-preserving prosociality candidate grid under pressure: "
            "helpful, principled, truth-calibrated, and autonomy-preserving "
            "without sycophancy, hallucination, coercion, or dependency."
        ),
        "claim_boundary": (
            "Recipe specs only. No Modal execution, no loaded vector files, and "
            "no human, neural, biological, clinical, or behavioral effect claims."
        ),
        "ck45_lineage": {
            "source": "CK-4.5 per-axis guardrail recipe grid",
            "inherited_axes": [
                axis_id
                for axis_id in DEFAULT_CK45_GUARDRAIL_AXES
                if axis_id in {axis.axis_id for axis in axes}
            ],
        },
        "target_hidden_size": TARGET_HIDDEN_SIZE,
        "controls_enabled": include_controls,
        "random_control_contract": {
            "root": str(random_control_root),
            "seed": random_seed,
            "hidden_size": TARGET_HIDDEN_SIZE,
            "note": (
                "Random-control paths are placeholders in this dry run and must "
                "be materialized as matched unit-norm vectors before execution."
            ),
        },
        "sweeps": {
            "ck1_direction": str(ck1_direction),
            "ck1_layers": list(ck1_layers),
            "ck1_strengths": list(ck1_strengths),
            "guardrail_strengths": list(guardrail_strengths),
            "schedules": [
                {
                    "schedule_id": schedule.schedule_id,
                    "label": schedule.label,
                    "ck1_schedule": schedule.ck1_schedule,
                    "guardrail_schedule": schedule.guardrail_schedule,
                }
                for schedule in schedules
            ],
        },
        "axes": [
            {
                "axis_id": axis.axis_id,
                "positive_pole": axis.positive_pole,
                "negative_pole": axis.negative_pole,
                "role": _axis_role(axis.axis_id),
                "contrasts": [contrast.contrast_id for contrast in axis.contrasts],
            }
            for axis in axes
        ],
        "artifacts": [
            {
                "axis_id": artifact.axis_id,
                "component_id": _guardrail_component_id(artifact.axis_id),
                "path": str(artifact.path),
                "layer": artifact.layer,
                "default_strength": artifact.strength,
            }
            for artifact in artifacts
        ],
        "recipes": [_recipe_record(recipe) for recipe in recipes],
        "control_specs": _control_specs(include_controls),
        "validation_gates": [
            "CK-1 layer/dose candidates beat baseline on CK-1 score under pressure prompts",
            "per-axis and bundled guardrails do not increase pseudo-attunement risk",
            "truth and uncertainty wording improves without unsupported certainty",
            "autonomy, privacy, dissent, and exit-right language remains available",
            "sign-flipped and random controls fail or move in the wrong direction",
            "qualitative review finds no sycophancy, hallucination, coercion, or dependency",
        ],
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if recipe_specs_output is not None:
        recipe_specs_output.parent.mkdir(parents=True, exist_ok=True)
        recipe_specs_output.write_text(
            "\n".join(recipe_to_cli_arg(recipe) for recipe in recipes) + "\n",
            encoding="utf-8",
        )
    return payload


def _axis_recipes(
    artifacts: Sequence[GuardrailArtifact],
    *,
    schedules: Sequence[CK7ScheduleSpec],
    guardrail_strengths: Sequence[float],
) -> list[CocktailRecipeSpec]:
    recipes: list[CocktailRecipeSpec] = []
    for artifact in artifacts:
        for strength in guardrail_strengths:
            for schedule in schedules:
                recipes.append(
                    CocktailRecipeSpec(
                        recipe_id=(
                            "ck7_axis_"
                            f"{artifact.axis_id}_{schedule.schedule_id}_g{_dose_slug(strength)}"
                        ),
                        label=(
                            f"{artifact.axis_id} {schedule.label} "
                            f"guardrail {strength:g}"
                        ),
                        components=(
                            _guardrail_component(
                                artifact,
                                strength=strength,
                                schedule=schedule.guardrail_schedule,
                            ),
                        ),
                    )
                )
    return recipes


def _bundle_recipes(
    artifacts: Sequence[GuardrailArtifact],
    *,
    schedules: Sequence[CK7ScheduleSpec],
    guardrail_strengths: Sequence[float],
) -> list[CocktailRecipeSpec]:
    recipes: list[CocktailRecipeSpec] = []
    for strength in guardrail_strengths:
        for schedule in schedules:
            recipes.append(
                CocktailRecipeSpec(
                    recipe_id=(
                        f"ck7_guardrail_bundle_{schedule.schedule_id}_g"
                        f"{_dose_slug(strength)}"
                    ),
                    label=f"CK-7 guardrail bundle {schedule.label} {strength:g}",
                    components=tuple(
                        _guardrail_component(
                            artifact,
                            strength=strength,
                            schedule=schedule.guardrail_schedule,
                        )
                        for artifact in artifacts
                    ),
                )
            )
    return recipes


def _ck1_sweep_recipes(
    ck1_direction: Path,
    *,
    ck1_layers: Sequence[int],
    ck1_strengths: Sequence[float],
    schedules: Sequence[CK7ScheduleSpec],
) -> list[CocktailRecipeSpec]:
    recipes: list[CocktailRecipeSpec] = []
    for layer in ck1_layers:
        for strength in ck1_strengths:
            for schedule in schedules:
                recipes.append(
                    CocktailRecipeSpec(
                        recipe_id=(
                            f"ck7_ck1_l{_layer_slug(layer)}_d{_dose_slug(strength)}_"
                            f"{schedule.schedule_id}"
                        ),
                        label=(
                            f"CK-1 layer {layer} dose {strength:g} "
                            f"{schedule.ck1_schedule}"
                        ),
                        components=(
                            _ck1_component(
                                ck1_direction,
                                layer=layer,
                                strength=strength,
                                schedule=schedule.ck1_schedule,
                            ),
                        ),
                    )
                )
    return recipes


def _pressure_bundle_recipes(
    artifacts: Sequence[GuardrailArtifact],
    *,
    ck1_direction: Path,
    ck1_layers: Sequence[int],
    ck1_strengths: Sequence[float],
    guardrail_strengths: Sequence[float],
    schedules: Sequence[CK7ScheduleSpec],
) -> list[CocktailRecipeSpec]:
    recipes: list[CocktailRecipeSpec] = []
    for layer in ck1_layers:
        for ck1_strength in ck1_strengths:
            for guardrail_strength in guardrail_strengths:
                for schedule in schedules:
                    recipes.append(
                        _pressure_recipe(
                            artifacts,
                            ck1_direction=ck1_direction,
                            layer=layer,
                            ck1_strength=ck1_strength,
                            guardrail_strength=guardrail_strength,
                            schedule=schedule,
                            control_prefix=None,
                        )
                    )
    return recipes


def _control_recipes(
    artifacts: Sequence[GuardrailArtifact],
    *,
    ck1_direction: Path,
    ck1_layers: Sequence[int],
    ck1_strengths: Sequence[float],
    guardrail_strengths: Sequence[float],
    schedules: Sequence[CK7ScheduleSpec],
    random_control_root: Path,
    random_seed: int,
) -> list[CocktailRecipeSpec]:
    recipes: list[CocktailRecipeSpec] = []
    for layer in ck1_layers:
        for ck1_strength in ck1_strengths:
            for guardrail_strength in guardrail_strengths:
                for schedule in schedules:
                    recipes.append(
                        _pressure_recipe(
                            artifacts,
                            ck1_direction=ck1_direction,
                            layer=layer,
                            ck1_strength=-ck1_strength,
                            guardrail_strength=-guardrail_strength,
                            schedule=schedule,
                            control_prefix="signflip",
                        )
                    )
                    recipes.append(
                        _random_matched_recipe(
                            artifacts,
                            layer=layer,
                            ck1_strength=ck1_strength,
                            guardrail_strength=guardrail_strength,
                            schedule=schedule,
                            random_control_root=random_control_root,
                            random_seed=random_seed,
                        )
                    )
    return recipes


def _pressure_recipe(
    artifacts: Sequence[GuardrailArtifact],
    *,
    ck1_direction: Path,
    layer: int,
    ck1_strength: float,
    guardrail_strength: float,
    schedule: CK7ScheduleSpec,
    control_prefix: str | None,
) -> CocktailRecipeSpec:
    recipe_prefix = "ck7_pressure_bundle"
    label_prefix = "CK-7 pressure bundle"
    if control_prefix is not None:
        recipe_prefix = f"ck7_control_{control_prefix}"
        label_prefix = f"CK-7 {control_prefix} control"
    recipe_id = (
        f"{recipe_prefix}_l{_layer_slug(layer)}_ck1_d{_dose_slug(ck1_strength)}_"
        f"g{_dose_slug(guardrail_strength)}_{schedule.schedule_id}"
    )
    return CocktailRecipeSpec(
        recipe_id=recipe_id,
        label=(
            f"{label_prefix} layer {layer} CK1 {ck1_strength:g} "
            f"guardrail {guardrail_strength:g} {schedule.label}"
        ),
        components=(
            _ck1_component(
                ck1_direction,
                layer=layer,
                strength=ck1_strength,
                schedule=schedule.ck1_schedule,
            ),
            *(
                _guardrail_component(
                    artifact,
                    strength=guardrail_strength,
                    schedule=schedule.guardrail_schedule,
                )
                for artifact in artifacts
            ),
        ),
    )


def _random_matched_recipe(
    artifacts: Sequence[GuardrailArtifact],
    *,
    layer: int,
    ck1_strength: float,
    guardrail_strength: float,
    schedule: CK7ScheduleSpec,
    random_control_root: Path,
    random_seed: int,
) -> CocktailRecipeSpec:
    recipe_id = (
        f"ck7_control_random_l{_layer_slug(layer)}_ck1_d{_dose_slug(ck1_strength)}_"
        f"g{_dose_slug(guardrail_strength)}_{schedule.schedule_id}"
    )
    return CocktailRecipeSpec(
        recipe_id=recipe_id,
        label=(
            f"CK-7 random matched control layer {layer} CK1 {ck1_strength:g} "
            f"guardrail {guardrail_strength:g} {schedule.label}"
        ),
        components=(
            _random_component(
                "random_ck1_boundary",
                random_control_root / f"ck1_random_seed{random_seed}_h896.npz",
                layer=layer,
                strength=ck1_strength,
                schedule=schedule.ck1_schedule,
            ),
            *(
                _random_component(
                    f"random_guardrail_{artifact.axis_id}",
                    random_control_root
                    / f"{artifact.axis_id}_random_seed{random_seed}_h896.npz",
                    layer=artifact.layer,
                    strength=guardrail_strength,
                    schedule=schedule.guardrail_schedule,
                )
                for artifact in artifacts
            ),
        ),
    )


def _recipe_record(recipe: CocktailRecipeSpec) -> dict[str, object]:
    return {
        "recipe_id": recipe.recipe_id,
        "label": recipe.label,
        "family": _recipe_family(recipe.recipe_id),
        "control_type": _control_type(recipe.recipe_id),
        "recipe_spec": recipe_to_cli_arg(recipe),
        "components": [
            {
                "component_id": component.component_id,
                "path": str(component.path),
                "layer": component.layer,
                "strength": component.strength,
                "hook_site": component.hook_site,
                "steering_position": component.steering_position,
                "steering_timing": component.steering_timing,
                "steering_schedule": component.steering_schedule,
            }
            for component in recipe.components
        ],
    }


def _control_specs(include_controls: bool) -> list[dict[str, str]]:
    if not include_controls:
        return []
    return [
        {
            "control_type": "sign_flipped",
            "description": (
                "Same CK-1 and guardrail artifacts as the pressure bundle with "
                "all strengths sign-flipped."
            ),
        },
        {
            "control_type": "random_matched",
            "description": (
                "Placeholder random vectors matched by component count, layer, "
                "schedule, strength, and hidden-size contract."
            ),
        },
    ]


def _guardrail_component(
    artifact: GuardrailArtifact,
    *,
    strength: float,
    schedule: str,
) -> CocktailDirectionSpec:
    return CocktailDirectionSpec(
        component_id=_guardrail_component_id(artifact.axis_id),
        path=artifact.path,
        layer=artifact.layer,
        strength=strength,
        steering_schedule=schedule,
    )


def _ck1_component(
    path: Path,
    *,
    layer: int,
    strength: float,
    schedule: str,
) -> CocktailDirectionSpec:
    return CocktailDirectionSpec(
        component_id="ck1_boundary",
        path=path,
        layer=layer,
        strength=strength,
        steering_schedule=schedule,
    )


def _random_component(
    component_id: str,
    path: Path,
    *,
    layer: int,
    strength: float,
    schedule: str,
) -> CocktailDirectionSpec:
    return CocktailDirectionSpec(
        component_id=component_id,
        path=path,
        layer=layer,
        strength=strength,
        steering_schedule=schedule,
    )


def _axis_role(axis_id: str) -> str:
    roles = {
        "repair_vs_harm_denial": "helpful_repair",
        "reciprocity_vs_extraction": "helpful_fairness",
        "truth_vs_deception": "truth_calibration",
        "autonomy_vs_coercion": "autonomy_preservation",
        "principled_respect_vs_sycophancy": "principled_non_sycophancy",
        "constructive_dissent_vs_conformity": "dissent_preservation",
        "manipulation_resistance_vs_persuasion_capture": "anti_coercion",
        "privacy_exit_vs_surveillance_lock_in": "anti_dependency",
    }
    return roles.get(axis_id, "guardrail")


def _recipe_family(recipe_id: str) -> str:
    if recipe_id == "baseline":
        return "baseline"
    if recipe_id.startswith("ck7_axis_"):
        return "per_axis_guardrail"
    if recipe_id.startswith("ck7_guardrail_bundle_"):
        return "guardrail_bundle"
    if recipe_id.startswith("ck7_ck1_"):
        return "ck1_layer_dose_sweep"
    if recipe_id.startswith("ck7_pressure_bundle_"):
        return "pressure_candidate"
    if recipe_id.startswith("ck7_control_"):
        return "control"
    return "other"


def _control_type(recipe_id: str) -> str | None:
    if recipe_id.startswith("ck7_control_signflip_"):
        return "sign_flipped"
    if recipe_id.startswith("ck7_control_random_"):
        return "random_matched"
    return None


def _guardrail_component_id(axis_id: str) -> str:
    return f"guardrail_{axis_id}"


def _layer_slug(layer: int) -> str:
    return f"m{abs(layer)}" if layer < 0 else str(layer)


def _dose_slug(value: float) -> str:
    sign = "neg_" if value < 0 else ""
    return sign + f"{abs(value):g}".replace(".", "p")


def _parse_schedule_value(value: str) -> tuple[str, str, str]:
    if "=" not in value or ":" not in value:
        raise ValueError(
            "CK-7 schedules must look like schedule_id=ck1_schedule:guardrail_schedule."
        )
    schedule_id, payload = value.split("=", 1)
    ck1_schedule, guardrail_schedule = payload.split(":", 1)
    schedule_id = schedule_id.strip().replace("-", "_")
    ck1_schedule = ck1_schedule.strip()
    guardrail_schedule = guardrail_schedule.strip()
    if not schedule_id:
        raise ValueError("CK-7 schedule id cannot be empty.")
    validate_schedule(ck1_schedule)
    validate_schedule(guardrail_schedule)
    return schedule_id, ck1_schedule, guardrail_schedule


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        action="append",
        choices=[axis.axis_id for axis in canonical_trait_axes()],
        help="Trait axis to include. Defaults to CK-7 pressure axes.",
    )
    parser.add_argument(
        "--direction-root",
        type=Path,
        default=config.paths.vectors / "ck45_guardrails",
        help="Root for inferred per-axis guardrail direction artifacts.",
    )
    parser.add_argument(
        "--direction-template",
        default="{axis_id}_direction.npz",
        help="Filename template under --direction-root. Supports {axis_id}.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        help="Override one inferred artifact path with axis_id=/path/vector.npz.",
    )
    parser.add_argument("--guardrail-layer", type=int, default=-1)
    parser.add_argument(
        "--guardrail-strength",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ck1-direction",
        type=Path,
        default=config.paths.vectors / "ck1_boundary_l2_direction.npz",
    )
    parser.add_argument(
        "--ck1-layer",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ck1-strength",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--schedule",
        action="append",
        default=None,
        help="Timing pair as schedule_id=ck1_schedule:guardrail_schedule.",
    )
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument(
        "--random-control-root",
        type=Path,
        default=config.paths.vectors / "ck7_controls",
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.reports / "ck7_candidate_recipe_grid.json",
    )
    parser.add_argument("--recipe-specs-output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
