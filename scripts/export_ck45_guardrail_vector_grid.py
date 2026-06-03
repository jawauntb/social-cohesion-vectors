"""Export CK-4.5 per-axis guardrail vector recipe specs without running Modal."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.ck3_cocktail import (
    CocktailDirectionSpec,
    CocktailRecipeSpec,
)
from social_cohesion_vectors.experiments.trait_axes import (
    TraitAxis,
    canonical_trait_axes,
)

DEFAULT_CK45_GUARDRAIL_AXES = (
    "principled_respect_vs_sycophancy",
    "truth_vs_deception",
    "manipulation_resistance_vs_persuasion_capture",
    "privacy_exit_vs_surveillance_lock_in",
)
DEFAULT_SCHEDULES = ("constant", "ramp-5-16")


@dataclass(frozen=True)
class GuardrailArtifactSpec:
    """A per-axis direction artifact before CK-3 recipe serialization."""

    axis_id: str
    path: Path
    layer: int
    strength: float


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    axes = select_guardrail_axes(args.axis)
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=args.direction_root,
        direction_template=args.direction_template,
        artifact_overrides=parse_artifact_overrides(args.artifact),
        layer=args.layer,
        strength=args.strength,
    )
    recipes = build_ck45_guardrail_recipes(
        artifacts,
        schedules=args.schedule or DEFAULT_SCHEDULES,
        ck1_direction=args.ck1_direction,
        ck1_layer=args.ck1_layer,
        ck1_strength=args.ck1_strength,
        ck1_schedule=args.ck1_schedule,
    )
    payload = write_ck45_guardrail_grid(
        axes,
        artifacts,
        recipes,
        json_output=args.output,
        recipe_specs_output=args.recipe_specs_output,
    )
    print(
        "CK-4.5 guardrail vector grid dry run: "
        f"axes={len(axes)} recipes={len(recipes)} json={args.output}"
    )
    for recipe in _sequence(payload.get("recipes")):
        recipe_map = _mapping(recipe)
        print(f"--recipe {recipe_map['recipe_spec']}")
    return 0


def select_guardrail_axes(axis_ids: Sequence[str] | None = None) -> tuple[TraitAxis, ...]:
    """Select canonical trait axes for CK-4.5 guardrail vectorization."""

    requested = tuple(axis_ids or DEFAULT_CK45_GUARDRAIL_AXES)
    axes_by_id = {axis.axis_id: axis for axis in canonical_trait_axes()}
    missing = sorted(set(requested).difference(axes_by_id))
    if missing:
        raise ValueError(f"Unknown trait axis id(s): {', '.join(missing)}")
    return tuple(axes_by_id[axis_id] for axis_id in requested)


def parse_artifact_overrides(values: Sequence[str] | None) -> dict[str, Path]:
    """Parse optional ``axis_id=path`` artifact overrides."""

    overrides: dict[str, Path] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError("Artifact overrides must look like axis_id=path.")
        axis_id, path = value.split("=", 1)
        axis_id = axis_id.strip()
        if not axis_id:
            raise ValueError("Artifact override axis id cannot be empty.")
        if axis_id in overrides:
            raise ValueError(f"Duplicate artifact override: {axis_id}")
        overrides[axis_id] = Path(path.strip())
    return overrides


def build_guardrail_artifact_specs(
    axes: Sequence[TraitAxis],
    *,
    direction_root: Path,
    direction_template: str,
    artifact_overrides: Mapping[str, Path] | None = None,
    layer: int = -1,
    strength: float = 0.25,
) -> list[GuardrailArtifactSpec]:
    """Build one positive-pole direction artifact spec per guardrail axis."""

    overrides = artifact_overrides or {}
    specs: list[GuardrailArtifactSpec] = []
    for axis in axes:
        path = overrides.get(
            axis.axis_id,
            direction_root / direction_template.format(axis_id=axis.axis_id),
        )
        specs.append(
            GuardrailArtifactSpec(
                axis_id=axis.axis_id,
                path=path,
                layer=layer,
                strength=strength,
            )
        )
    return specs


def build_ck45_guardrail_recipes(
    artifacts: Sequence[GuardrailArtifactSpec],
    *,
    schedules: Sequence[str] = DEFAULT_SCHEDULES,
    ck1_direction: Path | None = None,
    ck1_layer: int = -2,
    ck1_strength: float = 1.0,
    ck1_schedule: str = "decay-8",
) -> list[CocktailRecipeSpec]:
    """Build baseline, per-axis, bundle, and optional CK-1 clamp recipes."""

    recipes = [CocktailRecipeSpec("baseline", "Baseline", ())]
    for artifact in artifacts:
        for schedule in schedules:
            recipes.append(
                CocktailRecipeSpec(
                    recipe_id=f"guardrail_{artifact.axis_id}_{_schedule_slug(schedule)}",
                    label=f"{artifact.axis_id} {schedule}",
                    components=(
                        _guardrail_component(artifact, schedule=schedule),
                    ),
                )
            )
    if artifacts:
        recipes.append(
            CocktailRecipeSpec(
                recipe_id="guardrail_axis_bundle_constant",
                label="Per-axis guardrail bundle constant",
                components=tuple(
                    _guardrail_component(artifact, schedule="constant")
                    for artifact in artifacts
                ),
            )
        )
        recipes.append(
            CocktailRecipeSpec(
                recipe_id="guardrail_axis_bundle_ramp",
                label="Per-axis guardrail bundle ramp",
                components=tuple(
                    _guardrail_component(artifact, schedule="ramp-5-16")
                    for artifact in artifacts
                ),
            )
        )
    if ck1_direction is not None and artifacts:
        recipes.append(
            CocktailRecipeSpec(
                recipe_id="ck1_decay_per_axis_clamp",
                label="CK-1 decay plus per-axis guardrail clamp",
                components=(
                    CocktailDirectionSpec(
                        component_id="ck1_boundary",
                        path=ck1_direction,
                        layer=ck1_layer,
                        strength=ck1_strength,
                        steering_schedule=ck1_schedule,
                    ),
                    *(
                        _guardrail_component(artifact, schedule="ramp-5-16")
                        for artifact in artifacts
                    ),
                ),
            )
        )
    return recipes


def write_ck45_guardrail_grid(
    axes: Sequence[TraitAxis],
    artifacts: Sequence[GuardrailArtifactSpec],
    recipes: Sequence[CocktailRecipeSpec],
    *,
    json_output: Path,
    recipe_specs_output: Path | None = None,
) -> dict[str, object]:
    """Write dry-run CK-4.5 guardrail vector grid specs."""

    payload: dict[str, object] = {
        "experiment": "ck45_guardrail_vector_grid",
        "description": (
            "Compute-only CK-4.5 recipe grid that replaces broad CK-4 proxy "
            "guardrails with named per-axis trait guardrail vectors."
        ),
        "claim_boundary": (
            "Recipe specs only. No Modal execution, biology, human behavior, "
            "or neural validation is claimed."
        ),
        "modal_runtime_note": (
            "Use a persistent Hugging Face/model cache volume or prefetch step "
            "before the first generation batch to avoid repeating the CK-4 "
            "cold-load delay."
        ),
        "axes": [
            {
                "axis_id": axis.axis_id,
                "positive_pole": axis.positive_pole,
                "negative_pole": axis.negative_pole,
                "contrasts": [contrast.contrast_id for contrast in axis.contrasts],
            }
            for axis in axes
        ],
        "artifacts": [
            {
                "axis_id": artifact.axis_id,
                "component_id": _component_id(artifact.axis_id),
                "path": str(artifact.path),
                "layer": artifact.layer,
                "strength": artifact.strength,
            }
            for artifact in artifacts
        ],
        "recipes": [
            {
                "recipe_id": recipe.recipe_id,
                "label": recipe.label,
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
            for recipe in recipes
        ],
        "validation_gates": [
            "per-axis vector direction file exists and matches target hidden size",
            "per-axis constant recipes beat baseline without higher pseudo-risk",
            "bundle recipes beat CK-4 proxy_guardrails_only on CK-1 delta",
            "ramped clamp preserves or lowers pseudo-attunement risk",
            "projection telemetry confirms movement along intended axes",
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


def recipe_to_cli_arg(recipe: CocktailRecipeSpec) -> str:
    """Serialize a recipe to the CK-3/CK-4 ``--recipe`` argument format."""

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


def _guardrail_component(
    artifact: GuardrailArtifactSpec,
    *,
    schedule: str,
) -> CocktailDirectionSpec:
    return CocktailDirectionSpec(
        component_id=_component_id(artifact.axis_id),
        path=artifact.path,
        layer=artifact.layer,
        strength=artifact.strength,
        steering_schedule=schedule,
    )


def _component_id(axis_id: str) -> str:
    return f"guardrail_{axis_id}"


def _schedule_slug(schedule: str) -> str:
    return schedule.replace("-", "_")


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
        help="Trait-axis guardrail to include. Defaults to CK-4.5 priority axes.",
    )
    parser.add_argument(
        "--direction-root",
        type=Path,
        default=config.paths.vectors / "ck45_guardrails",
        help="Root for inferred per-axis direction artifacts.",
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
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument(
        "--schedule",
        action="append",
        default=None,
        help="Schedule for per-axis single-vector recipes.",
    )
    parser.add_argument("--ck1-direction", type=Path, default=None)
    parser.add_argument("--ck1-layer", type=int, default=-2)
    parser.add_argument("--ck1-strength", type=float, default=1.0)
    parser.add_argument("--ck1-schedule", default="decay-8")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.reports / "ck45_guardrail_vector_grid.json",
    )
    parser.add_argument("--recipe-specs-output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
