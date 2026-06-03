"""CK-3 cocktail steering recipe parsing, scoring, and reports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.ck1_steering import score_ck1_generations


@dataclass(frozen=True)
class CocktailDirectionSpec:
    """A local direction artifact before it is serialized for Modal."""

    component_id: str
    path: Path
    layer: int
    strength: float
    hook_site: str = "post"
    steering_position: str = "last"
    steering_timing: str = "generate"


@dataclass(frozen=True)
class CocktailRecipeSpec:
    """A named steering recipe composed from zero or more direction terms."""

    recipe_id: str
    label: str
    components: tuple[CocktailDirectionSpec, ...]


def parse_recipe_spec(
    value: str,
    *,
    default_hook_site: str = "post",
    default_position: str = "last",
    default_timing: str = "generate",
) -> CocktailRecipeSpec:
    """Parse ``recipe_id=label|component:path:layer:strength,...``."""

    if "=" not in value:
        raise ValueError("Recipe specs must look like recipe_id=label|component:...")
    recipe_id, payload = value.split("=", 1)
    recipe_id = recipe_id.strip()
    if not recipe_id:
        raise ValueError("Recipe id cannot be empty.")
    label = recipe_id
    component_payload = payload.strip()
    if "|" in payload:
        label, component_payload = payload.split("|", 1)
        label = label.strip() or recipe_id
    components = tuple(
        _parse_component_spec(
            component,
            default_hook_site=default_hook_site,
            default_position=default_position,
            default_timing=default_timing,
        )
        for component in component_payload.split(",")
        if component.strip()
    )
    return CocktailRecipeSpec(recipe_id=recipe_id, label=label, components=components)


def recipe_specs_to_modal_payload(
    recipes: Sequence[CocktailRecipeSpec],
) -> list[dict[str, Any]]:
    """Load direction files and build Modal-ready recipe dictionaries."""

    return [
        {
            "recipe_id": recipe.recipe_id,
            "label": recipe.label,
            "components": [
                {
                    "component_id": component.component_id,
                    "direction": _load_unit_direction(component.path).tolist(),
                    "layer": component.layer,
                    "strength": component.strength,
                    "hook_site": component.hook_site,
                    "steering_position": component.steering_position,
                    "steering_timing": component.steering_timing,
                }
                for component in recipe.components
            ],
        }
        for recipe in recipes
    ]


def shape_ck3_cocktail_report(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_recipe_id: str = "baseline",
) -> dict[str, Any]:
    """Summarize CK-3 recipe generations by recipe and prompt."""

    scored = score_ck1_generations(records)
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in scored:
        recipe_id = str(record["recipe_id"])
        prompt_id = str(record["prompt_id"])
        by_recipe.setdefault(recipe_id, []).append(record)
        by_prompt.setdefault(prompt_id, []).append(record)

    recipe_rows = [
        _recipe_row(recipe_id, rows, baseline_rows=by_recipe.get(baseline_recipe_id, []))
        for recipe_id, rows in sorted(by_recipe.items())
    ]
    prompt_rows = [
        _prompt_row(prompt_id, rows, baseline_recipe_id=baseline_recipe_id)
        for prompt_id, rows in sorted(by_prompt.items())
    ]
    best_recipe = max(
        recipe_rows,
        key=lambda row: float(row.get("mean_ck1_score", 0.0)),
        default={},
    )
    return {
        "experiment": "ck3_cocktail_steering",
        "description": (
            "Generates held-out CK-1 social-state prompts under named activation "
            "cocktail recipes, then scores safe-attunement benefit and "
            "pseudo-attunement side effects."
        ),
        "summary": {
            "generations": len(scored),
            "prompts": len(by_prompt),
            "recipes": len(by_recipe),
            "baseline_recipe_id": baseline_recipe_id,
            "best_recipe_id": str(best_recipe.get("recipe_id", "")),
            "best_mean_ck1_score": float(best_recipe.get("mean_ck1_score", 0.0))
            if best_recipe
            else 0.0,
            "best_minus_baseline_mean_ck1_delta": float(
                best_recipe.get("mean_ck1_delta_vs_baseline", 0.0)
            )
            if best_recipe
            else 0.0,
        },
        "recipes": recipe_rows,
        "prompts": prompt_rows,
        "records": scored,
    }


def write_ck3_cocktail_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    baseline_recipe_id: str = "baseline",
) -> dict[str, Any]:
    """Write JSON and Markdown CK-3 cocktail reports."""

    report = shape_ck3_cocktail_report(
        records,
        baseline_recipe_id=baseline_recipe_id,
    )
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_ck3_cocktail_markdown(report), encoding="utf-8")
    return report


def write_ck3_generations(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> int:
    """Write scored CK-3 generations to JSONL."""

    return write_jsonl(score_ck1_generations(records), path)


def render_ck3_cocktail_markdown(report: Mapping[str, Any]) -> str:
    """Render a CK-3 cocktail report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# CK-3 Cocktail Steering Report",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompts: {int(summary.get('prompts', 0))}",
        f"- Generations: {int(summary.get('generations', 0))}",
        f"- Recipes: {int(summary.get('recipes', 0))}",
        f"- Baseline recipe: `{summary.get('baseline_recipe_id', '')}`",
        f"- Best recipe: `{summary.get('best_recipe_id', '')}`",
        "- Best-minus-baseline mean CK-1 delta: "
        f"{float(summary.get('best_minus_baseline_mean_ck1_delta', 0.0)):+.3f}",
        "",
        "## Recipe Means",
        "",
        "| Recipe | Runs | Components | CK-1 | Delta vs baseline | Safe signal | Pseudo risk | Pseudo delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("recipes")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('recipe_id', '')} | "
            f"{int(row_map.get('runs', 0))} | "
            f"{int(row_map.get('components', 0))} | "
            f"{float(row_map.get('mean_ck1_score', 0.0)):.3f} | "
            f"{float(row_map.get('mean_ck1_delta_vs_baseline', 0.0)):+.3f} | "
            f"{float(row_map.get('mean_safe_attunement_signal', 0.0)):.3f} | "
            f"{float(row_map.get('mean_pseudo_attunement_risk', 0.0)):.3f} | "
            f"{float(row_map.get('mean_pseudo_delta_vs_baseline', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt Deltas",
            "",
            "| Prompt | Best non-baseline recipe | Best delta | Worst pseudo-risk delta |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in _sequence(report.get("prompts")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('prompt_id', '')} | "
            f"{row_map.get('best_recipe_id', '')} | "
            f"{float(row_map.get('best_ck1_delta_vs_baseline', 0.0)):+.3f} | "
            f"{float(row_map.get('worst_pseudo_delta_vs_baseline', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a compute-only cocktail assay. The provisional antagonist "
            "directions are trait-axis controls, not biological receptor models "
            "and not human or neural validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_component_spec(
    value: str,
    *,
    default_hook_site: str,
    default_position: str,
    default_timing: str,
) -> CocktailDirectionSpec:
    fields = [field.strip() for field in value.split(":")]
    if len(fields) not in {4, 7}:
        raise ValueError(
            "Component specs must look like component:path:layer:strength "
            "or component:path:layer:strength:hook_site:position:timing."
        )
    component_id, path, layer, strength = fields[:4]
    if not component_id:
        raise ValueError("Component id cannot be empty.")
    hook_site, position, timing = (
        fields[4:]
        if len(fields) == 7
        else [default_hook_site, default_position, default_timing]
    )
    return CocktailDirectionSpec(
        component_id=component_id,
        path=Path(path),
        layer=int(layer),
        strength=float(strength),
        hook_site=hook_site,
        steering_position=position,
        steering_timing=timing,
    )


def _load_unit_direction(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Direction file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        direction = np.asarray(data["direction"], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError(f"Direction file has zero norm: {path}")
    return direction / norm


def _recipe_row(
    recipe_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = _means(baseline_rows)
    means = _means(rows)
    components = _sequence(rows[0].get("components")) if rows else ()
    return {
        "recipe_id": recipe_id,
        "runs": len(rows),
        "components": len(components),
        "component_ids": [str(_mapping(component).get("component_id", "")) for component in components],
        "mean_ck1_score": means["ck1_score"],
        "mean_ck1_delta_vs_baseline": round(
            means["ck1_score"] - baseline["ck1_score"],
            6,
        ),
        "mean_safe_attunement_signal": means["safe_attunement_signal"],
        "mean_pseudo_attunement_risk": means["pseudo_attunement_risk"],
        "mean_pseudo_delta_vs_baseline": round(
            means["pseudo_attunement_risk"] - baseline["pseudo_attunement_risk"],
            6,
        ),
    }


def _prompt_row(
    prompt_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_recipe_id: str,
) -> dict[str, Any]:
    by_recipe = {str(row["recipe_id"]): row for row in rows}
    baseline = by_recipe.get(baseline_recipe_id)
    if baseline is None:
        return {
            "prompt_id": prompt_id,
            "best_recipe_id": "",
            "best_ck1_delta_vs_baseline": 0.0,
            "worst_pseudo_delta_vs_baseline": 0.0,
        }
    baseline_ck1 = float(baseline["ck1_score"])
    baseline_pseudo = _component(baseline, "pseudo_attunement_risk")
    candidates = [
        {
            "recipe_id": recipe_id,
            "ck1_delta": float(row["ck1_score"]) - baseline_ck1,
            "pseudo_delta": _component(row, "pseudo_attunement_risk")
            - baseline_pseudo,
        }
        for recipe_id, row in by_recipe.items()
        if recipe_id != baseline_recipe_id
    ]
    best = max(candidates, key=lambda row: row["ck1_delta"], default={})
    worst_pseudo = max(
        (row["pseudo_delta"] for row in candidates),
        default=0.0,
    )
    return {
        "prompt_id": prompt_id,
        "best_recipe_id": str(best.get("recipe_id", "")),
        "best_ck1_delta_vs_baseline": round(float(best.get("ck1_delta", 0.0)), 6),
        "worst_pseudo_delta_vs_baseline": round(float(worst_pseudo), 6),
    }


def _means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "ck1_score": _mean(float(row["ck1_score"]) for row in rows),
        "safe_attunement_signal": _mean(
            _component(row, "safe_attunement_signal") for row in rows
        ),
        "pseudo_attunement_risk": _mean(
            _component(row, "pseudo_attunement_risk") for row in rows
        ),
    }


def _component(row: Mapping[str, Any], key: str) -> float:
    components = _mapping(row.get("score_components"))
    return float(components.get(key, 0.0))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 6) if materialized else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()

