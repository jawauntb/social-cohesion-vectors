"""Build CK-4 scheduled cocktail recipes and optionally run the CK-3 Modal path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.ck3_cocktail import (
    CocktailRecipeSpec,
    parse_recipe_spec,
)

SCHEDULES = ("constant", "first-N", "after-N", "decay-N", "ramp-A-B")
_SCHEDULE_PATTERN = re.compile(
    r"^(?:constant|first-[1-9][0-9]*|after-[0-9]+|decay-[1-9][0-9]*|"
    r"ramp-[0-9]+-[1-9][0-9]*)$"
)


@dataclass(frozen=True)
class ScheduledComponentTemplate:
    """A component recipe before a named artifact path is resolved."""

    component_id: str
    layer: int
    strength: float
    schedule: str = "constant"
    hook_site: str = "post"
    position: str = "last"
    timing: str = "generate"


@dataclass(frozen=True)
class ScheduledRecipeTemplate:
    """A named CK-4 scheduled recipe template."""

    recipe_id: str
    label: str
    components: tuple[ScheduledComponentTemplate, ...]


DEFAULT_RECIPE_TEMPLATES: tuple[ScheduledRecipeTemplate, ...] = (
    ScheduledRecipeTemplate("baseline", "Baseline", ()),
    ScheduledRecipeTemplate(
        "guardrails_only",
        "Guardrails only",
        (
            ScheduledComponentTemplate("sycophancy", -1, -0.35, "constant"),
            ScheduledComponentTemplate("hallucination", -1, -0.35, "constant"),
        ),
    ),
    ScheduledRecipeTemplate(
        "split_timing",
        "Split timing",
        (
            ScheduledComponentTemplate("ck1", -2, 0.75, "first-4"),
            ScheduledComponentTemplate("sycophancy", -1, -0.35, "after-4"),
            ScheduledComponentTemplate("hallucination", -1, -0.35, "after-4"),
        ),
    ),
    ScheduledRecipeTemplate(
        "decay_then_clamp",
        "Decay then clamp",
        (
            ScheduledComponentTemplate("ck1", -2, 1.0, "decay-8"),
            ScheduledComponentTemplate("sycophancy", -1, -0.35, "ramp-5-16"),
            ScheduledComponentTemplate("hallucination", -1, -0.35, "ramp-5-16"),
        ),
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = parse_direction_artifacts(args.direction)
    templates = select_recipe_templates(args.recipe)
    recipe_specs = build_scheduled_recipe_specs(
        artifacts,
        templates=templates,
        default_hook_site=args.hook_site,
        default_position=args.steering_position,
        default_timing=args.steering_timing,
    )

    if args.dry_run or not args.run_modal:
        write_dry_run_specs(
            recipe_specs,
            json_output=args.specs_output,
            text_output=args.recipe_specs_output,
        )
        print(
            "CK-4 scheduled cocktail dry run: "
            f"recipes={len(recipe_specs)} json={args.specs_output}"
        )
        for spec in recipe_specs:
            print(f"--recipe {recipe_spec_to_cli_arg(spec)}")
        return 0

    run_ck3_modal_cocktail = load_ck3_modal_runner()
    ck3_args = [
        *([] if args.prompts is None else ["--prompts", str(args.prompts)]),
        *([] if args.limit is None else ["--limit", str(args.limit)]),
        *[
            item
            for recipe in recipe_specs
            for item in ("--recipe", recipe_spec_to_cli_arg(recipe))
        ],
        "--baseline-recipe-id",
        args.baseline_recipe_id,
        "--model-id",
        args.model_id,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-length",
        str(args.max_length),
        "--hook-site",
        args.hook_site,
        "--steering-position",
        args.steering_position,
        "--steering-timing",
        args.steering_timing,
        "--seed",
        str(args.seed),
        "--prompts-output",
        str(args.prompts_output),
        "--generations-output",
        str(args.generations_output),
        "--json-output",
        str(args.json_output),
        "--markdown-output",
        str(args.markdown_output),
    ]
    if args.no_chat_template:
        ck3_args.append("--no-chat-template")
    return run_ck3_modal_cocktail(ck3_args)


def load_ck3_modal_runner():
    """Load the sibling CK-3 runner without importing Modal in dry-run mode."""

    runner_path = Path(__file__).with_name("run_ck3_modal_cocktail.py")
    spec = importlib.util.spec_from_file_location("run_ck3_modal_cocktail", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CK-3 Modal runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def parse_direction_artifacts(values: Sequence[str]) -> dict[str, Path]:
    """Parse ``component_id=path`` direction artifact bindings."""

    artifacts: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Direction artifacts must look like component_id=path.")
        component_id, path = value.split("=", 1)
        component_id = component_id.strip()
        if not component_id:
            raise ValueError("Direction artifact component id cannot be empty.")
        if component_id in artifacts:
            raise ValueError(f"Duplicate direction artifact: {component_id}")
        artifacts[component_id] = Path(path.strip())
    return artifacts


def select_recipe_templates(
    requested_recipe_ids: Sequence[str] | None,
) -> tuple[ScheduledRecipeTemplate, ...]:
    """Select default CK-4 recipe templates by id."""

    if not requested_recipe_ids:
        return DEFAULT_RECIPE_TEMPLATES
    templates_by_id = {recipe.recipe_id: recipe for recipe in DEFAULT_RECIPE_TEMPLATES}
    unknown = sorted(set(requested_recipe_ids).difference(templates_by_id))
    if unknown:
        raise ValueError(f"Unknown CK-4 recipe template(s): {', '.join(unknown)}")
    return tuple(templates_by_id[recipe_id] for recipe_id in requested_recipe_ids)


def build_scheduled_recipe_specs(
    artifacts: Mapping[str, Path],
    *,
    templates: Sequence[ScheduledRecipeTemplate] = DEFAULT_RECIPE_TEMPLATES,
    default_hook_site: str = "post",
    default_position: str = "last",
    default_timing: str = "generate",
) -> list[CocktailRecipeSpec]:
    """Resolve named artifacts into CK-3-compatible scheduled recipe specs."""

    specs: list[CocktailRecipeSpec] = []
    for template in templates:
        missing = sorted(
            {
                component.component_id
                for component in template.components
                if component.component_id not in artifacts
            }
        )
        if missing:
            raise ValueError(
                f"Recipe {template.recipe_id!r} needs missing artifact(s): "
                f"{', '.join(missing)}"
            )
        spec = recipe_spec_to_cli_arg(
            scheduled_template_to_recipe_spec(
                template,
                artifacts=artifacts,
                default_hook_site=default_hook_site,
                default_position=default_position,
                default_timing=default_timing,
            )
        )
        specs.append(
            parse_recipe_spec(
                spec,
                default_hook_site=default_hook_site,
                default_position=default_position,
                default_timing=default_timing,
            )
        )
    return specs


def scheduled_template_to_recipe_spec(
    template: ScheduledRecipeTemplate,
    *,
    artifacts: Mapping[str, Path],
    default_hook_site: str = "post",
    default_position: str = "last",
    default_timing: str = "generate",
) -> CocktailRecipeSpec:
    """Convert one scheduled template into a parsed CK-3 recipe spec."""

    components: list[str] = []
    for component in template.components:
        validate_schedule(component.schedule)
        components.append(
            ":".join(
                (
                    component.component_id,
                    str(artifacts[component.component_id]),
                    str(component.layer),
                    str(component.strength),
                    component.hook_site or default_hook_site,
                    component.position or default_position,
                    component.timing or default_timing,
                    component.schedule,
                )
            )
        )
    return parse_recipe_spec(
        f"{template.recipe_id}={template.label}|{','.join(components)}",
        default_hook_site=default_hook_site,
        default_position=default_position,
        default_timing=default_timing,
    )


def recipe_spec_to_cli_arg(recipe: CocktailRecipeSpec) -> str:
    """Serialize a parsed recipe back to the CK-3 ``--recipe`` format."""

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


def write_dry_run_specs(
    recipes: Sequence[CocktailRecipeSpec],
    *,
    json_output: Path,
    text_output: Path | None = None,
) -> dict[str, object]:
    """Write dry-run specs without loading direction files or importing Modal."""

    payload = {
        "experiment": "ck4_scheduled_cocktail_dry_run",
        "description": (
            "Compute-only CK-4 scheduled activation-cocktail recipe specs. "
            "These specs do not make human, behavioral, or neural claims."
        ),
        "supported_schedules": list(SCHEDULES),
        "recipes": [
            {
                "recipe_id": recipe.recipe_id,
                "label": recipe.label,
                "recipe_spec": recipe_spec_to_cli_arg(recipe),
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
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if text_output is not None:
        text_output.parent.mkdir(parents=True, exist_ok=True)
        text_output.write_text(
            "\n".join(recipe_spec_to_cli_arg(recipe) for recipe in recipes) + "\n",
            encoding="utf-8",
        )
    return payload


def validate_schedule(schedule: str) -> None:
    """Validate CK-4 schedule grammar."""

    if not _SCHEDULE_PATTERN.match(schedule):
        raise ValueError(
            f"Unsupported schedule {schedule!r}. Use one of: "
            f"{', '.join(SCHEDULES)}."
        )
    if schedule.startswith("ramp-"):
        _, start, end = schedule.split("-")
        if int(end) <= int(start):
            raise ValueError("Ramp schedules must use ramp-A-B with B > A.")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        action="append",
        required=True,
        help="Named direction artifact binding, e.g. ck1=/path/ck1.npz.",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        choices=[recipe.recipe_id for recipe in DEFAULT_RECIPE_TEMPLATES],
        help="CK-4 recipe template to include. Defaults to all templates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print recipe specs without importing Modal.",
    )
    parser.add_argument(
        "--run-modal",
        action="store_true",
        help="Call scripts/run_ck3_modal_cocktail.py with the built recipes.",
    )
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-recipe-id", default="baseline")
    parser.add_argument("--model-id", default=config.model_ids.open_llm)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hook-site", choices=("pre", "post"), default="post")
    parser.add_argument("--steering-position", choices=("last", "all"), default="last")
    parser.add_argument(
        "--steering-timing",
        choices=("always", "prefill", "generate"),
        default="generate",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--specs-output",
        type=Path,
        default=config.paths.reports / "ck4_scheduled_cocktail_specs.json",
    )
    parser.add_argument("--recipe-specs-output", type=Path, default=None)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=config.paths.training / "ck4_scheduled_cocktail_prompts.jsonl",
    )
    parser.add_argument(
        "--generations-output",
        type=Path,
        default=config.paths.processed / "ck4_scheduled_cocktail_generations.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "ck4_scheduled_cocktail_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "ck4_scheduled_cocktail_report.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
