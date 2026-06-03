"""Run a CK-3 activation-cocktail assay on Modal."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.ck1_steering import (
    default_ck1_steering_prompt_records,
)
from social_cohesion_vectors.experiments.ck3_cocktail import (
    parse_recipe_spec,
    recipe_specs_to_modal_payload,
    write_ck3_cocktail_reports,
    write_ck3_generations,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prompts = _load_prompt_records(args.prompts, args.limit)
    recipes = [
        parse_recipe_spec(
            recipe,
            default_hook_site=args.hook_site,
            default_position=args.steering_position,
            default_timing=args.steering_timing,
        )
        for recipe in args.recipe
    ]
    if not any(recipe.recipe_id == args.baseline_recipe_id for recipe in recipes):
        raise SystemExit(
            f"At least one recipe must have id {args.baseline_recipe_id!r}."
        )
    modal_recipes = recipe_specs_to_modal_payload(recipes)
    args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompts, args.prompts_output)

    print(
        "generating CK-3 cocktail responses on Modal "
        f"model={args.model_id} recipes={len(recipes)} prompts={len(prompts)}"
    )
    from social_cohesion_vectors.modal_app.app import app
    from social_cohesion_vectors.modal_app.functions.activation_steering import (
        generate_with_activation_cocktail,
    )

    with app.run():
        generations = generate_with_activation_cocktail.remote(
            records=prompts,
            recipes=modal_recipes,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            use_chat_template=not args.no_chat_template,
            seed=args.seed,
        )

    generations = _merge_prompt_metadata(generations, prompts)
    write_ck3_generations(generations, args.generations_output)
    report = write_ck3_cocktail_reports(
        generations,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        baseline_recipe_id=args.baseline_recipe_id,
    )
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("CK-3 cocktail report summary is not a dictionary")
    print(
        "CK-3 cocktail: "
        f"best={summary['best_recipe_id']} "
        f"best_delta="
        f"{float(summary['best_minus_baseline_mean_ck1_delta']):+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--recipe",
        action="append",
        required=True,
        help=(
            "Recipe spec. Format: recipe_id=label|component:path:layer:strength,"
            "component:path:layer:strength. Use recipe_id=Label| for baseline."
        ),
    )
    parser.add_argument("--baseline-recipe-id", default="baseline")
    parser.add_argument("--model-id", default=config.model_ids.open_llm)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--hook-site",
        choices=("pre", "post"),
        default="post",
    )
    parser.add_argument(
        "--steering-position",
        choices=("last", "all"),
        default="last",
    )
    parser.add_argument(
        "--steering-timing",
        choices=("always", "prefill", "generate"),
        default="generate",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=config.paths.training / "ck3_cocktail_prompts.jsonl",
    )
    parser.add_argument(
        "--generations-output",
        type=Path,
        default=config.paths.processed / "ck3_cocktail_generations.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "ck3_cocktail_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "ck3_cocktail_report.md",
    )
    return parser.parse_args(argv)


def _load_prompt_records(path: Path | None, limit: int | None) -> list[dict[str, str]]:
    if path is None:
        records = default_ck1_steering_prompt_records()
    else:
        records = [
            {
                "prompt_id": str(record.get("prompt_id", record.get("id", index))),
                "phase": str(record.get("phase", "")),
                "mechanism": str(record.get("mechanism", "")),
                "text": str(record["text"]),
            }
            for index, record in enumerate(read_jsonl(path))
        ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SystemExit("No CK-3 cocktail prompt records found.")
    return records


def _merge_prompt_metadata(
    generations: list[dict[str, object]],
    prompts: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    metadata_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    enriched: list[dict[str, object]] = []
    for generation in generations:
        prompt_id = str(generation.get("prompt_id", ""))
        metadata = metadata_by_id.get(prompt_id, {})
        enriched.append(
            {
                **metadata,
                **generation,
                "phase": str(generation.get("phase") or metadata.get("phase", "")),
                "mechanism": str(
                    generation.get("mechanism") or metadata.get("mechanism", "")
                ),
            }
        )
    return enriched


if __name__ == "__main__":
    raise SystemExit(main())

