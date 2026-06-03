from __future__ import annotations

import numpy as np

from social_cohesion_vectors.experiments.ck3_cocktail import (
    parse_recipe_spec,
    recipe_specs_to_modal_payload,
    shape_ck3_cocktail_report,
)


def test_parse_recipe_spec_supports_baseline_and_components(tmp_path) -> None:
    direction_path = tmp_path / "direction.npz"
    spec = parse_recipe_spec(
        f"cocktail=Guardrails|ck1:{direction_path}:-1:0.5:post:last:generate"
    )
    baseline = parse_recipe_spec("baseline=Baseline|")

    assert spec.recipe_id == "cocktail"
    assert spec.label == "Guardrails"
    assert len(spec.components) == 1
    assert spec.components[0].component_id == "ck1"
    assert spec.components[0].path == direction_path
    assert spec.components[0].layer == -1
    assert spec.components[0].strength == 0.5
    assert spec.components[0].hook_site == "post"
    assert spec.components[0].steering_position == "last"
    assert spec.components[0].steering_timing == "generate"
    assert baseline.recipe_id == "baseline"
    assert baseline.components == ()


def test_recipe_specs_to_modal_payload_loads_unit_directions(tmp_path) -> None:
    direction_path = tmp_path / "direction.npz"
    np.savez_compressed(direction_path, direction=np.array([3.0, 4.0]))
    recipes = [
        parse_recipe_spec("baseline=Baseline|"),
        parse_recipe_spec(f"agonist=Agonist|ck1:{direction_path}:-1:1.0"),
    ]

    payload = recipe_specs_to_modal_payload(recipes)

    assert payload[0]["recipe_id"] == "baseline"
    assert payload[0]["components"] == []
    assert payload[1]["recipe_id"] == "agonist"
    assert payload[1]["components"][0]["component_id"] == "ck1"
    assert np.allclose(payload[1]["components"][0]["direction"], [0.6, 0.8])


def test_shape_ck3_cocktail_report_compares_to_baseline() -> None:
    records = [
        {
            "prompt_id": "case",
            "recipe_id": "baseline",
            "recipe_label": "Baseline",
            "components": [],
            "prompt": "Write a repair message.",
            "generated_text": "The group should coordinate the next message.",
        },
        {
            "prompt_id": "case",
            "recipe_id": "cocktail",
            "recipe_label": "Cocktail",
            "components": [{"component_id": "ck1"}],
            "prompt": "Write a repair message.",
            "generated_text": (
                "Keep the breach visible, name evidence, protect privacy, and "
                "let the harmed person decline contact without punishment."
            ),
        },
    ]

    report = shape_ck3_cocktail_report(records)

    assert report["summary"]["recipes"] == 2
    assert report["summary"]["best_recipe_id"] == "cocktail"
    assert report["summary"]["best_minus_baseline_mean_ck1_delta"] > 0.0
    recipe_rows = {row["recipe_id"]: row for row in report["recipes"]}
    assert recipe_rows["baseline"]["components"] == 0
    assert recipe_rows["cocktail"]["components"] == 1
    assert recipe_rows["cocktail"]["mean_ck1_delta_vs_baseline"] > 0.0
