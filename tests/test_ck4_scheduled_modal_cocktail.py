from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ck4_scheduled_modal_cocktail import (
    DEFAULT_RECIPE_TEMPLATES,
    ScheduledComponentTemplate,
    ScheduledRecipeTemplate,
    build_scheduled_recipe_specs,
    main,
    recipe_spec_to_cli_arg,
    validate_schedule,
    write_dry_run_specs,
)


def test_build_scheduled_recipe_specs_resolves_named_artifacts(tmp_path) -> None:
    artifacts = {
        "ck1": tmp_path / "ck1.npz",
        "sycophancy": tmp_path / "sycophancy.npz",
        "hallucination": tmp_path / "hallucination.npz",
    }

    recipes = build_scheduled_recipe_specs(artifacts)

    by_id = {recipe.recipe_id: recipe for recipe in recipes}
    assert set(by_id) == {recipe.recipe_id for recipe in DEFAULT_RECIPE_TEMPLATES}
    split = by_id["split_timing"]
    assert split.label == "Split timing"
    assert [component.component_id for component in split.components] == [
        "ck1",
        "sycophancy",
        "hallucination",
    ]
    assert [component.steering_schedule for component in split.components] == [
        "first-4",
        "after-4",
        "after-4",
    ]
    assert split.components[0].path == artifacts["ck1"]
    assert split.components[0].layer == -2
    assert split.components[0].strength == 0.75


def test_build_scheduled_recipe_specs_reports_missing_named_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing artifact"):
        build_scheduled_recipe_specs(
            {"ck1": tmp_path / "ck1.npz"},
            templates=[
                ScheduledRecipeTemplate(
                    "guardrails_only",
                    "Guardrails only",
                    (
                        ScheduledComponentTemplate(
                            "sycophancy",
                            -1,
                            -0.35,
                            "constant",
                        ),
                    ),
                )
            ],
        )


def test_validate_schedule_accepts_named_grammar_and_rejects_bad_ramp() -> None:
    for schedule in ("constant", "first-4", "after-4", "decay-8", "ramp-5-16"):
        validate_schedule(schedule)

    with pytest.raises(ValueError, match="B > A"):
        validate_schedule("ramp-8-5")


def test_write_dry_run_specs_writes_json_and_recipe_text(tmp_path) -> None:
    artifacts = {
        "ck1": tmp_path / "ck1.npz",
        "sycophancy": tmp_path / "sycophancy.npz",
        "hallucination": tmp_path / "hallucination.npz",
    }
    recipes = build_scheduled_recipe_specs(artifacts)
    json_output = tmp_path / "specs.json"
    text_output = tmp_path / "recipes.txt"

    payload = write_dry_run_specs(
        recipes,
        json_output=json_output,
        text_output=text_output,
    )

    from_disk = json.loads(json_output.read_text(encoding="utf-8"))
    assert from_disk == payload
    assert from_disk["experiment"] == "ck4_scheduled_cocktail_dry_run"
    assert from_disk["supported_schedules"] == [
        "constant",
        "first-N",
        "after-N",
        "decay-N",
        "ramp-A-B",
    ]
    assert "human, behavioral, or neural claims" in from_disk["description"]
    assert recipe_spec_to_cli_arg(recipes[0]) in text_output.read_text(
        encoding="utf-8"
    )


def test_main_dry_run_writes_specs_without_direction_files(tmp_path, capsys) -> None:
    specs_output = tmp_path / "dry_run.json"

    result = main(
        [
            "--dry-run",
            "--direction",
            f"ck1={tmp_path / 'missing_ck1.npz'}",
            "--direction",
            f"sycophancy={tmp_path / 'missing_sycophancy.npz'}",
            "--direction",
            f"hallucination={tmp_path / 'missing_hallucination.npz'}",
            "--specs-output",
            str(specs_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(specs_output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["recipes"][2]["recipe_id"] == "split_timing"
    assert "CK-4 scheduled cocktail dry run" in captured.out
    assert "--recipe split_timing=Split timing|" in captured.out
