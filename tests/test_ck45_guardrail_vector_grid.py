from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_ck45_guardrail_vector_grid import (
    DEFAULT_CK45_GUARDRAIL_AXES,
    build_ck45_guardrail_recipes,
    build_guardrail_artifact_specs,
    main,
    recipe_to_cli_arg,
    select_guardrail_axes,
    write_ck45_guardrail_grid,
)


def test_select_guardrail_axes_defaults_to_ck45_priority_axes() -> None:
    axes = select_guardrail_axes()

    assert [axis.axis_id for axis in axes] == list(DEFAULT_CK45_GUARDRAIL_AXES)
    assert axes[0].positive_pole == "principled respect"
    assert axes[0].negative_pole == "sycophancy"


def test_build_artifact_specs_uses_template_and_overrides(tmp_path) -> None:
    axes = select_guardrail_axes(["truth_vs_deception", "privacy_exit_vs_surveillance_lock_in"])

    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path / "vectors",
        direction_template="axis__{axis_id}.npz",
        artifact_overrides={
            "truth_vs_deception": tmp_path / "truth_override.npz",
        },
        layer=-2,
        strength=0.2,
    )

    assert artifacts[0].axis_id == "truth_vs_deception"
    assert artifacts[0].path == tmp_path / "truth_override.npz"
    assert artifacts[0].layer == -2
    assert artifacts[0].strength == 0.2
    assert artifacts[1].path == (
        tmp_path
        / "vectors"
        / "axis__privacy_exit_vs_surveillance_lock_in.npz"
    )


def test_build_recipes_includes_single_axis_bundle_and_ck1_clamp(tmp_path) -> None:
    axes = select_guardrail_axes(["principled_respect_vs_sycophancy"])
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path,
        direction_template="{axis_id}.npz",
    )

    recipes = build_ck45_guardrail_recipes(
        artifacts,
        schedules=("constant", "ramp-5-16"),
        ck1_direction=tmp_path / "ck1.npz",
    )

    by_id = {recipe.recipe_id: recipe for recipe in recipes}
    assert set(by_id) == {
        "baseline",
        "guardrail_principled_respect_vs_sycophancy_constant",
        "guardrail_principled_respect_vs_sycophancy_ramp_5_16",
        "guardrail_axis_bundle_constant",
        "guardrail_axis_bundle_ramp",
        "ck1_decay_per_axis_clamp",
    }
    assert by_id["baseline"].components == ()
    assert by_id["ck1_decay_per_axis_clamp"].components[0].component_id == "ck1_boundary"
    assert by_id["ck1_decay_per_axis_clamp"].components[0].steering_schedule == "decay-8"
    assert by_id["ck1_decay_per_axis_clamp"].components[1].steering_schedule == "ramp-5-16"


def test_write_grid_dry_run_records_claim_boundary_and_recipe_specs(tmp_path) -> None:
    axes = select_guardrail_axes(["truth_vs_deception"])
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path,
        direction_template="{axis_id}.npz",
    )
    recipes = build_ck45_guardrail_recipes(artifacts)
    output = tmp_path / "grid.json"
    recipe_specs_output = tmp_path / "recipes.txt"

    payload = write_ck45_guardrail_grid(
        axes,
        artifacts,
        recipes,
        json_output=output,
        recipe_specs_output=recipe_specs_output,
    )

    from_disk = json.loads(output.read_text(encoding="utf-8"))
    assert from_disk == payload
    assert from_disk["experiment"] == "ck45_guardrail_vector_grid"
    assert "No Modal execution" in from_disk["claim_boundary"]
    assert "persistent Hugging Face/model cache" in from_disk["modal_runtime_note"]
    assert recipe_to_cli_arg(recipes[1]) in recipe_specs_output.read_text(
        encoding="utf-8"
    )


def test_main_writes_dry_run_without_existing_vector_files(tmp_path, capsys) -> None:
    output = tmp_path / "ck45.json"

    result = main(
        [
            "--axis",
            "truth_vs_deception",
            "--direction-root",
            str(tmp_path / "missing_vectors"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["axes"][0]["axis_id"] == "truth_vs_deception"
    assert len(payload["recipes"]) == 5
    assert "CK-4.5 guardrail vector grid dry run" in captured.out
    assert "--recipe guardrail_truth_vs_deception_constant=" in captured.out


def test_main_custom_schedule_replaces_defaults(tmp_path) -> None:
    output = tmp_path / "ck45.json"

    result = main(
        [
            "--axis",
            "truth_vs_deception",
            "--direction-root",
            str(tmp_path / "missing_vectors"),
            "--schedule",
            "first-4",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    recipe_ids = [recipe["recipe_id"] for recipe in payload["recipes"]]
    assert result == 0
    assert "guardrail_truth_vs_deception_first_4" in recipe_ids
    assert "guardrail_truth_vs_deception_constant" not in recipe_ids
    assert "guardrail_truth_vs_deception_ramp_5_16" not in recipe_ids
