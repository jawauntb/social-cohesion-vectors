from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_ck7_candidate_recipe_grid import (
    DEFAULT_CK7_PRESSURE_AXES,
    CK7ScheduleSpec,
    build_ck7_candidate_recipes,
    build_guardrail_artifact_specs,
    main,
    parse_schedule_specs,
    recipe_to_cli_arg,
    select_ck7_axes,
    write_ck7_candidate_grid,
)
from scripts.export_ck45_guardrail_vector_grid import DEFAULT_CK45_GUARDRAIL_AXES


def test_select_ck7_axes_includes_ck45_guardrails_and_pressure_axes() -> None:
    axes = select_ck7_axes()
    axis_ids = [axis.axis_id for axis in axes]

    assert axis_ids == list(DEFAULT_CK7_PRESSURE_AXES)
    assert set(DEFAULT_CK45_GUARDRAIL_AXES).issubset(axis_ids)
    assert "repair_vs_harm_denial" in axis_ids
    assert "autonomy_vs_coercion" in axis_ids
    assert "constructive_dissent_vs_conformity" in axis_ids


def test_parse_schedule_specs_accepts_named_ck1_guardrail_pairs() -> None:
    schedules = parse_schedule_specs(["late_clamp=decay-8:ramp-5-16"])

    assert schedules == (
        CK7ScheduleSpec(
            schedule_id="late_clamp",
            label="late clamp",
            ck1_schedule="decay-8",
            guardrail_schedule="ramp-5-16",
        ),
    )


def test_parse_schedule_specs_rejects_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="Unsupported schedule"):
        parse_schedule_specs(["bad=always:ramp-5-16"])


def test_build_candidate_recipes_expands_layer_dose_and_controls(tmp_path) -> None:
    axes = select_ck7_axes(["truth_vs_deception"])
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path / "vectors",
        direction_template="{axis_id}.npz",
        layer=-1,
        strength=0.25,
    )
    schedule = CK7ScheduleSpec(
        schedule_id="decay_then_ramp",
        label="decay then ramp",
        ck1_schedule="decay-8",
        guardrail_schedule="ramp-5-16",
    )

    recipes = build_ck7_candidate_recipes(
        artifacts,
        ck1_direction=tmp_path / "ck1.npz",
        ck1_layers=(-2, -1),
        ck1_strengths=(0.5,),
        guardrail_strengths=(0.25,),
        schedules=(schedule,),
        include_controls=True,
        random_control_root=tmp_path / "random",
        random_seed=13,
    )

    by_id = {recipe.recipe_id: recipe for recipe in recipes}
    assert set(by_id) == {
        "baseline",
        "ck7_axis_truth_vs_deception_decay_then_ramp_g0p25",
        "ck7_guardrail_bundle_decay_then_ramp_g0p25",
        "ck7_ck1_lm2_d0p5_decay_then_ramp",
        "ck7_ck1_lm1_d0p5_decay_then_ramp",
        "ck7_pressure_bundle_lm2_ck1_d0p5_g0p25_decay_then_ramp",
        "ck7_pressure_bundle_lm1_ck1_d0p5_g0p25_decay_then_ramp",
        "ck7_control_signflip_lm2_ck1_dneg_0p5_gneg_0p25_decay_then_ramp",
        "ck7_control_signflip_lm1_ck1_dneg_0p5_gneg_0p25_decay_then_ramp",
        "ck7_control_random_lm2_ck1_d0p5_g0p25_decay_then_ramp",
        "ck7_control_random_lm1_ck1_d0p5_g0p25_decay_then_ramp",
    }
    pressure = by_id[
        "ck7_pressure_bundle_lm2_ck1_d0p5_g0p25_decay_then_ramp"
    ]
    assert pressure.components[0].component_id == "ck1_boundary"
    assert pressure.components[0].layer == -2
    assert pressure.components[0].strength == 0.5
    assert pressure.components[0].steering_schedule == "decay-8"
    assert pressure.components[1].component_id == "guardrail_truth_vs_deception"
    assert pressure.components[1].steering_schedule == "ramp-5-16"

    signflip = by_id[
        "ck7_control_signflip_lm2_ck1_dneg_0p5_gneg_0p25_decay_then_ramp"
    ]
    assert [component.strength for component in signflip.components] == [-0.5, -0.25]

    random = by_id["ck7_control_random_lm2_ck1_d0p5_g0p25_decay_then_ramp"]
    assert "ck1_random_seed13_h896.npz" in str(random.components[0].path)
    assert "truth_vs_deception_random_seed13_h896.npz" in str(
        random.components[1].path
    )


def test_write_grid_records_specs_only_contract_and_recipe_text(tmp_path) -> None:
    axes = select_ck7_axes(["autonomy_vs_coercion"])
    artifacts = build_guardrail_artifact_specs(
        axes,
        direction_root=tmp_path / "vectors",
        direction_template="{axis_id}.npz",
    )
    schedule = CK7ScheduleSpec(
        schedule_id="steady_constant",
        label="steady constant",
        ck1_schedule="constant",
        guardrail_schedule="constant",
    )
    recipes = build_ck7_candidate_recipes(
        artifacts,
        ck1_direction=tmp_path / "ck1.npz",
        ck1_layers=(-2,),
        ck1_strengths=(0.5,),
        guardrail_strengths=(0.25,),
        schedules=(schedule,),
        include_controls=False,
    )
    output = tmp_path / "ck7.json"
    recipe_specs_output = tmp_path / "recipes.txt"

    payload = write_ck7_candidate_grid(
        axes,
        artifacts,
        recipes,
        ck1_direction=tmp_path / "ck1.npz",
        ck1_layers=(-2,),
        ck1_strengths=(0.5,),
        guardrail_strengths=(0.25,),
        schedules=(schedule,),
        include_controls=False,
        random_control_root=tmp_path / "random",
        random_seed=7,
        json_output=output,
        recipe_specs_output=recipe_specs_output,
    )

    from_disk = json.loads(output.read_text(encoding="utf-8"))
    assert from_disk == payload
    assert from_disk["experiment"] == "ck7_candidate_recipe_grid"
    assert from_disk["dry_run"] is True
    assert "No Modal execution" in from_disk["claim_boundary"]
    assert from_disk["controls_enabled"] is False
    assert from_disk["axes"][0]["role"] == "autonomy_preservation"
    assert "sign-flipped and random controls fail" in "\n".join(
        from_disk["validation_gates"]
    )
    assert recipe_to_cli_arg(recipes[-1]) in recipe_specs_output.read_text(
        encoding="utf-8"
    )


def test_main_writes_dry_run_without_existing_vector_files(tmp_path, capsys) -> None:
    output = tmp_path / "ck7.json"

    result = main(
        [
            "--axis",
            "truth_vs_deception",
            "--direction-root",
            str(tmp_path / "missing_vectors"),
            "--ck1-direction",
            str(tmp_path / "missing_ck1.npz"),
            "--ck1-layer",
            "-2",
            "--ck1-strength",
            "0.5",
            "--guardrail-strength",
            "0.25",
            "--schedule",
            "late_clamp=decay-8:ramp-5-16",
            "--no-controls",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    recipe_ids = [recipe["recipe_id"] for recipe in payload["recipes"]]
    assert result == 0
    assert payload["axes"][0]["axis_id"] == "truth_vs_deception"
    assert payload["controls_enabled"] is False
    assert len(payload["recipes"]) == 5
    assert "ck7_pressure_bundle_lm2_ck1_d0p5_g0p25_late_clamp" in recipe_ids
    assert "CK-7 candidate recipe grid dry run" in captured.out
    assert "--recipe ck7_pressure_bundle_lm2_ck1_d0p5_g0p25_late_clamp=" in captured.out
