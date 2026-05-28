from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_activation_layer_sweep import (  # noqa: E402
    aggregate_reports,
    experiment_command,
    extraction_command,
    layer_artifact_paths,
    render_summary_markdown,
    summary_output_paths,
)


def test_layer_artifact_paths_put_outputs_under_layer_sweep(
    tmp_path: Path,
) -> None:
    paths = layer_artifact_paths(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-4,
        features_root=tmp_path / "features",
        vectors_root=tmp_path / "vectors",
        reports_root=tmp_path / "reports",
    )

    expected_stem = "activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-4"
    assert paths.activation_npz == (
        tmp_path / "features" / "open_llm" / "layer_sweep" / f"{expected_stem}.npz"
    )
    assert paths.vector_output == (
        tmp_path / "vectors" / "open_llm" / "layer_sweep" / f"{expected_stem}.npz"
    )
    assert paths.json_report == tmp_path / "reports" / "layer_sweep" / (
        f"{expected_stem}.json"
    )
    assert paths.markdown_report == tmp_path / "reports" / "layer_sweep" / (
        f"{expected_stem}.md"
    )


def test_summary_output_paths_are_stable(tmp_path: Path) -> None:
    json_path, markdown_path = summary_output_paths(reports_root=tmp_path / "reports")

    assert json_path == tmp_path / "reports" / "layer_sweep" / "summary.json"
    assert markdown_path == tmp_path / "reports" / "layer_sweep" / "summary.md"


def test_commands_reuse_existing_activation_scripts(tmp_path: Path) -> None:
    extraction = extraction_command(
        prompts=tmp_path / "training" / "custom_prompts.jsonl",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-2,
        limit=5,
        batch_size=3,
        max_length=128,
        output=tmp_path / "features" / "layer-2.npz",
    )
    experiment = experiment_command(
        activation_npz=tmp_path / "features" / "layer-2.npz",
        vector_output=tmp_path / "vectors" / "layer-2.npz",
        json_output=tmp_path / "reports" / "layer-2.json",
        markdown_output=tmp_path / "reports" / "layer-2.md",
    )

    assert extraction[1].endswith("run_modal_activation_extraction.py")
    assert "--prompts" in extraction
    assert str(tmp_path / "training" / "custom_prompts.jsonl") in extraction
    assert "--limit" in extraction
    assert "5" in extraction
    assert experiment[1].endswith("run_activation_vector_experiment.py")
    assert str(tmp_path / "reports" / "layer-2.json") in experiment


def test_aggregate_reports_summarizes_fake_report_jsons(tmp_path: Path) -> None:
    first_report = tmp_path / "layer-1.json"
    second_report = tmp_path / "layer-2.json"
    write_report(
        first_report,
        activation_npz="features/layer-1.npz",
        vector_output="vectors/layer-1.npz",
        loo_accuracy=0.5,
        loo_margin=-0.1,
    )
    write_report(
        second_report,
        activation_npz="features/layer-2.npz",
        vector_output="vectors/layer-2.npz",
        loo_accuracy=0.75,
        loo_margin=0.2,
    )

    summary = aggregate_reports(
        [(-1, first_report), (-2, second_report)],
        model_id="test/model",
        layers=[-1, -2],
    )

    assert summary["model_id"] == "test/model"
    assert summary["layers"] == [-1, -2]
    assert summary["best_layer_by_leave_one_pair_out_accuracy"] == -2
    assert summary["reports"][0]["layer"] == -1
    assert summary["reports"][0]["leave_one_pair_out_pairwise_accuracy"] == 0.5
    assert summary["reports"][1]["activation_npz"] == "features/layer-2.npz"

    markdown = render_summary_markdown(summary)
    assert "# Activation Layer Sweep" in markdown
    assert "| -2 | 12 | 8 | 0.667 | 0.750 | +0.200 | `layer-2.json` |" in markdown


def test_custom_dataset_and_multi_model_paths_do_not_overwrite(tmp_path: Path) -> None:
    paths = layer_artifact_paths(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        layer=-1,
        dataset_name="autonomy_stress",
        features_root=tmp_path / "features",
        vectors_root=tmp_path / "vectors",
        reports_root=tmp_path / "reports",
    )
    summary_json, summary_markdown = summary_output_paths(
        reports_root=tmp_path / "reports",
        dataset_name="autonomy_stress",
        model_ids=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    )

    assert paths.activation_npz.name == (
        "autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__layer-1.npz"
    )
    assert paths.json_report.name == (
        "autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__layer-1.json"
    )
    assert summary_json.name == "autonomy_stress__multi_model__summary.json"
    assert summary_markdown.name == "autonomy_stress__multi_model__summary.md"


def write_report(
    path: Path,
    *,
    activation_npz: str,
    vector_output: str,
    loo_accuracy: float,
    loo_margin: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "activation_npz": activation_npz,
                "vector_output": vector_output,
                "n_prompts": 12,
                "activation_dim": 8,
                "in_sample": {
                    "n_pairs": 6,
                    "pairwise_accuracy": 0.667,
                    "mean_projection_margin": 0.5,
                },
                "leave_one_pair_out": {
                    "n_pairs": 6,
                    "pairwise_accuracy": loo_accuracy,
                    "mean_projection_margin": loo_margin,
                },
            }
        ),
        encoding="utf-8",
    )
