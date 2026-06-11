from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.heldout_domain_direction_audit import (
    render_bridge_direction_comparison_markdown,
    render_bridge_set_sufficiency_audit_markdown,
    render_cross_model_bridge_transport_markdown,
    render_fresh_generated_bridge_diagnostic_markdown,
    render_heldout_domain_direction_audit_markdown,
    render_minimal_bridge_direction_audit_markdown,
    render_pair_bridge_direction_audit_markdown,
    run_bridge_direction_comparison_from_files,
    run_bridge_set_sufficiency_audit_from_files,
    run_cross_model_bridge_transport_from_files,
    run_fresh_generated_bridge_diagnostic_from_files,
    run_heldout_domain_direction_audit_from_files,
    run_minimal_bridge_direction_audit_from_files,
    run_pair_bridge_direction_audit_from_files,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_heldout_domain_direction_audit_bridge_trains_across_domains(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_heldout_domain_direction_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        source_name="generated",
        target_name="control",
    )
    markdown = render_heldout_domain_direction_audit_markdown(report)

    assert report["summary"]["ready_for_heldout_domain_claims"] is True
    assert report["summary"]["source_holdout_folds"] == 2
    assert report["summary"]["target_holdout_folds"] == 2
    assert report["summary"]["source_holdout_min_accuracy"] == 1.0
    assert report["summary"]["target_holdout_min_accuracy"] == 1.0
    assert report["summary"]["failed_pair_count"] == 0
    assert report["source_holdout_folds"][0]["train_primary_dataset"] == "control"
    assert report["target_holdout_folds"][0]["train_primary_dataset"] == "generated"
    assert "Held-Out Domain Direction Audit" in markdown
    assert "No failed pairs." in markdown


def test_heldout_domain_direction_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "held-out domain direction audit" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_heldout_domain_claims"] is True
    assert markdown_output.exists()


def test_minimal_bridge_direction_audit_reports_bridge_counts(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_minimal_bridge_direction_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        source_name="generated",
        target_name="control",
    )
    markdown = render_minimal_bridge_direction_audit_markdown(report)

    assert report["summary"]["ready_for_minimal_bridge_claims"] is True
    assert report["summary"]["source_min_ready_bridge_groups"] == 1
    assert report["summary"]["target_min_ready_bridge_groups"] == 1
    assert report["source_by_bridge_count"][0]["bridge_group_count"] == 0
    assert report["source_by_bridge_count"][0]["ready"] is False
    assert report["source_by_bridge_count"][1]["bridge_group_count"] == 1
    assert report["source_by_bridge_count"][1]["ready"] is True
    assert report["target_by_bridge_count"][0]["ready"] is False
    assert report["target_by_bridge_count"][1]["ready"] is True
    assert "Minimal Bridge Direction Audit" in markdown
    assert "Failed Ablation Folds" in markdown


def test_minimal_bridge_direction_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_minimal_bridge_direction_audit.py")
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "minimal.json"
    markdown_output = tmp_path / "minimal.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "minimal bridge direction audit" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["source_min_ready_bridge_groups"] == 1
    assert markdown_output.exists()


def test_pair_bridge_direction_audit_reports_path_stratified_counts(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_pair_bridge_direction_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        source_name="generated",
        target_name="control",
        max_subsets_per_count=4,
    )
    markdown = render_pair_bridge_direction_audit_markdown(report)

    assert report["summary"]["ready_for_pair_bridge_claims"] is True
    assert report["summary"]["source_min_ready_bridge_pairs"] == 1
    assert report["summary"]["target_min_ready_bridge_pairs"] == 1
    assert report["source_by_bridge_pair_count"][0]["bridge_pair_count"] == 0
    assert report["source_by_bridge_pair_count"][0]["ready"] is False
    assert report["source_by_bridge_pair_count"][1]["bridge_pair_count"] == 1
    assert report["source_by_bridge_pair_count"][1]["ready"] is True
    assert report["source_by_bridge_pair_count"][1]["min_bridge_path_count"] == 1
    assert report["source_pair_ablation_folds"][0]["bridge_path_values"] == []
    assert "Pair Bridge Direction Audit" in markdown
    assert "Target Holdout By Bridge Pair Count" in markdown


def test_pair_bridge_direction_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_pair_bridge_direction_audit.py")
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "pair.json"
    markdown_output = tmp_path / "pair.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--max-subsets-per-count",
            "4",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "pair bridge direction audit" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["target_min_ready_bridge_pairs"] == 1
    assert markdown_output.exists()


def test_bridge_set_sufficiency_audit_constructs_ready_sets(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_bridge_set_sufficiency_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        source_name="generated",
        target_name="control",
        bridge_pair_count=1,
    )
    markdown = render_bridge_set_sufficiency_audit_markdown(report)

    assert report["summary"]["ready_for_bridge_set_claims"] is True
    assert report["summary"]["failed_fold_count"] == 0
    assert report["summary"]["source_path_complete_folds"] == 2
    assert report["summary"]["target_path_complete_folds"] == 2
    assert report["source_bridge_set_folds"][0]["path_complete"] is True
    assert report["source_bridge_set_folds"][0]["bridge_pair_count"] == 1
    assert "Bridge Set Sufficiency Audit" in markdown
    assert "No failed bridge sets." in markdown


def test_bridge_set_sufficiency_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_bridge_set_sufficiency_audit.py")
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "bridge_set.json"
    markdown_output = tmp_path / "bridge_set.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--bridge-pair-count",
            "1",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "bridge-set sufficiency audit" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_bridge_set_claims"] is True
    assert markdown_output.exists()


def test_bridge_direction_comparison_scores_constructed_directions(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_bridge_direction_comparison_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        source_name="generated",
        target_name="control",
        bridge_pair_count=1,
    )
    markdown = render_bridge_direction_comparison_markdown(report)

    assert report["summary"]["ready_for_constructed_bridge_direction_claims"] is True
    assert report["summary"]["constructed_direction_count"] == 4
    assert report["summary"]["constructed_source_min_accuracy"] == 1.0
    assert report["summary"]["constructed_target_min_accuracy"] == 1.0
    assert report["summary"]["constructed_min_joint_cosine"] > 0.0
    assert "Bridge Direction Comparison" in markdown
    assert "No failed constructed directions." in markdown


def test_bridge_direction_comparison_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_bridge_direction_comparison.py")
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "bridge_direction.json"
    markdown_output = tmp_path / "bridge_direction.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--bridge-pair-count",
            "1",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "bridge direction comparison" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_constructed_bridge_direction_claims"] is True
    assert markdown_output.exists()


def test_fresh_generated_bridge_diagnostic_scores_constructed_directions(
    tmp_path: Path,
) -> None:
    paths = _write_fresh_bridge_diagnostic_fixture(tmp_path)

    report = run_fresh_generated_bridge_diagnostic_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        fresh_source_activation_npz=paths["fresh_source_activation"],
        fresh_source_pairs_path=paths["fresh_source_pairs"],
        fresh_target_activation_npz=paths["fresh_target_activation"],
        fresh_target_pairs_path=paths["fresh_target_pairs"],
        source_name="generated",
        target_name="control",
        fresh_source_name="fresh_generated",
        fresh_target_name="fresh_control",
        bridge_pair_count=1,
        target_bridge_secondary_repetitions=2,
    )
    markdown = render_fresh_generated_bridge_diagnostic_markdown(report)

    assert report["summary"]["ready_for_fresh_generated_bridge_claims"] is True
    assert report["summary"]["constructed_direction_count"] == 4
    assert report["summary"]["constructed_fresh_source_min_margin"] > 0.0
    assert report["summary"]["constructed_fresh_target_min_margin"] > 0.0
    assert report["inputs"]["target_bridge_secondary_repetitions"] == 2
    assert report["summary"]["source_fresh_joint_fresh_target_min_margin"] == 0.0
    assert report["summary"]["failed_pair_evaluation_count"] > 0
    assert "Fresh Generated Bridge Diagnostic" in markdown
    assert "Target bridge repetitions" in markdown
    assert "`source_fresh_joint`" in markdown


def test_fresh_generated_bridge_diagnostic_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_fresh_generated_bridge_diagnostic.py")
    paths = _write_fresh_bridge_diagnostic_fixture(tmp_path)
    json_output = tmp_path / "fresh_bridge.json"
    markdown_output = tmp_path / "fresh_bridge.md"

    exit_code = script.main(
        [
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--source-pairs",
            str(paths["source_pairs"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
            "--target-pairs",
            str(paths["target_pairs"]),
            "--fresh-source-activation-npz",
            str(paths["fresh_source_activation"]),
            "--fresh-source-pairs",
            str(paths["fresh_source_pairs"]),
            "--fresh-target-activation-npz",
            str(paths["fresh_target_activation"]),
            "--fresh-target-pairs",
            str(paths["fresh_target_pairs"]),
            "--source-name",
            "generated",
            "--target-name",
            "control",
            "--fresh-source-name",
            "fresh_generated",
            "--fresh-target-name",
            "fresh_control",
            "--bridge-pair-count",
            "1",
            "--target-bridge-secondary-repetitions",
            "2",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fresh generated bridge diagnostic" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_fresh_generated_bridge_claims"] is True
    assert loaded["inputs"]["fresh_source_name"] == "fresh_generated"
    assert loaded["inputs"]["target_bridge_secondary_repetitions"] == 2
    assert markdown_output.exists()


def test_fresh_generated_bridge_diagnostic_rejects_invalid_repetitions(
    tmp_path: Path,
) -> None:
    paths = _write_fresh_bridge_diagnostic_fixture(tmp_path)

    with pytest.raises(ValueError, match="repetitions must be at least 1"):
        run_fresh_generated_bridge_diagnostic_from_files(
            source_activation_npz=paths["source_activation"],
            source_pairs_path=paths["source_pairs"],
            target_activation_npz=paths["target_activation"],
            target_pairs_path=paths["target_pairs"],
            fresh_source_activation_npz=paths["fresh_source_activation"],
            fresh_source_pairs_path=paths["fresh_source_pairs"],
            fresh_target_activation_npz=paths["fresh_target_activation"],
            fresh_target_pairs_path=paths["fresh_target_pairs"],
            target_bridge_secondary_repetitions=0,
        )


def test_cross_model_bridge_transport_maps_constructed_directions(
    tmp_path: Path,
) -> None:
    paths = _write_cross_model_fixture(tmp_path)

    report = run_cross_model_bridge_transport_from_files(
        model_a_source_activation_npz=paths["model_a_source_activation"],
        model_a_source_pairs_path=paths["source_pairs"],
        model_a_target_activation_npz=paths["model_a_target_activation"],
        model_a_target_pairs_path=paths["target_pairs"],
        model_b_source_activation_npz=paths["model_b_source_activation"],
        model_b_source_pairs_path=paths["source_pairs"],
        model_b_target_activation_npz=paths["model_b_target_activation"],
        model_b_target_pairs_path=paths["target_pairs"],
        model_a_fresh_source_activation_npz=paths["model_a_fresh_source_activation"],
        model_a_fresh_source_pairs_path=paths["fresh_source_pairs"],
        model_a_fresh_target_activation_npz=paths["model_a_fresh_target_activation"],
        model_a_fresh_target_pairs_path=paths["fresh_target_pairs"],
        model_b_fresh_source_activation_npz=paths["model_b_fresh_source_activation"],
        model_b_fresh_source_pairs_path=paths["fresh_source_pairs"],
        model_b_fresh_target_activation_npz=paths["model_b_fresh_target_activation"],
        model_b_fresh_target_pairs_path=paths["fresh_target_pairs"],
        model_a_name="a",
        model_b_name="b",
        bridge_pair_count=1,
        knn_k=1,
        ridge_alpha=1e-6,
    )
    markdown = render_cross_model_bridge_transport_markdown(report)

    assert report["summary"]["ready_for_cross_model_bridge_transport_claims"] is True
    assert report["summary"]["model_a_to_b_direction_count"] == 4
    assert report["summary"]["model_b_to_a_direction_count"] == 4
    assert report["summary"]["model_a_to_b_min_bridge_cosine"] > 0.0
    assert report["summary"]["model_b_to_a_min_bridge_cosine"] > 0.0
    assert report["summary"]["model_a_to_b_source_min_margin"] > 0.0
    assert report["summary"]["model_b_to_a_target_min_margin"] > 0.0
    assert report["summary"]["model_a_to_b_leave_heldout_min_margin"] > 0.0
    assert report["summary"]["model_b_to_a_leave_heldout_min_margin"] > 0.0
    assert report["summary"]["model_a_to_b_fresh_source_min_margin"] > 0.0
    assert report["summary"]["model_b_to_a_fresh_target_min_margin"] > 0.0
    assert "Cross-Model Bridge Transport Audit" in markdown
    assert "fresh source min accuracy/margin" in markdown
    assert "No failed transported directions." in markdown


def test_cross_model_bridge_transport_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script("run_cross_model_bridge_transport.py")
    paths = _write_cross_model_fixture(tmp_path)
    json_output = tmp_path / "cross_model_bridge_transport.json"
    markdown_output = tmp_path / "cross_model_bridge_transport.md"

    exit_code = script.main(
        [
            "--model-a-source-activation-npz",
            str(paths["model_a_source_activation"]),
            "--model-a-source-pairs",
            str(paths["source_pairs"]),
            "--model-a-target-activation-npz",
            str(paths["model_a_target_activation"]),
            "--model-a-target-pairs",
            str(paths["target_pairs"]),
            "--model-b-source-activation-npz",
            str(paths["model_b_source_activation"]),
            "--model-b-source-pairs",
            str(paths["source_pairs"]),
            "--model-b-target-activation-npz",
            str(paths["model_b_target_activation"]),
            "--model-b-target-pairs",
            str(paths["target_pairs"]),
            "--model-a-fresh-source-activation-npz",
            str(paths["model_a_fresh_source_activation"]),
            "--model-a-fresh-source-pairs",
            str(paths["fresh_source_pairs"]),
            "--model-a-fresh-target-activation-npz",
            str(paths["model_a_fresh_target_activation"]),
            "--model-a-fresh-target-pairs",
            str(paths["fresh_target_pairs"]),
            "--model-b-fresh-source-activation-npz",
            str(paths["model_b_fresh_source_activation"]),
            "--model-b-fresh-source-pairs",
            str(paths["fresh_source_pairs"]),
            "--model-b-fresh-target-activation-npz",
            str(paths["model_b_fresh_target_activation"]),
            "--model-b-fresh-target-pairs",
            str(paths["fresh_target_pairs"]),
            "--fresh-source-name",
            "fresh_generated",
            "--fresh-target-name",
            "fresh_control",
            "--bridge-pair-count",
            "1",
            "--knn-k",
            "1",
            "--ridge-alpha",
            "1e-6",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "cross-model bridge transport" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_cross_model_bridge_transport_claims"] is True
    assert loaded["inputs"]["fresh_source_name"] == "fresh_generated"
    assert loaded["summary"]["model_a_to_b_fresh_target_min_margin"] > 0.0
    assert markdown_output.exists()


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_pairs = tmp_path / "source_pairs.jsonl"
    target_pairs = tmp_path / "target_pairs.jsonl"
    source_activation = tmp_path / "source_activation.npz"
    target_activation = tmp_path / "target_activation.npz"
    write_jsonl(
        [
            _pair("s1", "source_a"),
            _pair("s2", "source_b"),
        ],
        source_pairs,
    )
    write_jsonl(
        [
            _pair("t1", "target_a"),
            _pair("t2", "target_b"),
        ],
        target_pairs,
    )
    np.savez(
        source_activation,
        activations=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(["s1", "s1", "s2", "s2"], dtype=str),
        labels=np.asarray(
            ["positive", "negative", "positive", "negative"],
            dtype=str,
        ),
    )
    np.savez(
        target_activation,
        activations=np.asarray(
            [
                [0.0, 2.0],
                [0.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(["t1", "t1", "t2", "t2"], dtype=str),
        labels=np.asarray(
            ["positive", "negative", "positive", "negative"],
            dtype=str,
        ),
    )
    return {
        "source_pairs": source_pairs,
        "target_pairs": target_pairs,
        "source_activation": source_activation,
        "target_activation": target_activation,
    }


def _write_fresh_bridge_diagnostic_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _write_fixture(tmp_path)
    fresh_source_pairs = tmp_path / "fresh_source_pairs.jsonl"
    fresh_target_pairs = tmp_path / "fresh_target_pairs.jsonl"
    fresh_source_activation = tmp_path / "fresh_source_activation.npz"
    fresh_target_activation = tmp_path / "fresh_target_activation.npz"
    write_jsonl([_pair("fs1", "source_c")], fresh_source_pairs)
    write_jsonl([_pair("ft1", "target_c")], fresh_target_pairs)
    _write_activation(
        fresh_source_activation,
        activations=np.asarray([[1.5, 0.0], [0.0, 0.0]], dtype=np.float64),
        pair_ids=np.asarray(["fs1", "fs1"], dtype=str),
    )
    _write_activation(
        fresh_target_activation,
        activations=np.asarray([[0.0, 1.5], [0.0, 0.0]], dtype=np.float64),
        pair_ids=np.asarray(["ft1", "ft1"], dtype=str),
    )
    return {
        **paths,
        "fresh_source_pairs": fresh_source_pairs,
        "fresh_target_pairs": fresh_target_pairs,
        "fresh_source_activation": fresh_source_activation,
        "fresh_target_activation": fresh_target_activation,
    }


def _write_cross_model_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _write_fixture(tmp_path)
    model_a_source = paths["source_activation"]
    model_a_target = paths["target_activation"]
    model_b_source = tmp_path / "model_b_source_activation.npz"
    model_b_target = tmp_path / "model_b_target_activation.npz"
    fresh_source_pairs = tmp_path / "fresh_source_pairs.jsonl"
    fresh_target_pairs = tmp_path / "fresh_target_pairs.jsonl"
    model_a_fresh_source = tmp_path / "model_a_fresh_source_activation.npz"
    model_a_fresh_target = tmp_path / "model_a_fresh_target_activation.npz"
    model_b_fresh_source = tmp_path / "model_b_fresh_source_activation.npz"
    model_b_fresh_target = tmp_path / "model_b_fresh_target_activation.npz"
    transform = np.asarray(
        [
            [1.0, 0.5],
            [0.5, 1.0],
        ],
        dtype=np.float64,
    )
    _write_transformed_activation(model_a_source, model_b_source, transform=transform)
    _write_transformed_activation(model_a_target, model_b_target, transform=transform)
    write_jsonl([_pair("fs1", "source_c")], fresh_source_pairs)
    write_jsonl([_pair("ft1", "target_c")], fresh_target_pairs)
    _write_activation(
        model_a_fresh_source,
        activations=np.asarray([[1.5, 0.0], [0.0, 0.0]], dtype=np.float64),
        pair_ids=np.asarray(["fs1", "fs1"], dtype=str),
    )
    _write_activation(
        model_a_fresh_target,
        activations=np.asarray([[0.0, 1.5], [0.0, 0.0]], dtype=np.float64),
        pair_ids=np.asarray(["ft1", "ft1"], dtype=str),
    )
    _write_transformed_activation(
        model_a_fresh_source,
        model_b_fresh_source,
        transform=transform,
    )
    _write_transformed_activation(
        model_a_fresh_target,
        model_b_fresh_target,
        transform=transform,
    )
    return {
        "source_pairs": paths["source_pairs"],
        "target_pairs": paths["target_pairs"],
        "fresh_source_pairs": fresh_source_pairs,
        "fresh_target_pairs": fresh_target_pairs,
        "model_a_source_activation": model_a_source,
        "model_a_target_activation": model_a_target,
        "model_b_source_activation": model_b_source,
        "model_b_target_activation": model_b_target,
        "model_a_fresh_source_activation": model_a_fresh_source,
        "model_a_fresh_target_activation": model_a_fresh_target,
        "model_b_fresh_source_activation": model_b_fresh_source,
        "model_b_fresh_target_activation": model_b_fresh_target,
    }


def _write_activation(
    path: Path,
    *,
    activations: np.ndarray,
    pair_ids: np.ndarray,
) -> None:
    labels = np.asarray(["positive", "negative"], dtype=str)
    sample_ids = np.asarray(
        [f"{pair_id}:{label}" for pair_id, label in zip(pair_ids, labels, strict=True)],
        dtype=str,
    )
    np.savez(
        path,
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        sample_ids=sample_ids,
    )


def _write_transformed_activation(
    source_path: Path,
    target_path: Path,
    *,
    transform: np.ndarray,
) -> None:
    with np.load(source_path, allow_pickle=False) as data:
        activations = np.asarray(data["activations"], dtype=np.float64)
        pair_ids = np.asarray(data["pair_ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=str)
    sample_ids = np.asarray(
        [f"{pair_id}:{label}" for pair_id, label in zip(pair_ids, labels, strict=True)],
        dtype=str,
    )
    np.savez(
        target_path,
        activations=activations @ transform,
        pair_ids=pair_ids,
        labels=labels,
        sample_ids=sample_ids,
    )
    np.savez(
        source_path,
        activations=activations,
        pair_ids=pair_ids,
        labels=labels,
        sample_ids=sample_ids,
    )


def _pair(pair_id: str, source: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata={
            "source": source,
            "slack_options_tested": f"path_{pair_id}",
        },
    )


def _load_script(filename: str = "run_heldout_domain_direction_audit.py") -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(
        Path(filename).stem,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
