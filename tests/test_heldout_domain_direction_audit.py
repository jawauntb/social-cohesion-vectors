from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.heldout_domain_direction_audit import (
    render_heldout_domain_direction_audit_markdown,
    render_minimal_bridge_direction_audit_markdown,
    render_pair_bridge_direction_audit_markdown,
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
