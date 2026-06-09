from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.fresh_pair_geometry_audit import (
    run_fresh_pair_geometry_audit_from_files,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_pair_geometry_audit_detects_hard_residual(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = run_fresh_pair_geometry_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        fresh_source_activation_npz=paths["fresh_activation"],
        fresh_source_pairs_path=paths["fresh_pairs"],
        focus_base_contrast_id="accountability_after_harm",
    )

    assert report["summary"]["hard_pair_geometry_residual"] is True
    assert report["summary"]["original_joint_focus_margin"] < 0.0
    assert report["summary"]["full_augmented_focus_margin"] < 0.0
    assert report["summary"]["fresh_only_focus_margin"] > 0.0
    assert report["summary"]["source_same_base_original_min_margin"] > 0.0
    assert report["summary"]["source_same_base_pairs"] == 1


def test_pair_geometry_cli_writes_reports(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    script = _load_script("run_fresh_pair_geometry_audit.py")

    assert (
        script.main(
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
                str(paths["fresh_activation"]),
                "--fresh-source-pairs",
                str(paths["fresh_pairs"]),
                "--focus-base-contrast-id",
                "accountability_after_harm",
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["hard_pair_geometry_residual"] is True
    assert "# Fresh Pair Geometry Audit" in markdown_output.read_text(encoding="utf-8")


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_pairs = tmp_path / "source_pairs.jsonl"
    target_pairs_path = tmp_path / "target_pairs.jsonl"
    fresh_pairs = tmp_path / "fresh_pairs.jsonl"
    source_activation = tmp_path / "source.npz"
    target_activation = tmp_path / "target.npz"
    fresh_activation = tmp_path / "fresh.npz"
    source_pair = _pair(
        "generated-fault::accountability_after_harm__source",
        "accountability_after_harm",
    )
    target_pair_records = [
        _pair(f"target_case_{index}", "target_case") for index in range(10)
    ]
    focus_pair = _pair(
        "generated-fault::accountability_after_harm__fresh",
        "accountability_after_harm",
    )
    other_fresh = _pair("generated-fault::care_boundary__fresh", "care_boundary")
    write_jsonl([source_pair], source_pairs)
    write_jsonl(target_pair_records, target_pairs_path)
    write_jsonl([focus_pair, other_fresh], fresh_pairs)
    _write_activation(source_activation, {source_pair.pair_id: ([20.0, 0.0], [0.0, 0.0])})
    _write_activation(
        target_activation,
        {pair.pair_id: ([20.0, 0.0], [0.0, 0.0]) for pair in target_pair_records},
    )
    _write_activation(
        fresh_activation,
        {
            focus_pair.pair_id: ([0.0, 1.0], [10.0, 0.0]),
            other_fresh.pair_id: ([0.0, 3.0], [0.0, 0.0]),
        },
    )
    return {
        "source_pairs": source_pairs,
        "target_pairs": target_pairs_path,
        "fresh_pairs": fresh_pairs,
        "source_activation": source_activation,
        "target_activation": target_activation,
        "fresh_activation": fresh_activation,
    }


def _write_activation(
    path: Path,
    pairs: dict[str, tuple[list[float], list[float]]],
) -> None:
    activations: list[list[float]] = []
    pair_ids: list[str] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for pair_id, (positive, negative) in pairs.items():
        for label, vector in (("positive", positive), ("negative", negative)):
            pair_ids.append(pair_id)
            labels.append(label)
            sample_ids.append(f"{pair_id}:{label}")
            activations.append(vector)
    np.savez(
        path,
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
        sample_ids=np.asarray(sample_ids, dtype=str),
    )


def _pair(pair_id: str, base_contrast_id: str) -> PairwiseExample:
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
            "base_contrast_id": base_contrast_id,
            "source": "fixture",
        },
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
