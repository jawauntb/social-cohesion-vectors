from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.fresh_augmented_direction_audit import (
    run_fresh_augmented_direction_audit_from_files,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_fresh_augmented_direction_audit_passes_leave_one_out(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = run_fresh_augmented_direction_audit_from_files(
        source_activation_npz=paths["source_activation"],
        source_pairs_path=paths["source_pairs"],
        target_activation_npz=paths["target_activation"],
        target_pairs_path=paths["target_pairs"],
        fresh_source_activation_npz=paths["fresh_source_activation"],
        fresh_source_pairs_path=paths["fresh_source_pairs"],
        fresh_target_activation_npz=paths["fresh_target_activation"],
        fresh_target_pairs_path=paths["fresh_target_pairs"],
    )

    assert report["summary"]["ready_for_fresh_augmented_direction_claims"] is True
    assert report["summary"]["fresh_loo_failed_heldout_pairs"] == 0
    assert report["summary"]["fresh_loo_heldout_min_margin"] > 0.0


def test_fresh_augmented_direction_cli_writes_report(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    script = _load_script("run_fresh_augmented_direction_audit.py")

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
                str(paths["fresh_source_activation"]),
                "--fresh-source-pairs",
                str(paths["fresh_source_pairs"]),
                "--fresh-target-activation-npz",
                str(paths["fresh_target_activation"]),
                "--fresh-target-pairs",
                str(paths["fresh_target_pairs"]),
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_fresh_augmented_direction_claims"] is True
    assert "# Fresh-Augmented Direction Audit" in markdown_output.read_text(
        encoding="utf-8"
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_pairs = tmp_path / "source_pairs.jsonl"
    target_pairs = tmp_path / "target_pairs.jsonl"
    fresh_source_pairs = tmp_path / "fresh_source_pairs.jsonl"
    fresh_target_pairs = tmp_path / "fresh_target_pairs.jsonl"
    source_activation = tmp_path / "source.npz"
    target_activation = tmp_path / "target.npz"
    fresh_source_activation = tmp_path / "fresh_source.npz"
    fresh_target_activation = tmp_path / "fresh_target.npz"
    write_jsonl([_pair("source_a")], source_pairs)
    write_jsonl([_pair("target_a")], target_pairs)
    write_jsonl([_pair("fresh_a"), _pair("fresh_b")], fresh_source_pairs)
    write_jsonl([_pair("fresh_target_a")], fresh_target_pairs)
    _write_activation(source_activation, {"source_a": (4.0, 0.0)})
    _write_activation(target_activation, {"target_a": (3.0, 0.0)})
    _write_activation(fresh_source_activation, {"fresh_a": (2.0, 0.0), "fresh_b": (2.2, 0.0)})
    _write_activation(fresh_target_activation, {"fresh_target_a": (1.5, 0.0)})
    return {
        "source_pairs": source_pairs,
        "target_pairs": target_pairs,
        "fresh_source_pairs": fresh_source_pairs,
        "fresh_target_pairs": fresh_target_pairs,
        "source_activation": source_activation,
        "target_activation": target_activation,
        "fresh_source_activation": fresh_source_activation,
        "fresh_target_activation": fresh_target_activation,
    }


def _write_activation(path: Path, pairs: dict[str, tuple[float, float]]) -> None:
    activations: list[list[float]] = []
    pair_ids: list[str] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for pair_id, (positive_x, negative_x) in pairs.items():
        for label, x_value in (("positive", positive_x), ("negative", negative_x)):
            pair_ids.append(pair_id)
            labels.append(label)
            sample_ids.append(f"{pair_id}:{label}")
            activations.append([x_value, 0.0])
    np.savez(
        path,
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
        sample_ids=np.asarray(sample_ids, dtype=str),
    )


def _pair(pair_id: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata={"source": "fixture"},
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
