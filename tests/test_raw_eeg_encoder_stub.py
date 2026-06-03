from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_raw_eeg_encoder_stub_fits_tiny_arrays(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    manifest = _write_stub_manifest(tmp_path)
    output = tmp_path / "report.json"

    report = cli_module.run_raw_eeg_encoder_stub(
        manifest_path=manifest,
        data_root=tmp_path,
        output_path=output,
        ridge_alpha=0.01,
        shuffle_seed=0,
    )

    assert report["dataset"] == "THINGS-EEG2"
    assert report["manifest_version"] == "raw-eeg-bridge-manifest-v0"
    assert report["feature_field"] == "visual_embedding_path"
    assert report["train_rows"] == 3
    assert report["eval_rows"] == 1
    assert report["feature_dim"] == 2
    assert report["response_dim"] == 2
    assert report["split_policy"] == "held-out-image"
    assert report["eval_split"] == "test"
    assert report["mse"] < report["shuffle_mse"]
    assert "no neural synchrony" in report["claim_boundary"]

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["mse_delta_vs_shuffle"] < 0.0


def test_raw_eeg_encoder_stub_rejects_absolute_manifest_paths(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    manifest = _write_stub_manifest(tmp_path)
    rows = _read_jsonl(manifest)
    rows[0]["features"]["visual_embedding_path"] = "/tmp/feature.npy"
    _write_jsonl(manifest, rows)

    with pytest.raises(ValueError, match="array path must be relative"):
        cli_module.run_raw_eeg_encoder_stub(
            manifest_path=manifest,
            data_root=tmp_path,
        )


def test_raw_eeg_encoder_stub_cli_writes_report(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    manifest = _write_stub_manifest(tmp_path)
    output = tmp_path / "cli-report.json"

    assert (
        cli_module.main(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(tmp_path),
                "--output",
                str(output),
                "--ridge-alpha",
                "0.01",
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["train_rows"] == 3
    assert report["eval_rows"] == 1
    assert report["claim_boundary"].startswith("Representation-learning encoder")


def _write_stub_manifest(tmp_path: Path) -> Path:
    rows = [
        _manifest_row("stim-001", "train", [1.0, 0.0], [1.0, 2.0]),
        _manifest_row("stim-002", "train", [0.0, 1.0], [3.0, 4.0]),
        _manifest_row("stim-003", "train", [1.0, 1.0], [4.0, 6.0]),
        _manifest_row("stim-004", "test", [2.0, 1.0], [5.0, 8.0]),
    ]
    for row in rows:
        feature_path = tmp_path / row["features"]["visual_embedding_path"]
        response_path = tmp_path / row["brain_response"]["response_path"]
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, np.asarray(row.pop("_feature"), dtype=np.float64))
        np.save(response_path, np.asarray(row.pop("_response"), dtype=np.float64))

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    return manifest


def _manifest_row(
    stimulus_id: str,
    split_name: str,
    feature: list[float],
    response: list[float],
) -> dict[str, Any]:
    return {
        "manifest_version": "raw-eeg-bridge-manifest-v0",
        "dataset": "THINGS-EEG2",
        "features": {
            "stimulus_id": stimulus_id,
            "visual_embedding_path": f"features/visual/{stimulus_id}.npy",
        },
        "brain_response": {
            "stimulus_id": stimulus_id,
            "response_path": f"responses/{stimulus_id}.npy",
        },
        "quality": {"include": True},
        "split": {
            "split_name": split_name,
            "split_policy": "held-out-image",
            "split_version": "2026-06-03-v0",
        },
        "_feature": feature,
        "_response": response,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "run_raw_eeg_encoder_stub.py"
    spec = importlib.util.spec_from_file_location(
        "run_raw_eeg_encoder_stub",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
