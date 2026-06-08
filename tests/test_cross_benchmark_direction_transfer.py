from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from social_cohesion_vectors.experiments.cross_benchmark_direction_transfer import (
    render_cross_benchmark_direction_transfer_markdown,
    run_cross_benchmark_direction_transfer_from_files,
)


def test_cross_benchmark_direction_transfer_scores_both_datasets(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_cross_benchmark_direction_transfer_from_files(
        source_vector_npz=paths["source_vector"],
        source_activation_npz=paths["source_activation"],
        target_vector_npz=paths["target_vector"],
        target_activation_npz=paths["target_activation"],
        source_name="generated",
        target_name="control",
    )
    markdown = render_cross_benchmark_direction_transfer_markdown(report)

    assert report["summary"]["ready_for_direction_transfer_claims"] is True
    assert report["summary"]["direction_cosine"] == pytest.approx(0.707107)
    assert report["summary"]["source_to_target_accuracy"] == 1.0
    assert report["summary"]["target_to_source_accuracy"] == 1.0
    assert report["summary"]["source_to_target_min_margin"] > 0.0
    assert report["summary"]["target_to_source_min_margin"] > 0.0
    assert "Cross-Benchmark Direction Transfer" in markdown


def test_cross_benchmark_direction_transfer_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    exit_code = script.main(
        [
            "--source-vector-npz",
            str(paths["source_vector"]),
            "--source-activation-npz",
            str(paths["source_activation"]),
            "--target-vector-npz",
            str(paths["target_vector"]),
            "--target-activation-npz",
            str(paths["target_activation"]),
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
    assert "cross-benchmark direction transfer" in captured.out
    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_direction_transfer_claims"] is True
    assert markdown_output.exists()


def test_cross_benchmark_direction_transfer_rejects_dimension_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    np.savez(tmp_path / "bad_vector.npz", direction=np.asarray([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="share a dimension"):
        run_cross_benchmark_direction_transfer_from_files(
            source_vector_npz=paths["source_vector"],
            source_activation_npz=paths["source_activation"],
            target_vector_npz=tmp_path / "bad_vector.npz",
            target_activation_npz=paths["target_activation"],
        )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_vector = tmp_path / "source_vector.npz"
    target_vector = tmp_path / "target_vector.npz"
    source_activation = tmp_path / "source_activation.npz"
    target_activation = tmp_path / "target_activation.npz"
    np.savez(source_vector, direction=np.asarray([1.0, 0.0], dtype=np.float64))
    np.savez(
        target_vector,
        direction=np.asarray([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
    )
    np.savez(
        source_activation,
        activations=np.asarray(
            [
                [2.0, 0.2],
                [0.1, 0.2],
                [1.5, -0.2],
                [-0.1, -0.2],
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
                [1.6, 1.2],
                [0.2, 0.1],
                [1.3, 1.0],
                [0.3, 0.2],
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
        "source_vector": source_vector,
        "source_activation": source_activation,
        "target_vector": target_vector,
        "target_activation": target_activation,
    }


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_cross_benchmark_direction_transfer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_cross_benchmark_direction_transfer",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
