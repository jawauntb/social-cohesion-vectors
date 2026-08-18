from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_activation_vector_experiment import run_activation_vector_experiment
from scripts.run_modal_activation_extraction import (
    collect_remote_activations,
    default_output_path,
    model_slug,
    run_modal_activation_extraction,
)
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    build_activation_npz_bytes,
    load_activation_npz_bytes,
)


def test_activation_npz_payload_round_trips_without_torch() -> None:
    records = _records()
    activations = np.array([[-1.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    payload = build_activation_npz_bytes(
        records=records,
        activations=activations,
        model_id="test/model",
        layer=-1,
    )
    arrays = load_activation_npz_bytes(payload)

    assert payload[:2] == b"PK"
    assert np.allclose(arrays["activations"], activations)
    assert arrays["sample_ids"].tolist() == ["pair-a:negative", "pair-a:positive"]
    assert arrays["labels"].tolist() == ["negative", "positive"]
    assert str(arrays["model_id"].item()) == "test/model"
    assert int(arrays["layer"].item()) == -1


def test_modal_extraction_runner_collects_fake_remote_chunks(tmp_path: Path) -> None:
    records = [
        _record("pair-a:negative", "pair-a", "negative", 0.0),
        _record("pair-a:positive", "pair-a", "positive", 1.0),
        _record("pair-b:negative", "pair-b", "negative", 0.0),
    ]
    fake_remote = FakeRemote()

    activations, summaries = collect_remote_activations(
        records,
        remote_function=fake_remote,
        model_id="Org/Test Model",
        layer=3,
        chunk_size=2,
        batch_size=1,
        max_length=32,
    )

    assert len(fake_remote.calls) == 2
    assert summaries == [{"count": 2}, {"count": 1}]
    assert activations.shape == (3, 2)
    assert np.allclose(activations[0], np.array([0.0, 1.0], dtype=np.float32))

    input_path = tmp_path / "activation_prompts.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features" / "out.npz"
    summary = run_modal_activation_extraction(
        input_path=input_path,
        output_path=output_path,
        model_id="Org/Test Model",
        layer=3,
        chunk_size=2,
        batch_size=1,
        max_length=32,
        remote_function=FakeRemote(),
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert data["activations"].shape == (3, 2)
        assert data["pair_ids"].tolist() == ["pair-a", "pair-a", "pair-b"]
    assert summary["output_path"] == str(output_path)
    assert (
        default_output_path(
            input_path,
            model_id="Org/Test Model",
            layer=3,
            features_root=tmp_path,
        )
        == tmp_path / "open_llm" / "activation_prompts__org-test-model__layer3.npz"
    )
    assert model_slug("Org/Test Model") == "org-test-model"


def test_activation_vector_experiment_writes_reports_and_vector(tmp_path: Path) -> None:
    activation_path = tmp_path / "activation_prompts__test-model__layer-1.npz"
    records = [
        _record("pair-a:negative", "pair-a", "negative", 0.0),
        _record("pair-a:positive", "pair-a", "positive", 1.0),
        _record("pair-b:negative", "pair-b", "negative", 0.0),
        _record("pair-b:positive", "pair-b", "positive", 1.0),
    ]
    activations = np.array(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    activation_path.write_bytes(
        build_activation_npz_bytes(
            records=records,
            activations=activations,
            model_id="test/model",
            layer=-1,
        )
    )
    vector_path = tmp_path / "models" / "vector.npz"
    report_json = tmp_path / "reports" / "activation_vector_experiment.json"
    report_md = tmp_path / "reports" / "activation_vector_experiment.md"

    summary = run_activation_vector_experiment(
        activation_path=activation_path,
        vector_output_path=vector_path,
        report_json_path=report_json,
        report_md_path=report_md,
        top_fraction=0.5,
    )

    assert summary["pairwise_accuracy"]["accuracy"] == 1.0
    assert summary["pairwise_accuracy"]["pair_count"] == 2
    assert vector_path.exists()
    assert report_json.exists()
    assert report_md.exists()

    saved = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved["vector_path"] == str(vector_path)
    assert "not evidence of real human" in report_md.read_text(encoding="utf-8")


class FakeRemote:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def remote(self, **kwargs: Any) -> dict[str, Any]:
        records = kwargs["records"]
        start = len(self.calls) * 10
        activations = np.arange(start, start + len(records) * 2, dtype=np.float32)
        activations = activations.reshape(len(records), 2)
        self.calls.append(kwargs)
        return {
            "npz_bytes": build_activation_npz_bytes(
                records=records,
                activations=activations,
                model_id=kwargs["model_id"],
                layer=kwargs["layer"],
            ),
            "summary": {"count": len(records)},
        }


def _records() -> list[dict[str, Any]]:
    return [
        _record("pair-a:negative", "pair-a", "negative", 0.0),
        _record("pair-a:positive", "pair-a", "positive", 1.0),
    ]


def _record(
    sample_id: str,
    pair_id: str,
    label: str,
    score: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "pair_id": pair_id,
        "label": label,
        "target_score": score,
        "text": f"{label} text for {pair_id}",
    }
