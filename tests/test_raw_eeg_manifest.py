from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.raw_eeg_manifest import (
    CLAIM_BOUNDARY,
    build_raw_eeg_manifest,
    validate_raw_eeg_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_raw_eeg_manifest_joins_local_metadata_stubs(tmp_path: Path) -> None:
    paths = _write_stub_tables(tmp_path)

    records = _build_manifest_from_paths(paths)

    assert len(records) == 2
    first = records[0]
    second = records[1]

    assert first["manifest_version"] == "raw-eeg-bridge-manifest-v0"
    assert first["dataset"] == "THINGS-EEG2"
    assert first["claim_boundary"] == CLAIM_BOUNDARY
    assert first["provenance"]["source_doi"] == "https://doi.org/10.1016/j.neuroimage.2022.119754"
    assert first["stimulus"]["stimulus_id"] == "things-eeg2-img-000001"
    assert first["stimulus"]["source_corpus"] == "THINGS"
    assert first["features"]["feature_row_id"] == (
        "THINGS-EEG2:things-eeg2-img-000001:2026-06-03-models-v0"
    )
    assert first["features"]["visual_embedding_path"] == (
        "features/visual/things-eeg2-img-000001.npy"
    )
    assert first["brain_response"]["response_path"] == (
        "restricted/responses/sub-01/ses-01/run-01/trial-000001_epoch.npy"
    )
    assert first["subject_session"] == {
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "run_id": "run-01",
        "trial_id": "trial-000001",
        "repetition_index": 1,
        "onset_ms": 12345,
    }
    assert first["split"] == {
        "split_name": "train",
        "split_policy": "held-out-image",
        "split_version": "2026-06-03-v0",
    }
    assert first["quality"] == {
        "include": True,
        "exclusion_reason": None,
        "quality_flags": [],
    }
    assert second["quality"]["include"] is False
    assert second["quality"]["quality_flags"] == ["blink", "high_impedance"]

    validation = validate_raw_eeg_manifest(records)
    assert validation.valid
    assert validation.errors == ()


def test_build_raw_eeg_manifest_rejects_orphaned_response(tmp_path: Path) -> None:
    paths = _write_stub_tables(tmp_path)
    _write_csv(
        paths["responses_path"],
        [
            {
                **_response_row("trial-000001", "missing-stimulus"),
                "response_path": "restricted/responses/sub-01/missing.npy",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing stimulus_id"):
        _build_manifest_from_paths(paths)


def test_validate_raw_eeg_manifest_rejects_absolute_paths_and_duplicates(
    tmp_path: Path,
) -> None:
    records = _build_manifest_from_paths(_write_stub_tables(tmp_path))
    records[0]["brain_response"]["response_path"] = "/tmp/raw/sub-01.npy"
    records.append(records[1])

    validation = validate_raw_eeg_manifest(records)

    assert not validation.valid
    assert any("response_path must be a relative path" in error for error in validation.errors)
    assert any("duplicate response trial key" in error for error in validation.errors)


def test_cli_writes_jsonl_manifest_from_stub_tables(tmp_path: Path) -> None:
    paths = _write_stub_tables(tmp_path)
    output = tmp_path / "manifest.jsonl"
    cli_module = _load_cli_module()

    assert (
        cli_module.main(
            [
                "--dataset-access",
                str(paths["dataset_access_path"]),
                "--stimuli",
                str(paths["stimuli_path"]),
                "--features",
                str(paths["features_path"]),
                "--responses",
                str(paths["responses_path"]),
                "--splits",
                str(paths["splits_path"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    records = read_jsonl(output)

    assert len(records) == 2
    assert records[0]["dataset"] == "THINGS-EEG2"
    assert records[0]["brain_response"]["modality"] == "EEG"


def _write_stub_tables(tmp_path: Path) -> dict[str, Path]:
    dataset_access_path = tmp_path / "dataset_access.tsv"
    stimuli_path = tmp_path / "stimuli.csv"
    features_path = tmp_path / "features.jsonl"
    responses_path = tmp_path / "responses.csv"
    splits_path = tmp_path / "splits.csv"

    _write_csv(
        dataset_access_path,
        [
            {
                "dataset": "THINGS-EEG2",
                "dataset_version": "figshare-18470912",
                "source_url": "https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912",
                "access_date": "2026-06-03",
                "license_or_terms": "stubbed access note; verify before download",
                "allowed_use": "research-only metadata stub",
                "redistribution": "metadata-only",
                "source_doi": "https://doi.org/10.1016/j.neuroimage.2022.119754",
                "citation_key": "Gifford2022THINGSEEG2",
                "downloaded_by": "codex-test",
                "download_method": "local-metadata-stub",
                "raw_checksum_manifest": "restricted/checksums/things_eeg2.tsv",
            }
        ],
        delimiter="\t",
    )
    _write_csv(
        stimuli_path,
        [
            _stimulus_row("things-eeg2-img-000001", "accordion"),
            _stimulus_row("things-eeg2-img-000002", "airplane"),
        ],
    )
    _write_jsonl(
        features_path,
        [
            _feature_row("things-eeg2-img-000001"),
            _feature_row("things-eeg2-img-000002"),
        ],
    )
    _write_csv(
        responses_path,
        [
            _response_row("trial-000001", "things-eeg2-img-000001"),
            {
                **_response_row("trial-000002", "things-eeg2-img-000002"),
                "include": "false",
                "exclusion_reason": "metadata stub exclusion",
                "quality_flags": '["blink", "high_impedance"]',
            },
        ],
    )
    _write_csv(
        splits_path,
        [
            {
                "stimulus_id": "things-eeg2-img-000001",
                "split_name": "train",
                "split_policy": "held-out-image",
                "split_version": "2026-06-03-v0",
            },
            {
                "stimulus_id": "things-eeg2-img-000002",
                "split_name": "test",
                "split_policy": "held-out-image",
                "split_version": "2026-06-03-v0",
            },
        ],
    )
    return {
        "dataset_access_path": dataset_access_path,
        "stimuli_path": stimuli_path,
        "features_path": features_path,
        "responses_path": responses_path,
        "splits_path": splits_path,
    }


def _build_manifest_from_paths(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return build_raw_eeg_manifest(
        dataset_access_path=paths["dataset_access_path"],
        stimuli_path=paths["stimuli_path"],
        features_path=paths["features_path"],
        responses_path=paths["responses_path"],
        splits_path=paths["splits_path"],
    )


def _stimulus_row(stimulus_id: str, concept_label: str) -> dict[str, str]:
    return {
        "stimulus_id": stimulus_id,
        "source_image_id": stimulus_id.replace("things-eeg2-", "things-"),
        "source_corpus": "THINGS",
        "stimulus_type": "image",
        "image_path": f"restricted/stimuli/{stimulus_id}.jpg",
        "image_checksum": "sha256-stub",
        "concept_id": f"concept-{concept_label}",
        "concept_label": concept_label,
        "presentation_duration_ms": "100",
    }


def _feature_row(stimulus_id: str) -> dict[str, str]:
    return {
        "stimulus_id": stimulus_id,
        "feature_version": "2026-06-03-models-v0",
        "visual_embedding_path": f"features/visual/{stimulus_id}.npy",
        "semantic_embedding_path": f"features/semantic/{stimulus_id}.npy",
        "affect_feature_path": f"features/affect/{stimulus_id}.npy",
        "weak_social_feature_path": f"features/weak_social/{stimulus_id}.npy",
        "visual_model": "clip-vit-stub",
        "semantic_model": "text-embedding-stub",
        "affect_model": "affect-control-stub",
        "weak_social_model": "weak-social-scene-control-stub",
    }


def _response_row(trial_id: str, stimulus_id: str) -> dict[str, str]:
    return {
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "run_id": "run-01",
        "trial_id": trial_id,
        "stimulus_id": stimulus_id,
        "feature_version": "2026-06-03-models-v0",
        "repetition_index": "1",
        "onset_ms": "12345",
        "response_path": f"restricted/responses/sub-01/ses-01/run-01/{trial_id}_epoch.npy",
        "response_checksum": "sha256-response-stub",
        "modality": "EEG",
        "response_level": "epoch",
        "preprocessing_state": "preprocessed-stub",
        "channel_or_roi_space": "64ch EEG stub",
        "include": "true",
        "exclusion_reason": "",
        "quality_flags": "",
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "build_raw_eeg_manifest_stub.py"
    spec = importlib.util.spec_from_file_location(
        "build_raw_eeg_manifest_stub",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
