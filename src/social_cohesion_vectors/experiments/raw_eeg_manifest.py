"""Tiny metadata-only manifest builder for raw EEG bridge datasets.

The first supported target is THINGS-EEG2. This module reads local CSV, TSV, or
JSONL metadata stubs and writes auditable JSONL manifest rows. It never reads
raw EEG arrays, image bytes, feature arrays, or subject-level data.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl

MANIFEST_VERSION = "raw-eeg-bridge-manifest-v0"
DEFAULT_DATASET = "THINGS-EEG2"
CLAIM_BOUNDARY = (
    "Representation-learning manifest only; no human behavioral, intervention, "
    "social-cohesion, empathy, bonding, cooperation, or neural-synchrony claim."
)

ACCESS_REQUIRED = (
    "dataset",
    "dataset_version",
    "source_url",
    "access_date",
    "license_or_terms",
    "allowed_use",
    "redistribution",
)
PROVENANCE_REQUIRED = (
    "source_doi",
    "citation_key",
    "downloaded_by",
    "download_method",
    "raw_checksum_manifest",
)
STIMULUS_REQUIRED = (
    "stimulus_id",
    "source_image_id",
    "source_corpus",
    "stimulus_type",
    "image_path",
)
FEATURE_REQUIRED = (
    "stimulus_id",
    "feature_version",
    "visual_embedding_path",
    "semantic_embedding_path",
    "affect_feature_path",
    "weak_social_feature_path",
)
RESPONSE_REQUIRED = (
    "subject_id",
    "session_id",
    "run_id",
    "trial_id",
    "stimulus_id",
    "response_path",
    "modality",
    "response_level",
    "preprocessing_state",
    "channel_or_roi_space",
)
SPLIT_REQUIRED = ("split_name", "split_policy", "split_version")
PATH_FIELDS = (
    ("stimulus", "image_path"),
    ("features", "visual_embedding_path"),
    ("features", "semantic_embedding_path"),
    ("features", "affect_feature_path"),
    ("features", "weak_social_feature_path"),
    ("brain_response", "response_path"),
)


@dataclass(frozen=True)
class ManifestValidation:
    """Validation result for a metadata-only raw EEG manifest."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def build_raw_eeg_manifest(
    *,
    dataset_access_path: str | Path,
    stimuli_path: str | Path,
    features_path: str | Path,
    responses_path: str | Path,
    splits_path: str | Path,
    dataset: str = DEFAULT_DATASET,
    manifest_version: str = MANIFEST_VERSION,
) -> list[dict[str, Any]]:
    """Build nested JSON-ready manifest rows from local metadata stubs."""

    access = _dataset_access_record(dataset_access_path, dataset=dataset)
    stimuli = _index_unique(
        _load_metadata_table(stimuli_path),
        key_fields=("stimulus_id",),
        table_name="stimuli",
    )
    features = _index_unique(
        _load_metadata_table(features_path),
        key_fields=("stimulus_id", "feature_version"),
        table_name="features",
    )
    responses = _load_metadata_table(responses_path)
    splits = _load_metadata_table(splits_path)
    split_index = _index_splits(splits)

    _require_columns([access], ACCESS_REQUIRED + PROVENANCE_REQUIRED, "dataset_access")
    _require_columns(stimuli.values(), STIMULUS_REQUIRED, "stimuli")
    _require_columns(features.values(), FEATURE_REQUIRED, "features")
    _require_columns(responses, RESPONSE_REQUIRED, "responses")
    _require_columns(splits, SPLIT_REQUIRED, "splits")

    if access["dataset"] != dataset:
        msg = f"dataset_access dataset must be {dataset!r}, got {access['dataset']!r}"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    for response in responses:
        stimulus_id = _required_value(response, "stimulus_id", "responses")
        stimulus = stimuli.get((stimulus_id,))
        if stimulus is None:
            raise ValueError(f"responses row references missing stimulus_id {stimulus_id!r}")

        feature_version = response.get("feature_version") or _latest_feature_version(
            features,
            stimulus_id,
        )
        feature = features.get((stimulus_id, feature_version))
        if feature is None:
            msg = (
                "responses row references missing feature row "
                f"stimulus_id={stimulus_id!r} feature_version={feature_version!r}"
            )
            raise ValueError(msg)

        split = _matching_split(split_index, response)
        if split is None:
            raise ValueError(f"no split row matches stimulus_id {stimulus_id!r}")

        records.append(
            _manifest_record(
                access=access,
                stimulus=stimulus,
                feature=feature,
                response=response,
                split=split,
                manifest_version=manifest_version,
            )
        )

    validation = validate_raw_eeg_manifest(records, expected_dataset=dataset)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return records


def validate_raw_eeg_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_dataset: str = DEFAULT_DATASET,
) -> ManifestValidation:
    """Validate required fields and metadata-only safety boundaries."""

    errors: list[str] = []
    warnings: list[str] = []
    seen_trial_keys: set[tuple[str, ...]] = set()
    materialized = list(records)

    if not materialized:
        errors.append("manifest has no records")

    for index, record in enumerate(materialized, start=1):
        prefix = f"record {index}"
        errors.extend(_validate_record_header(record, prefix, expected_dataset))
        errors.extend(_validate_record_sections(record, prefix))
        errors.extend(_validate_stimulus_alignment(record, prefix))
        errors.extend(_validate_unique_trial_key(record, prefix, seen_trial_keys))
        path_errors, path_warnings = _validate_relative_paths(record, prefix)
        errors.extend(path_errors)
        warnings.extend(path_warnings)

    return ManifestValidation(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _validate_record_header(
    record: Mapping[str, Any],
    prefix: str,
    expected_dataset: str,
) -> list[str]:
    errors: list[str] = []
    if record.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"{prefix}: unsupported manifest_version")
    if record.get("dataset") != expected_dataset:
        errors.append(f"{prefix}: dataset must be {expected_dataset}")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{prefix}: claim_boundary is missing or changed")
    return errors


def _validate_record_sections(record: Mapping[str, Any], prefix: str) -> list[str]:
    errors: list[str] = []
    for section, fields in _validation_sections().items():
        value = record.get(section)
        if not isinstance(value, Mapping):
            errors.append(f"{prefix}: missing section {section}")
            continue
        errors.extend(
            f"{prefix}: missing {section}.{field}"
            for field in fields
            if field not in value
        )
    return errors


def _validation_sections() -> dict[str, Sequence[str]]:
    return {
        "access": ACCESS_REQUIRED,
        "provenance": PROVENANCE_REQUIRED,
        "stimulus": STIMULUS_REQUIRED,
        "features": FEATURE_REQUIRED,
        "brain_response": RESPONSE_REQUIRED,
        "subject_session": (
            "subject_id",
            "session_id",
            "run_id",
            "trial_id",
            "repetition_index",
            "onset_ms",
        ),
        "split": SPLIT_REQUIRED,
        "quality": ("include", "exclusion_reason", "quality_flags"),
    }


def _validate_stimulus_alignment(record: Mapping[str, Any], prefix: str) -> list[str]:
    stimulus = record.get("stimulus")
    response = record.get("brain_response")
    if not isinstance(stimulus, Mapping) or not isinstance(response, Mapping):
        return []
    if response.get("stimulus_id") == stimulus.get("stimulus_id"):
        return []
    return [f"{prefix}: brain_response stimulus_id does not match stimulus"]


def _validate_unique_trial_key(
    record: Mapping[str, Any],
    prefix: str,
    seen_trial_keys: set[tuple[str, ...]],
) -> list[str]:
    response = record.get("brain_response")
    if not isinstance(record.get("subject_session"), Mapping) or not isinstance(
        response,
        Mapping,
    ):
        return []
    trial_key = tuple(
        str(response.get(field, ""))
        for field in (
            "subject_id",
            "session_id",
            "run_id",
            "trial_id",
            "stimulus_id",
        )
    )
    if trial_key in seen_trial_keys:
        return [f"{prefix}: duplicate response trial key {trial_key}"]
    seen_trial_keys.add(trial_key)
    return []


def _validate_relative_paths(
    record: Mapping[str, Any],
    prefix: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for section, field in PATH_FIELDS:
        section_value = record.get(section)
        if not isinstance(section_value, Mapping):
            continue
        path_value = section_value.get(field)
        if path_value in (None, "", "null"):
            warnings.append(f"{prefix}: empty {section}.{field}")
            continue
        if Path(str(path_value)).is_absolute():
            errors.append(f"{prefix}: {section}.{field} must be a relative path")
    return errors, warnings


def export_raw_eeg_manifest(
    *,
    dataset_access_path: str | Path,
    stimuli_path: str | Path,
    features_path: str | Path,
    responses_path: str | Path,
    splits_path: str | Path,
    output_path: str | Path,
    dataset: str = DEFAULT_DATASET,
) -> int:
    """Build, validate, and write a JSONL raw EEG metadata manifest."""

    records = build_raw_eeg_manifest(
        dataset_access_path=dataset_access_path,
        stimuli_path=stimuli_path,
        features_path=features_path,
        responses_path=responses_path,
        splits_path=splits_path,
        dataset=dataset,
    )
    return write_jsonl(records, output_path)


def _manifest_record(
    *,
    access: Mapping[str, str],
    stimulus: Mapping[str, str],
    feature: Mapping[str, str],
    response: Mapping[str, str],
    split: Mapping[str, str],
    manifest_version: str,
) -> dict[str, Any]:
    stimulus_id = _required_value(stimulus, "stimulus_id", "stimuli")
    feature_version = _required_value(feature, "feature_version", "features")

    return {
        "manifest_version": manifest_version,
        "dataset": access["dataset"],
        "dataset_version": access["dataset_version"],
        "claim_boundary": CLAIM_BOUNDARY,
        "access": {
            field: _string_or_none(access.get(field)) for field in ACCESS_REQUIRED
        },
        "provenance": {
            field: _string_or_none(access.get(field)) for field in PROVENANCE_REQUIRED
        },
        "stimulus": {
            "stimulus_id": stimulus_id,
            "source_image_id": _string_or_none(stimulus.get("source_image_id")),
            "source_corpus": _string_or_none(stimulus.get("source_corpus")),
            "stimulus_type": _string_or_none(stimulus.get("stimulus_type")),
            "image_path": _string_or_none(stimulus.get("image_path")),
            "image_checksum": _string_or_none(stimulus.get("image_checksum")),
            "concept_id": _string_or_none(stimulus.get("concept_id")),
            "concept_label": _string_or_none(stimulus.get("concept_label")),
            "presentation_duration_ms": _int_or_none(
                stimulus.get("presentation_duration_ms")
            ),
        },
        "features": {
            "feature_row_id": f"{access['dataset']}:{stimulus_id}:{feature_version}",
            "stimulus_id": stimulus_id,
            "feature_version": feature_version,
            "visual_embedding_path": _string_or_none(
                feature.get("visual_embedding_path")
            ),
            "semantic_embedding_path": _string_or_none(
                feature.get("semantic_embedding_path")
            ),
            "affect_feature_path": _string_or_none(feature.get("affect_feature_path")),
            "weak_social_feature_path": _string_or_none(
                feature.get("weak_social_feature_path")
            ),
            "feature_models": {
                "visual": _string_or_none(feature.get("visual_model")),
                "semantic": _string_or_none(feature.get("semantic_model")),
                "affect": _string_or_none(feature.get("affect_model")),
                "weak_social": _string_or_none(feature.get("weak_social_model")),
            },
        },
        "brain_response": {
            "stimulus_id": stimulus_id,
            "subject_id": _string_or_none(response.get("subject_id")),
            "session_id": _string_or_none(response.get("session_id")),
            "run_id": _string_or_none(response.get("run_id")),
            "trial_id": _string_or_none(response.get("trial_id")),
            "modality": _string_or_none(response.get("modality")),
            "response_level": _string_or_none(response.get("response_level")),
            "response_path": _string_or_none(response.get("response_path")),
            "response_checksum": _string_or_none(response.get("response_checksum")),
            "preprocessing_state": _string_or_none(response.get("preprocessing_state")),
            "channel_or_roi_space": _string_or_none(
                response.get("channel_or_roi_space")
            ),
        },
        "subject_session": {
            "subject_id": _string_or_none(response.get("subject_id")),
            "session_id": _string_or_none(response.get("session_id")),
            "run_id": _string_or_none(response.get("run_id")),
            "trial_id": _string_or_none(response.get("trial_id")),
            "repetition_index": _int_or_none(response.get("repetition_index")),
            "onset_ms": _int_or_none(response.get("onset_ms")),
        },
        "split": {
            "split_name": _string_or_none(split.get("split_name")),
            "split_policy": _string_or_none(split.get("split_policy")),
            "split_version": _string_or_none(split.get("split_version")),
        },
        "quality": {
            "include": _bool_value(response.get("include"), default=True),
            "exclusion_reason": _string_or_none(response.get("exclusion_reason")),
            "quality_flags": _list_value(response.get("quality_flags")),
        },
    }


def _load_metadata_table(path: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(path)
    suffix = metadata_path.suffix.lower()
    if suffix == ".jsonl":
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in read_jsonl(metadata_path)
        ]

    delimiter = "\t" if suffix == ".tsv" else ","
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{metadata_path} has no header row")
        return [
            {
                str(key): "" if value is None else value
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]


def _dataset_access_record(path: str | Path, *, dataset: str) -> dict[str, str]:
    records = [row for row in _load_metadata_table(path) if row.get("dataset") == dataset]
    if len(records) != 1:
        raise ValueError(
            f"dataset_access must contain exactly one {dataset!r} row, got {len(records)}"
        )
    return records[0]


def _require_columns(
    rows: Iterable[Mapping[str, str]],
    fields: Sequence[str],
    table_name: str,
) -> None:
    for row_index, row in enumerate(rows, start=1):
        missing = [field for field in fields if not row.get(field)]
        if missing:
            msg = f"{table_name} row {row_index} missing required fields: {missing}"
            raise ValueError(msg)


def _index_unique(
    rows: Iterable[Mapping[str, str]],
    *,
    key_fields: Sequence[str],
    table_name: str,
) -> dict[tuple[str, ...], Mapping[str, str]]:
    indexed: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(_required_value(row, field, table_name) for field in key_fields)
        if key in indexed:
            raise ValueError(f"{table_name} has duplicate key {key}")
        indexed[key] = row
    return indexed


def _index_splits(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, ...], Mapping[str, str]]:
    indexed: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = _split_key(row)
        if key in indexed:
            raise ValueError(f"splits has duplicate key {key}")
        indexed[key] = row
    return indexed


def _matching_split(
    split_index: Mapping[tuple[str, ...], Mapping[str, str]],
    response: Mapping[str, str],
) -> Mapping[str, str] | None:
    trial_key = _trial_split_key(response)
    stimulus_key = ("stimulus", _required_value(response, "stimulus_id", "responses"))
    return split_index.get(trial_key) or split_index.get(stimulus_key)


def _split_key(row: Mapping[str, str]) -> tuple[str, ...]:
    trial_fields = ("subject_id", "session_id", "run_id", "trial_id", "stimulus_id")
    if all(row.get(field) for field in trial_fields):
        return _trial_split_key(row)
    stimulus_id = _required_value(row, "stimulus_id", "splits")
    return ("stimulus", stimulus_id)


def _trial_split_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return (
        "trial",
        _required_value(row, "subject_id", "splits"),
        _required_value(row, "session_id", "splits"),
        _required_value(row, "run_id", "splits"),
        _required_value(row, "trial_id", "splits"),
        _required_value(row, "stimulus_id", "splits"),
    )


def _latest_feature_version(
    features: Mapping[tuple[str, ...], Mapping[str, str]],
    stimulus_id: str,
) -> str:
    versions = sorted(key[1] for key in features if key[0] == stimulus_id)
    if not versions:
        raise ValueError(f"no feature rows found for stimulus_id {stimulus_id!r}")
    return versions[-1]


def _required_value(row: Mapping[str, str], field: str, table_name: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"{table_name} row missing required field {field!r}")
    return str(value)


def _string_or_none(value: object) -> str | None:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _int_or_none(value: object) -> int | None:
    if value in (None, "", "null", "None"):
        return None
    return int(str(value))


def _bool_value(value: object, *, default: bool) -> bool:
    if value in (None, "", "null", "None"):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _list_value(value: object) -> list[str]:
    if value in (None, "", "null", "None"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"quality_flags JSON value must be a list: {value!r}")
        return [str(item) for item in parsed]
    return [item.strip() for item in text.split("|") if item.strip()]
