"""Recover pair manifests from activation extraction payloads.

Activation NPZ files intentionally keep prompts, labels, target scores, and
pair ids alongside activations. This module reconstructs PairwiseExample JSONL
manifests from those payloads when the original generated artifacts are no
longer available, while marking the provenance as inferred.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    FUTURE_OPTION_ORDER,
    future_options_for_contrast,
)
from social_cohesion_vectors.experiments.fault_taxonomy import (
    annotation_for_contrast,
    annotation_metadata_for_pair,
    base_contrast_id,
)
from social_cohesion_vectors.experiments.procedural_justice_control import (
    CONTROL_CONTRACT_VERSION,
    DEFAULT_CONTROL_CASES,
    DEFAULT_CONTROL_SOURCES,
)
from social_cohesion_vectors.schemas import PairwiseExample

DatasetKind = Literal["generated_fault", "procedural_control"]

DEFAULT_SOURCE_FAMILY_MAP: dict[str, str] = {
    "generated_neighborhood_forum_modal_hf": "generated_fault_class_primary",
    "generated_neighborhood_forum_constrained_repair": (
        "generated_fault_class_source_diverse"
    ),
    "generated_neighborhood_forum_constrained_repair_adv": (
        "generated_fault_class_lexical_adversarial"
    ),
    "generated_neighborhood_forum_constrained_repair_cross": (
        "generated_fault_class_cross_fault"
    ),
    "generated_neighborhood_forum_modal_hf_tournament": (
        "generated_fault_class_repair_v2_tournament"
    ),
}


@dataclass(frozen=True)
class ActivationPairManifestRecoveryResult:
    """Recovered pair examples plus a compact provenance report."""

    pairs: list[PairwiseExample]
    report: dict[str, Any]


@dataclass(frozen=True)
class _ActivationPromptRow:
    pair_id: str
    label: str
    sample_id: str
    text: str
    target_score: float


def recover_pair_manifest_from_activation_npz(
    activation_npz: str | Path,
    *,
    dataset_kind: DatasetKind,
    source_family_map: Mapping[str, str] | None = None,
    control_contract_version: str = CONTROL_CONTRACT_VERSION,
) -> ActivationPairManifestRecoveryResult:
    """Recover PairwiseExample records from an activation extraction NPZ."""

    activation_path = Path(activation_npz)
    rows, payload_summary = _load_activation_prompt_rows(activation_path)
    grouped = _group_prompt_rows(rows)
    mapping = {**DEFAULT_SOURCE_FAMILY_MAP, **dict(source_family_map or {})}

    pairs: list[PairwiseExample] = []
    for pair_id in sorted(grouped):
        pair_rows = grouped[pair_id]
        positive = _single_label_row(pair_rows, pair_id=pair_id, label="positive")
        negative = _single_label_row(pair_rows, pair_id=pair_id, label="negative")
        if dataset_kind == "generated_fault":
            metadata = _generated_fault_metadata(
                pair_id=pair_id,
                positive=positive,
                negative=negative,
                source_family_map=mapping,
            )
            scenario_id = metadata["scenario_id"]
        elif dataset_kind == "procedural_control":
            metadata = _procedural_control_metadata(
                pair_id=pair_id,
                positive=positive,
                negative=negative,
                control_contract_version=control_contract_version,
            )
            scenario_id = f"procedural_justice_control::{pair_id.split('::', 1)[0]}"
        else:
            msg = f"Unsupported dataset_kind: {dataset_kind!r}"
            raise ValueError(msg)

        pair_metadata = dict(metadata)
        pair_metadata.pop("scenario_id", None)
        pairs.append(
            PairwiseExample(
                pair_id=pair_id,
                scenario_id=str(scenario_id),
                positive_run_id=positive.sample_id,
                negative_run_id=negative.sample_id,
                positive_text=positive.text,
                negative_text=negative.text,
                positive_score=positive.target_score,
                negative_score=negative.target_score,
                metadata=pair_metadata,
            )
        )

    report = _recovery_report(
        activation_path=activation_path,
        dataset_kind=dataset_kind,
        pairs=pairs,
        payload_summary=payload_summary,
    )
    return ActivationPairManifestRecoveryResult(pairs=pairs, report=report)


def write_recovered_pair_manifest(
    activation_npz: str | Path,
    *,
    dataset_kind: DatasetKind,
    pairs_output: str | Path,
    json_report_output: str | Path | None = None,
    source_family_map: Mapping[str, str] | None = None,
    control_contract_version: str = CONTROL_CONTRACT_VERSION,
) -> ActivationPairManifestRecoveryResult:
    """Recover and write a PairwiseExample JSONL manifest."""

    result = recover_pair_manifest_from_activation_npz(
        activation_npz,
        dataset_kind=dataset_kind,
        source_family_map=source_family_map,
        control_contract_version=control_contract_version,
    )
    count = write_jsonl(result.pairs, pairs_output)
    report = {
        **result.report,
        "artifacts": {
            "pairs_output": str(pairs_output),
            "json_report_output": (
                str(json_report_output) if json_report_output is not None else None
            ),
        },
        "summary": {
            **_mapping(result.report.get("summary")),
            "written_pairwise_examples": count,
        },
    }
    if json_report_output is not None:
        _write_json(report, json_report_output)
    return ActivationPairManifestRecoveryResult(pairs=result.pairs, report=report)


def _load_activation_prompt_rows(path: Path) -> tuple[list[_ActivationPromptRow], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        pair_ids = _npz_array(data, "pair_ids", path=path)
        labels = _npz_array(data, "labels", path=path)
        sample_ids = _npz_array(data, "sample_ids", path=path)
        texts = _npz_array(data, "texts", path=path)
        target_scores = _npz_array(data, "target_scores", path=path)
        model_id = _optional_scalar(data, "model_id")
        layer = _optional_scalar(data, "layer")
        pooling = _optional_scalar(data, "pooling")

    lengths = {
        len(pair_ids),
        len(labels),
        len(sample_ids),
        len(texts),
        len(target_scores),
    }
    if len(lengths) != 1:
        msg = (
            f"Activation prompt arrays in {path} have mismatched lengths: "
            f"pair_ids={len(pair_ids)} labels={len(labels)} "
            f"sample_ids={len(sample_ids)} texts={len(texts)} "
            f"target_scores={len(target_scores)}"
        )
        raise ValueError(msg)

    rows = [
        _ActivationPromptRow(
            pair_id=str(pair_id),
            label=str(label),
            sample_id=str(sample_id),
            text=str(text),
            target_score=float(score),
        )
        for pair_id, label, sample_id, text, score in zip(
            pair_ids,
            labels,
            sample_ids,
            texts,
            target_scores,
            strict=True,
        )
    ]
    payload_summary = {
        "activation_npz": str(path),
        "prompt_rows": len(rows),
        "model_id": model_id,
        "layer": layer,
        "pooling": pooling,
    }
    return rows, payload_summary


def _npz_array(data: Mapping[str, np.ndarray], key: str, *, path: Path) -> np.ndarray:
    if key not in data:
        msg = f"Activation payload {path} is missing required array {key!r}."
        raise ValueError(msg)
    return np.asarray(data[key])


def _optional_scalar(data: Mapping[str, np.ndarray], key: str) -> str | float | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape != ():
        return str(value.tolist())
    item = value.item()
    if isinstance(item, int | float):
        return float(item)
    return str(item)


def _group_prompt_rows(
    rows: list[_ActivationPromptRow],
) -> dict[str, list[_ActivationPromptRow]]:
    grouped: dict[str, list[_ActivationPromptRow]] = defaultdict(list)
    for row in rows:
        grouped[row.pair_id].append(row)
    return dict(grouped)


def _single_label_row(
    rows: list[_ActivationPromptRow],
    *,
    pair_id: str,
    label: str,
) -> _ActivationPromptRow:
    matches = [row for row in rows if row.label == label]
    if len(matches) != 1:
        msg = (
            f"Pair {pair_id!r} should have exactly one {label!r} prompt row; "
            f"found {len(matches)}."
        )
        raise ValueError(msg)
    return matches[0]


def _generated_fault_metadata(
    *,
    pair_id: str,
    positive: _ActivationPromptRow,
    negative: _ActivationPromptRow,
    source_family_map: Mapping[str, str],
) -> dict[str, str | float]:
    contrast_id = _generated_contrast_id(pair_id)
    base_id = base_contrast_id(contrast_id)
    annotation = annotation_for_contrast(contrast_id)
    fault_classes = annotation.fault_classes if annotation is not None else ()
    raw_source = _generated_source_suffix(contrast_id)
    source = source_family_map.get(raw_source, raw_source or "generated_fault_recovered")
    options = future_options_for_contrast(contrast_id)
    metadata: dict[str, str | float] = {
        "artifact_class": "generated_fault_recovered_from_activation",
        "metadata_recovery": "inferred_from_activation_payload",
        "metadata_recovery_warning": (
            "Original generated raw output manifest was unavailable; text, labels, "
            "and scores come from the activation NPZ, while taxonomy/source fields "
            "are inferred from stable pair ids."
        ),
        "source": source,
        "provider": "recovered_activation_payload",
        "raw_source_family": raw_source,
        "base_contrast_id": base_id,
        "scenario_id": contrast_id,
        "generated_variant": raw_source,
        "generated_style": "recovered_activation_payload",
        "primary_fault_class": fault_classes[0] if fault_classes else "unknown",
        "positive_label": "genuine_cohesion",
        "negative_label": "pseudo_cohesion",
        "score_margin": round(positive.target_score - negative.target_score, 6),
        "slack_options_tested": ",".join(options),
        "positive_slack_preservation": positive.target_score,
        "negative_slack_preservation": negative.target_score,
        "slack_preservation_margin": round(
            positive.target_score - negative.target_score,
            6,
        ),
        "claim_boundary": "activation_manifest_recovery_not_original_generation_log",
    }
    metadata.update(annotation_metadata_for_pair(contrast_id))
    return metadata


def _generated_contrast_id(pair_id: str) -> str:
    return pair_id.strip().removeprefix("generated-fault::")


def _generated_source_suffix(contrast_id: str) -> str:
    if "__" not in contrast_id:
        return ""
    return contrast_id.split("__", 1)[1]


def _procedural_control_metadata(
    *,
    pair_id: str,
    positive: _ActivationPromptRow,
    negative: _ActivationPromptRow,
    control_contract_version: str,
) -> dict[str, str | float]:
    case_id, source_id = _procedural_pair_parts(pair_id)
    case_by_id = {case.case_id: case for case in DEFAULT_CONTROL_CASES}
    source_by_id = {source.source_id: source for source in DEFAULT_CONTROL_SOURCES}
    if case_id not in case_by_id:
        msg = f"Unknown procedural control case id in pair {pair_id!r}: {case_id!r}."
        raise ValueError(msg)
    if source_id not in source_by_id:
        msg = f"Unknown procedural control source id in pair {pair_id!r}: {source_id!r}."
        raise ValueError(msg)
    case = case_by_id[case_id]
    source = source_by_id[source_id]
    return {
        "artifact_class": "non_generated_control",
        "metadata_recovery": "inferred_from_activation_payload",
        "metadata_recovery_warning": (
            "Original pair manifest was unavailable; text, labels, and scores "
            "come from the activation NPZ, while procedural-control metadata is "
            "inferred from stable case/source ids."
        ),
        "control_contract_version": control_contract_version,
        "primary_fault_class": case.primary_fault_class,
        "fault_classes": case.primary_fault_class,
        "case_title": case.title,
        "source": source.source_id,
        "source_label": source.source_label,
        "provider": "hand_authored",
        "generated_style": "none",
        "slack_options_tested": ",".join(case.options),
        "positive_slack_preservation": 0.88,
        "negative_slack_preservation": 0.26,
        "slack_preservation_margin": 0.62,
        "score_margin": round(positive.target_score - negative.target_score, 6),
        "future_option_universe": ",".join(FUTURE_OPTION_ORDER),
        "claim_boundary": "generated_text_excluded_non_human_control",
    }


def _procedural_pair_parts(pair_id: str) -> tuple[str, str]:
    if "::" not in pair_id:
        msg = f"Procedural control pair id must contain '::': {pair_id!r}."
        raise ValueError(msg)
    case_id, source_id = pair_id.split("::", 1)
    return case_id, source_id


def _recovery_report(
    *,
    activation_path: Path,
    dataset_kind: DatasetKind,
    pairs: list[PairwiseExample],
    payload_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    fault_counts = Counter(
        str(pair.metadata.get("primary_fault_class", "")) for pair in pairs
    )
    recovered_pairs = sum(
        1
        for pair in pairs
        if pair.metadata.get("metadata_recovery") == "inferred_from_activation_payload"
    )
    return {
        "experiment": "activation_pair_manifest_recovery",
        "description": (
            "Recovers PairwiseExample manifests from activation extraction payloads "
            "when the original JSONL artifacts are unavailable."
        ),
        "inputs": {
            **dict(payload_summary),
            "dataset_kind": dataset_kind,
        },
        "summary": {
            "status": "recovered_manifest_ready" if pairs else "empty_manifest",
            "pairwise_examples": len(pairs),
            "prompt_rows": int(payload_summary.get("prompt_rows", 0)),
            "recovered_pairwise_examples": recovered_pairs,
            "sources": len(source_counts),
            "source_counts": dict(sorted(source_counts.items())),
            "primary_fault_class_counts": dict(sorted(fault_counts.items())),
            "activation_npz": str(activation_path),
            "claim_boundary": "metadata_inferred_from_activation_payload",
        },
    }


def _mapping(raw_value: object) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
