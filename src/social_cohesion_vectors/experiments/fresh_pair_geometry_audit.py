"""Pair-level geometry audit for hard fresh generated residuals."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.transfer import load_activation_payload
from social_cohesion_vectors.schemas import PairwiseExample

_GENERATED_PREFIX = "generated-fault::"


@dataclass(frozen=True)
class _Dataset:
    name: str
    activations: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    pairs: tuple[PairwiseExample, ...]


@dataclass(frozen=True)
class _PairVectors:
    pair_id: str
    positive: np.ndarray
    negative: np.ndarray
    delta: np.ndarray


@dataclass(frozen=True)
class _PromptVector:
    sample_id: str
    pair_id: str
    label: str
    base_contrast_id: str
    source: str
    vector: np.ndarray


def run_fresh_pair_geometry_audit_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    fresh_source_activation_npz: str | Path,
    fresh_source_pairs_path: str | Path,
    focus_base_contrast_id: str,
    focus_pair_id: str | None = None,
    model_name: str = "model",
    nearest_k: int = 8,
) -> dict[str, Any]:
    """Load activation payloads and run a pair-level geometry audit."""

    source = _load_dataset(
        name="source",
        activation_npz=source_activation_npz,
        pairs_path=source_pairs_path,
    )
    target = _load_dataset(
        name="target",
        activation_npz=target_activation_npz,
        pairs_path=target_pairs_path,
    )
    fresh_source = _load_dataset(
        name="fresh_source",
        activation_npz=fresh_source_activation_npz,
        pairs_path=fresh_source_pairs_path,
    )
    return run_fresh_pair_geometry_audit(
        source=source,
        target=target,
        fresh_source=fresh_source,
        focus_base_contrast_id=focus_base_contrast_id,
        focus_pair_id=focus_pair_id,
        model_name=model_name,
        nearest_k=nearest_k,
        input_paths={
            "source_activation_npz": str(source_activation_npz),
            "source_pairs": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs": str(target_pairs_path),
            "fresh_source_activation_npz": str(fresh_source_activation_npz),
            "fresh_source_pairs": str(fresh_source_pairs_path),
        },
    )


def run_fresh_pair_geometry_audit(
    *,
    source: _Dataset,
    target: _Dataset,
    fresh_source: _Dataset,
    focus_base_contrast_id: str,
    focus_pair_id: str | None = None,
    model_name: str = "model",
    nearest_k: int = 8,
    input_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare a hard fresh pair against same-base source geometry."""

    _validate_shared_dim(source, target, fresh_source)
    source_vectors = _pair_vectors(source)
    fresh_vectors = _pair_vectors(fresh_source)
    focus_pair_id = focus_pair_id or _resolve_focus_pair_id(
        fresh_source.pairs,
        focus_base_contrast_id=focus_base_contrast_id,
    )
    if focus_pair_id not in fresh_vectors:
        msg = f"Focus pair {focus_pair_id!r} not found in fresh source activations."
        raise ValueError(msg)
    focus = fresh_vectors[focus_pair_id]
    source_same_base_ids = [
        pair.pair_id
        for pair in source.pairs
        if _base_id(pair) == focus_base_contrast_id and pair.pair_id in source_vectors
    ]
    if not source_same_base_ids:
        msg = f"No source same-base pairs found for {focus_base_contrast_id!r}."
        raise ValueError(msg)

    directions = _directions(
        source=source,
        target=target,
        fresh_source=fresh_source,
        focus_pair_id=focus_pair_id,
    )
    focus_projection_rows = [
        _pair_projection_row(
            pair=focus,
            direction_id=direction_id,
            direction=direction,
        )
        for direction_id, direction in directions.items()
    ]
    source_same_base_rows = [
        _source_same_base_row(
            pair=source_vectors[pair_id],
            focus=focus,
            directions=directions,
        )
        for pair_id in source_same_base_ids
    ]
    source_delta_neighbors = _delta_neighbors(
        focus=focus,
        candidate_vectors=source_vectors,
        candidate_pairs=source.pairs,
        nearest_k=nearest_k,
    )
    prompt_neighbors = _prompt_neighbor_sections(
        focus=focus,
        source=source,
        fresh_source=fresh_source,
        nearest_k=nearest_k,
    )
    readiness = _readiness(
        focus_projection_rows=focus_projection_rows,
        source_same_base_rows=source_same_base_rows,
    )
    return {
        "experiment": "fresh_pair_geometry_audit",
        "description": (
            "Audits whether a hard fresh generated pair is aligned with "
            "same-base source-pair deltas, direction projections, and nearest "
            "prompt neighborhoods."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "model_name": model_name,
            "focus_base_contrast_id": focus_base_contrast_id,
            "focus_pair_id": focus_pair_id,
            "nearest_k": nearest_k,
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "fresh_source_pairs": len(_unique_pair_ids(fresh_source)),
        },
        "summary": _summary(
            focus_projection_rows=focus_projection_rows,
            source_same_base_rows=source_same_base_rows,
            readiness=readiness,
        ),
        "readiness": readiness,
        "focus_projection_rows": focus_projection_rows,
        "source_same_base_rows": source_same_base_rows,
        "source_delta_neighbors": source_delta_neighbors,
        "prompt_neighbors": prompt_neighbors,
        "interpretation_guardrail": (
            "Fresh pair geometry audits are text-benchmark activation "
            "diagnostics. They do not establish causal steering, human "
            "behavioral, neural, clinical, deployment, or real-world "
            "social-effect claims."
        ),
    }


def save_fresh_pair_geometry_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown pair-geometry audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fresh_pair_geometry_audit_markdown(report),
        encoding="utf-8",
    )


def render_fresh_pair_geometry_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render pair-geometry audit results as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Fresh Pair Geometry Audit",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Model: `{inputs.get('model_name', '')}`",
            f"- Focus base: `{inputs.get('focus_base_contrast_id', '')}`",
            f"- Focus pair: `{inputs.get('focus_pair_id', '')}`",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'unknown')}`",
            f"- Original joint focus margin: "
            f"{float(summary.get('original_joint_focus_margin', 0.0)):+.3f}",
            f"- Full augmented focus margin: "
            f"{float(summary.get('full_augmented_focus_margin', 0.0)):+.3f}",
            f"- Fresh-only focus margin: "
            f"{float(summary.get('fresh_only_focus_margin', 0.0)):+.3f}",
            f"- Mean source same-base delta cosine: "
            f"{float(summary.get('mean_source_same_base_delta_cosine', 0.0)):+.3f}",
            f"- Min source same-base original margin: "
            f"{float(summary.get('source_same_base_original_min_margin', 0.0)):+.3f}",
            "",
            "## Focus Projections",
            "",
            *_focus_projection_table(report.get("focus_projection_rows")),
            "",
            "## Same-Base Source Deltas",
            "",
            *_same_base_table(report.get("source_same_base_rows")),
            "",
            "## Nearest Source Deltas",
            "",
            *_delta_neighbor_table(report.get("source_delta_neighbors")),
            "",
            "## Prompt Neighbors",
            "",
            *_prompt_neighbor_tables(report.get("prompt_neighbors")),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def _load_dataset(
    *,
    name: str,
    activation_npz: str | Path,
    pairs_path: str | Path,
) -> _Dataset:
    payload = load_activation_payload(activation_npz)
    pairs = tuple(load_pairwise_examples_jsonl(pairs_path))
    pair_ids = {pair.pair_id for pair in pairs}
    activation_pair_ids = {str(pair_id) for pair_id in payload.pair_ids}
    missing_metadata = sorted(activation_pair_ids - pair_ids)
    missing_activations = sorted(pair_ids - activation_pair_ids)
    if missing_metadata or missing_activations:
        msg = (
            f"Activation/pair metadata mismatch for {name}: "
            f"missing_metadata={missing_metadata[:5]} "
            f"missing_activations={missing_activations[:5]}"
        )
        raise ValueError(msg)
    return _Dataset(
        name=name,
        activations=np.asarray(payload.activations, dtype=np.float64),
        labels=np.asarray(payload.labels, dtype=str),
        pair_ids=np.asarray(payload.pair_ids, dtype=str),
        pairs=pairs,
    )


def _validate_shared_dim(*datasets: _Dataset) -> None:
    dims = {int(dataset.activations.shape[1]) for dataset in datasets}
    if len(dims) != 1:
        msg = f"Activation dimensions must match; got {sorted(dims)}."
        raise ValueError(msg)


def _pair_vectors(dataset: _Dataset) -> dict[str, _PairVectors]:
    vectors: dict[str, _PairVectors] = {}
    for pair_id in sorted(_unique_pair_ids(dataset)):
        positive = _label_vector(dataset, pair_id=pair_id, label="positive")
        negative = _label_vector(dataset, pair_id=pair_id, label="negative")
        vectors[pair_id] = _PairVectors(
            pair_id=pair_id,
            positive=positive,
            negative=negative,
            delta=positive - negative,
        )
    return vectors


def _label_vector(dataset: _Dataset, *, pair_id: str, label: str) -> np.ndarray:
    mask = (dataset.pair_ids == pair_id) & (dataset.labels == label)
    if not bool(np.any(mask)):
        msg = f"Dataset {dataset.name} missing {label!r} activation for {pair_id!r}."
        raise ValueError(msg)
    return np.asarray(dataset.activations[mask].mean(axis=0), dtype=np.float64)


def _directions(
    *,
    source: _Dataset,
    target: _Dataset,
    fresh_source: _Dataset,
    focus_pair_id: str,
) -> dict[str, np.ndarray]:
    source_pairs = _unique_pair_ids(source)
    target_pairs = _unique_pair_ids(target)
    fresh_pairs = _unique_pair_ids(fresh_source)
    return {
        "original_source_target_joint": _direction_from_datasets(
            (source, source_pairs),
            (target, target_pairs),
        ),
        "fresh_source_only": _direction_from_datasets((fresh_source, fresh_pairs)),
        "full_fresh_augmented": _direction_from_datasets(
            (source, source_pairs),
            (target, target_pairs),
            (fresh_source, fresh_pairs),
        ),
        "leave_focus_out_augmented": _direction_from_datasets(
            (source, source_pairs),
            (target, target_pairs),
            (fresh_source, fresh_pairs - {focus_pair_id}),
        ),
    }


def _direction_from_datasets(*parts: tuple[_Dataset, set[str]]) -> np.ndarray:
    activation_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for dataset, pair_ids in parts:
        mask = _mask(dataset, pair_ids)
        activation_parts.append(dataset.activations[mask])
        label_parts.append(dataset.labels[mask])
    return train_direction_from_arrays(
        np.concatenate(activation_parts, axis=0),
        labels=np.concatenate(label_parts, axis=0),
    ).direction


def _pair_projection_row(
    *,
    pair: _PairVectors,
    direction_id: str,
    direction: np.ndarray,
) -> dict[str, Any]:
    positive_projection = float(pair.positive @ direction)
    negative_projection = float(pair.negative @ direction)
    margin = positive_projection - negative_projection
    return {
        "direction_id": direction_id,
        "positive_projection": round(positive_projection, 6),
        "negative_projection": round(negative_projection, 6),
        "margin": round(margin, 6),
        "delta_cosine_to_direction": round(_cosine(pair.delta, direction), 6),
        "passed": margin > 0.0,
    }


def _source_same_base_row(
    *,
    pair: _PairVectors,
    focus: _PairVectors,
    directions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_id": pair.pair_id,
        "delta_cosine_to_focus": round(_cosine(pair.delta, focus.delta), 6),
    }
    for direction_id, direction in directions.items():
        row[f"{direction_id}_margin"] = round(float(pair.delta @ direction), 6)
    return row


def _delta_neighbors(
    *,
    focus: _PairVectors,
    candidate_vectors: Mapping[str, _PairVectors],
    candidate_pairs: Sequence[PairwiseExample],
    nearest_k: int,
) -> list[dict[str, Any]]:
    metadata = {pair.pair_id: pair for pair in candidate_pairs}
    rows = [
        {
            "pair_id": pair_id,
            "base_contrast_id": _base_id(metadata[pair_id]),
            "source": str(metadata[pair_id].metadata.get("source", "")),
            "delta_cosine_to_focus": round(_cosine(focus.delta, vectors.delta), 6),
        }
        for pair_id, vectors in candidate_vectors.items()
        if pair_id in metadata
    ]
    return sorted(rows, key=lambda row: float(row["delta_cosine_to_focus"]), reverse=True)[
        :nearest_k
    ]


def _prompt_neighbor_sections(
    *,
    focus: _PairVectors,
    source: _Dataset,
    fresh_source: _Dataset,
    nearest_k: int,
) -> dict[str, list[dict[str, Any]]]:
    source_prompts = _prompt_vectors(source)
    fresh_prompts = [
        prompt
        for prompt in _prompt_vectors(fresh_source)
        if prompt.pair_id != focus.pair_id
    ]
    return {
        "focus_positive_nearest_source": _nearest_prompts(
            focus.positive,
            source_prompts,
            nearest_k=nearest_k,
        ),
        "focus_negative_nearest_source": _nearest_prompts(
            focus.negative,
            source_prompts,
            nearest_k=nearest_k,
        ),
        "focus_positive_nearest_fresh": _nearest_prompts(
            focus.positive,
            fresh_prompts,
            nearest_k=nearest_k,
        ),
        "focus_negative_nearest_fresh": _nearest_prompts(
            focus.negative,
            fresh_prompts,
            nearest_k=nearest_k,
        ),
    }


def _prompt_vectors(dataset: _Dataset) -> list[_PromptVector]:
    pair_by_id = {pair.pair_id: pair for pair in dataset.pairs}
    rows: list[_PromptVector] = []
    for index, (pair_id, label, vector) in enumerate(
        zip(dataset.pair_ids, dataset.labels, dataset.activations, strict=True)
    ):
        pair = pair_by_id[str(pair_id)]
        rows.append(
            _PromptVector(
                sample_id=f"{pair_id}:{label}:{index}",
                pair_id=str(pair_id),
                label=str(label),
                base_contrast_id=_base_id(pair),
                source=str(pair.metadata.get("source", "")),
                vector=np.asarray(vector, dtype=np.float64),
            )
        )
    return rows


def _nearest_prompts(
    query: np.ndarray,
    candidates: Sequence[_PromptVector],
    *,
    nearest_k: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "sample_id": candidate.sample_id,
            "pair_id": candidate.pair_id,
            "label": candidate.label,
            "base_contrast_id": candidate.base_contrast_id,
            "source": candidate.source,
            "cosine": round(_cosine(query, candidate.vector), 6),
        }
        for candidate in candidates
    ]
    return sorted(rows, key=lambda row: float(row["cosine"]), reverse=True)[:nearest_k]


def _readiness(
    *,
    focus_projection_rows: Sequence[Mapping[str, Any]],
    source_same_base_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    margins = {str(row["direction_id"]): float(row["margin"]) for row in focus_projection_rows}
    source_original_margins = [
        float(row["original_source_target_joint_margin"]) for row in source_same_base_rows
    ]
    hard_residual = (
        margins.get("original_source_target_joint", 0.0) <= 0.0
        and margins.get("full_fresh_augmented", 0.0) <= 0.0
        and margins.get("fresh_source_only", 0.0) > 0.0
        and bool(source_original_margins)
        and min(source_original_margins) > 0.0
    )
    status = "hard_pair_geometry_residual" if hard_residual else "not_hard_residual"
    return {
        "status": status,
        "hard_residual": hard_residual,
        "criteria": {
            "original_joint_focus_margin": margins.get(
                "original_source_target_joint",
                0.0,
            ),
            "full_augmented_focus_margin": margins.get("full_fresh_augmented", 0.0),
            "fresh_only_focus_margin": margins.get("fresh_source_only", 0.0),
            "source_same_base_original_min_margin": (
                min(source_original_margins) if source_original_margins else 0.0
            ),
        },
    }


def _summary(
    *,
    focus_projection_rows: Sequence[Mapping[str, Any]],
    source_same_base_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    by_direction = {str(row["direction_id"]): row for row in focus_projection_rows}
    source_cosines = [
        float(row["delta_cosine_to_focus"]) for row in source_same_base_rows
    ]
    source_original_margins = [
        float(row["original_source_target_joint_margin"]) for row in source_same_base_rows
    ]
    return {
        "readiness": str(readiness.get("status", "")),
        "hard_pair_geometry_residual": bool(readiness.get("hard_residual", False)),
        "original_joint_focus_margin": _row_margin(
            by_direction,
            "original_source_target_joint",
        ),
        "full_augmented_focus_margin": _row_margin(
            by_direction,
            "full_fresh_augmented",
        ),
        "fresh_only_focus_margin": _row_margin(by_direction, "fresh_source_only"),
        "leave_focus_out_focus_margin": _row_margin(
            by_direction,
            "leave_focus_out_augmented",
        ),
        "mean_source_same_base_delta_cosine": (
            round(sum(source_cosines) / len(source_cosines), 6)
            if source_cosines
            else 0.0
        ),
        "min_source_same_base_delta_cosine": (
            round(min(source_cosines), 6) if source_cosines else 0.0
        ),
        "max_source_same_base_delta_cosine": (
            round(max(source_cosines), 6) if source_cosines else 0.0
        ),
        "source_same_base_original_min_margin": (
            round(min(source_original_margins), 6) if source_original_margins else 0.0
        ),
        "source_same_base_pairs": len(source_same_base_rows),
    }


def _row_margin(rows: Mapping[str, Mapping[str, Any]], direction_id: str) -> float:
    return float(rows.get(direction_id, {}).get("margin", 0.0))


def _resolve_focus_pair_id(
    pairs: Sequence[PairwiseExample],
    *,
    focus_base_contrast_id: str,
) -> str:
    matches = [pair.pair_id for pair in pairs if _base_id(pair) == focus_base_contrast_id]
    if len(matches) != 1:
        msg = (
            f"Expected exactly one fresh pair for {focus_base_contrast_id!r}; "
            f"found {len(matches)}."
        )
        raise ValueError(msg)
    return matches[0]


def _base_id(pair: PairwiseExample) -> str:
    raw = str(pair.metadata.get("base_contrast_id") or pair.pair_id)
    raw = raw.removeprefix(_GENERATED_PREFIX)
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    return raw


def _mask(dataset: _Dataset, pair_ids: set[str]) -> np.ndarray:
    return np.asarray([str(pair_id) in pair_ids for pair_id in dataset.pair_ids], dtype=bool)


def _unique_pair_ids(dataset: _Dataset) -> set[str]:
    return {str(pair_id) for pair_id in dataset.pair_ids}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(left @ right / denominator)


def _focus_projection_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No focus projection rows."]
    lines = [
        "| Direction | Margin | Positive projection | Negative projection | Delta cosine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('direction_id', '')}` | "
            f"{float(row.get('margin', 0.0)):+.3f} | "
            f"{float(row.get('positive_projection', 0.0)):+.3f} | "
            f"{float(row.get('negative_projection', 0.0)):+.3f} | "
            f"{float(row.get('delta_cosine_to_direction', 0.0)):+.3f} |"
        )
    return lines


def _same_base_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No same-base source rows."]
    lines = [
        "| Pair | Delta cosine | Original margin | Full augmented margin | Fresh-only margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{_short_pair_id(str(row.get('pair_id', '')))}` | "
            f"{float(row.get('delta_cosine_to_focus', 0.0)):+.3f} | "
            f"{float(row.get('original_source_target_joint_margin', 0.0)):+.3f} | "
            f"{float(row.get('full_fresh_augmented_margin', 0.0)):+.3f} | "
            f"{float(row.get('fresh_source_only_margin', 0.0)):+.3f} |"
        )
    return lines


def _delta_neighbor_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    if not rows:
        return ["No source delta neighbors."]
    lines = [
        "| Pair | Base | Source | Delta cosine |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{_short_pair_id(str(row.get('pair_id', '')))}` | "
            f"`{row.get('base_contrast_id', '')}` | "
            f"`{row.get('source', '')}` | "
            f"{float(row.get('delta_cosine_to_focus', 0.0)):+.3f} |"
        )
    return lines


def _prompt_neighbor_tables(raw_sections: object) -> list[str]:
    sections = _mapping(raw_sections)
    if not sections:
        return ["No prompt neighbor sections."]
    lines: list[str] = []
    for section_name, raw_rows in sections.items():
        lines.extend(
            [
                f"### `{section_name}`",
                "",
                "| Pair | Label | Base | Source | Cosine |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in [_mapping(item) for item in _sequence(raw_rows)]:
            lines.append(
                "| "
                f"`{_short_pair_id(str(row.get('pair_id', '')))}` | "
                f"`{row.get('label', '')}` | "
                f"`{row.get('base_contrast_id', '')}` | "
                f"`{row.get('source', '')}` | "
                f"{float(row.get('cosine', 0.0)):+.3f} |"
            )
        lines.append("")
    return lines


def _short_pair_id(pair_id: str) -> str:
    return re.sub(r"^generated-fault::", "", pair_id)


def _mapping(raw_value: object) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _sequence(raw_value: object) -> list[Any]:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str):
        return list(raw_value)
    return []
