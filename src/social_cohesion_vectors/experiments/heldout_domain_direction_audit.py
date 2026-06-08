"""Held-out domain bridge-training audit for cross-benchmark directions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.transfer import load_activation_payload

_EPSILON = 1e-12


@dataclass(frozen=True)
class _DomainDataset:
    name: str
    activations: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    pair_groups: Mapping[str, str]
    pair_metadata: Mapping[str, Mapping[str, str]]


def run_heldout_domain_direction_audit_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    source_name: str = "source",
    target_name: str = "target",
    source_group_key: str = "source",
    target_group_key: str = "source",
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Train bridge directions and evaluate held-out source families."""

    source = _load_domain_dataset(
        name=source_name,
        activation_npz=source_activation_npz,
        pairs_path=source_pairs_path,
        group_key=source_group_key,
    )
    target = _load_domain_dataset(
        name=target_name,
        activation_npz=target_activation_npz,
        pairs_path=target_pairs_path,
        group_key=target_group_key,
    )
    _validate_shared_dim(source, target)

    target_holdout_folds = [
        _bridge_fold(
            train_primary=source,
            train_secondary=target,
            held_out_dataset=target,
            held_out_group=group,
        )
        for group in _groups(target)
    ]
    source_holdout_folds = [
        _bridge_fold(
            train_primary=target,
            train_secondary=source,
            held_out_dataset=source,
            held_out_group=group,
        )
        for group in _groups(source)
    ]
    readiness = _readiness(
        source_holdout_folds=source_holdout_folds,
        target_holdout_folds=target_holdout_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "heldout_domain_direction_audit",
        "description": (
            "Trains bridge directions on one full benchmark domain plus all but "
            "one source family from the other domain, then evaluates the held-out "
            "source family."
        ),
        "inputs": {
            "source_name": source_name,
            "target_name": target_name,
            "source_activation_npz": str(source_activation_npz),
            "source_pairs_path": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs_path": str(target_pairs_path),
            "source_group_key": source_group_key,
            "target_group_key": target_group_key,
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "source_groups": len(_groups(source)),
            "target_groups": len(_groups(target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": _summary(
            source_holdout_folds=source_holdout_folds,
            target_holdout_folds=target_holdout_folds,
            readiness=readiness,
        ),
        "readiness": readiness,
        "source_holdout_folds": source_holdout_folds,
        "target_holdout_folds": target_holdout_folds,
        "interpretation_guardrail": (
            "A held-out domain bridge-training pass supports a text-benchmark "
            "activation diagnostic. It does not establish a human, neural, "
            "clinical, deployment, or causal steering claim."
        ),
    }


def run_minimal_bridge_direction_audit_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    source_name: str = "source",
    target_name: str = "target",
    source_group_key: str = "source",
    target_group_key: str = "source",
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Evaluate all same-domain bridge source-family subset sizes."""

    source = _load_domain_dataset(
        name=source_name,
        activation_npz=source_activation_npz,
        pairs_path=source_pairs_path,
        group_key=source_group_key,
    )
    target = _load_domain_dataset(
        name=target_name,
        activation_npz=target_activation_npz,
        pairs_path=target_pairs_path,
        group_key=target_group_key,
    )
    _validate_shared_dim(source, target)

    target_ablation_folds = _bridge_ablation_folds(
        train_primary=source,
        train_secondary=target,
        held_out_dataset=target,
    )
    source_ablation_folds = _bridge_ablation_folds(
        train_primary=target,
        train_secondary=source,
        held_out_dataset=source,
    )
    target_by_count = _ablation_by_bridge_count(
        target_ablation_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    source_by_count = _ablation_by_bridge_count(
        source_ablation_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    readiness = _minimal_bridge_readiness(
        source_by_count=source_by_count,
        target_by_count=target_by_count,
    )
    return {
        "experiment": "minimal_bridge_direction_audit",
        "description": (
            "Ablates same-domain bridge source families while training on the "
            "full opposite benchmark domain, then evaluates held-out source "
            "families."
        ),
        "inputs": {
            "source_name": source_name,
            "target_name": target_name,
            "source_activation_npz": str(source_activation_npz),
            "source_pairs_path": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs_path": str(target_pairs_path),
            "source_group_key": source_group_key,
            "target_group_key": target_group_key,
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "source_groups": len(_groups(source)),
            "target_groups": len(_groups(target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": {
            "readiness": readiness["status"],
            "ready_for_minimal_bridge_claims": readiness["ready"],
            "source_min_ready_bridge_groups": _min_ready_bridge_count(
                source_by_count
            ),
            "target_min_ready_bridge_groups": _min_ready_bridge_count(
                target_by_count
            ),
            "source_fold_count": len(source_ablation_folds),
            "target_fold_count": len(target_ablation_folds),
            "source_failed_fold_count": _failed_fold_count(
                source_ablation_folds,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            ),
            "target_failed_fold_count": _failed_fold_count(
                target_ablation_folds,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            ),
        },
        "readiness": readiness,
        "source_by_bridge_count": source_by_count,
        "target_by_bridge_count": target_by_count,
        "source_ablation_folds": source_ablation_folds,
        "target_ablation_folds": target_ablation_folds,
        "interpretation_guardrail": (
            "A minimal bridge pass estimates how much same-domain bridge data is "
            "needed for a text-benchmark activation diagnostic. It does not "
            "establish a human, neural, clinical, deployment, or causal steering "
            "claim."
        ),
    }


def run_pair_bridge_direction_audit_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    source_name: str = "source",
    target_name: str = "target",
    source_group_key: str = "source",
    target_group_key: str = "source",
    bridge_stratum_key: str = "slack_options_tested",
    max_subsets_per_count: int = 32,
    max_bridge_pair_count: int | None = None,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Evaluate path-stratified bridge-pair subset sizes."""

    if max_subsets_per_count < 1:
        raise ValueError("max_subsets_per_count must be at least 1.")
    if max_bridge_pair_count is not None and max_bridge_pair_count < 0:
        raise ValueError("max_bridge_pair_count must be non-negative.")

    source = _load_domain_dataset(
        name=source_name,
        activation_npz=source_activation_npz,
        pairs_path=source_pairs_path,
        group_key=source_group_key,
    )
    target = _load_domain_dataset(
        name=target_name,
        activation_npz=target_activation_npz,
        pairs_path=target_pairs_path,
        group_key=target_group_key,
    )
    _validate_shared_dim(source, target)

    target_ablation_folds = _pair_bridge_ablation_folds(
        train_primary=source,
        train_secondary=target,
        held_out_dataset=target,
        bridge_stratum_key=bridge_stratum_key,
        max_subsets_per_count=max_subsets_per_count,
        max_bridge_pair_count=max_bridge_pair_count,
    )
    source_ablation_folds = _pair_bridge_ablation_folds(
        train_primary=target,
        train_secondary=source,
        held_out_dataset=source,
        bridge_stratum_key=bridge_stratum_key,
        max_subsets_per_count=max_subsets_per_count,
        max_bridge_pair_count=max_bridge_pair_count,
    )
    target_by_count = _ablation_by_bridge_pair_count(
        target_ablation_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    source_by_count = _ablation_by_bridge_pair_count(
        source_ablation_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    readiness = _pair_bridge_readiness(
        source_by_count=source_by_count,
        target_by_count=target_by_count,
    )
    return {
        "experiment": "pair_bridge_direction_audit",
        "description": (
            "Ablates individual same-domain bridge pairs while training on the "
            "full opposite benchmark domain. Large bridge-pair combinations are "
            "sampled with deterministic path-stratified subsets."
        ),
        "inputs": {
            "source_name": source_name,
            "target_name": target_name,
            "source_activation_npz": str(source_activation_npz),
            "source_pairs_path": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs_path": str(target_pairs_path),
            "source_group_key": source_group_key,
            "target_group_key": target_group_key,
            "bridge_stratum_key": bridge_stratum_key,
            "max_subsets_per_count": int(max_subsets_per_count),
            "max_bridge_pair_count": max_bridge_pair_count,
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "source_groups": len(_groups(source)),
            "target_groups": len(_groups(target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": {
            "readiness": readiness["status"],
            "ready_for_pair_bridge_claims": readiness["ready"],
            "source_min_ready_bridge_pairs": _min_ready_bridge_pair_count(
                source_by_count
            ),
            "target_min_ready_bridge_pairs": _min_ready_bridge_pair_count(
                target_by_count
            ),
            "source_min_exhaustive_ready_bridge_pairs": (
                _min_ready_bridge_pair_count(source_by_count, require_exhaustive=True)
            ),
            "target_min_exhaustive_ready_bridge_pairs": (
                _min_ready_bridge_pair_count(target_by_count, require_exhaustive=True)
            ),
            "source_evaluated_subset_count": len(source_ablation_folds),
            "target_evaluated_subset_count": len(target_ablation_folds),
            "source_failed_subset_count": _failed_fold_count(
                source_ablation_folds,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            ),
            "target_failed_subset_count": _failed_fold_count(
                target_ablation_folds,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            ),
        },
        "readiness": readiness,
        "source_by_bridge_pair_count": source_by_count,
        "target_by_bridge_pair_count": target_by_count,
        "source_pair_ablation_folds": source_ablation_folds,
        "target_pair_ablation_folds": target_ablation_folds,
        "interpretation_guardrail": (
            "A pair bridge pass estimates sampled path-stratified text-benchmark "
            "bridge sufficiency. It does not establish a human, neural, "
            "clinical, deployment, or causal steering claim."
        ),
    }


def save_heldout_domain_direction_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown held-out domain audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_heldout_domain_direction_audit_markdown(report),
        encoding="utf-8",
    )


def save_minimal_bridge_direction_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown minimal bridge audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_minimal_bridge_direction_audit_markdown(report),
        encoding="utf-8",
    )


def save_pair_bridge_direction_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown pair bridge audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_pair_bridge_direction_audit_markdown(report),
        encoding="utf-8",
    )


def render_heldout_domain_direction_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a held-out domain bridge-training audit as Markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Held-Out Domain Direction Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Source benchmark: `{inputs.get('source_name', '')}`",
        f"- Target benchmark: `{inputs.get('target_name', '')}`",
        f"- Source group key: `{inputs.get('source_group_key', '')}`",
        f"- Target group key: `{inputs.get('target_group_key', '')}`",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        f"- Source pairs/groups: {int(inputs.get('source_pairs', 0))}/"
        f"{int(inputs.get('source_groups', 0))}",
        f"- Target pairs/groups: {int(inputs.get('target_pairs', 0))}/"
        f"{int(inputs.get('target_groups', 0))}",
        "",
        "## Summary",
        "",
        f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
        f"- Ready for held-out domain claims: "
        f"{bool(summary.get('ready_for_heldout_domain_claims', False))}",
        f"- Source holdout min accuracy: "
        f"{float(summary.get('source_holdout_min_accuracy', 0.0)):.3f}",
        f"- Source holdout min margin: "
        f"{float(summary.get('source_holdout_min_margin', 0.0)):+.3f}",
        f"- Target holdout min accuracy: "
        f"{float(summary.get('target_holdout_min_accuracy', 0.0)):.3f}",
        f"- Target holdout min margin: "
        f"{float(summary.get('target_holdout_min_margin', 0.0)):+.3f}",
        f"- Failed pairs: {int(summary.get('failed_pair_count', 0))}",
        "",
        "## Target Holdout Folds",
        "",
        *_fold_table(report.get("target_holdout_folds")),
        "",
        "## Source Holdout Folds",
        "",
        *_fold_table(report.get("source_holdout_folds")),
        "",
        "## Failed Pairs",
        "",
        *_failed_pair_table(report),
        "",
        "## Interpretation Guardrail",
        "",
        str(report.get("interpretation_guardrail", "")),
        "",
    ]
    return "\n".join(lines)


def render_minimal_bridge_direction_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a minimal bridge ablation audit as Markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Minimal Bridge Direction Audit",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Source benchmark: `{inputs.get('source_name', '')}`",
            f"- Target benchmark: `{inputs.get('target_name', '')}`",
            f"- Source group key: `{inputs.get('source_group_key', '')}`",
            f"- Target group key: `{inputs.get('target_group_key', '')}`",
            f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
            f"- Source pairs/groups: {int(inputs.get('source_pairs', 0))}/"
            f"{int(inputs.get('source_groups', 0))}",
            f"- Target pairs/groups: {int(inputs.get('target_pairs', 0))}/"
            f"{int(inputs.get('target_groups', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for minimal bridge claims: "
            f"{bool(summary.get('ready_for_minimal_bridge_claims', False))}",
            f"- Source minimum ready bridge groups: "
            f"{summary.get('source_min_ready_bridge_groups')}",
            f"- Target minimum ready bridge groups: "
            f"{summary.get('target_min_ready_bridge_groups')}",
            f"- Source failed fold count: "
            f"{int(summary.get('source_failed_fold_count', 0))}",
            f"- Target failed fold count: "
            f"{int(summary.get('target_failed_fold_count', 0))}",
            "",
            "## Target Holdout By Bridge Count",
            "",
            *_ablation_summary_table(report.get("target_by_bridge_count")),
            "",
            "## Source Holdout By Bridge Count",
            "",
            *_ablation_summary_table(report.get("source_by_bridge_count")),
            "",
            "## Failed Ablation Folds",
            "",
            *_failed_ablation_fold_table(report),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def render_pair_bridge_direction_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a pair bridge ablation audit as Markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Pair Bridge Direction Audit",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Source benchmark: `{inputs.get('source_name', '')}`",
            f"- Target benchmark: `{inputs.get('target_name', '')}`",
            f"- Source group key: `{inputs.get('source_group_key', '')}`",
            f"- Target group key: `{inputs.get('target_group_key', '')}`",
            f"- Bridge stratum key: `{inputs.get('bridge_stratum_key', '')}`",
            f"- Max subsets per count: "
            f"{int(inputs.get('max_subsets_per_count', 0))}",
            f"- Max bridge pair count: {inputs.get('max_bridge_pair_count')}",
            f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
            f"- Source pairs/groups: {int(inputs.get('source_pairs', 0))}/"
            f"{int(inputs.get('source_groups', 0))}",
            f"- Target pairs/groups: {int(inputs.get('target_pairs', 0))}/"
            f"{int(inputs.get('target_groups', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for pair bridge claims: "
            f"{bool(summary.get('ready_for_pair_bridge_claims', False))}",
            f"- Source minimum ready bridge pairs: "
            f"{summary.get('source_min_ready_bridge_pairs')}",
            f"- Target minimum ready bridge pairs: "
            f"{summary.get('target_min_ready_bridge_pairs')}",
            f"- Source minimum exhaustive ready bridge pairs: "
            f"{summary.get('source_min_exhaustive_ready_bridge_pairs')}",
            f"- Target minimum exhaustive ready bridge pairs: "
            f"{summary.get('target_min_exhaustive_ready_bridge_pairs')}",
            f"- Source failed sampled subsets: "
            f"{int(summary.get('source_failed_subset_count', 0))}",
            f"- Target failed sampled subsets: "
            f"{int(summary.get('target_failed_subset_count', 0))}",
            "",
            "## Target Holdout By Bridge Pair Count",
            "",
            *_pair_ablation_summary_table(
                report.get("target_by_bridge_pair_count")
            ),
            "",
            "## Source Holdout By Bridge Pair Count",
            "",
            *_pair_ablation_summary_table(
                report.get("source_by_bridge_pair_count")
            ),
            "",
            "## Failed Sampled Folds",
            "",
            *_failed_pair_bridge_fold_table(report),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def _load_domain_dataset(
    *,
    name: str,
    activation_npz: str | Path,
    pairs_path: str | Path,
    group_key: str,
) -> _DomainDataset:
    payload = load_activation_payload(activation_npz)
    pairs = load_pairwise_examples_jsonl(pairs_path)
    pair_groups: dict[str, str] = {}
    pair_metadata: dict[str, dict[str, str]] = {}
    for pair in pairs:
        raw_group = pair.metadata.get(group_key)
        group = str(raw_group or "").strip()
        if not group:
            msg = f"Pair {pair.pair_id} is missing metadata key {group_key!r}."
            raise ValueError(msg)
        pair_groups[pair.pair_id] = group
        pair_metadata[pair.pair_id] = {
            str(key): str(value) for key, value in pair.metadata.items()
        }

    activation_pairs = {str(pair_id) for pair_id in payload.pair_ids}
    pair_metadata_ids = set(pair_groups)
    missing_metadata = sorted(activation_pairs - pair_metadata_ids)
    missing_activations = sorted(pair_metadata_ids - activation_pairs)
    if missing_metadata or missing_activations:
        msg = (
            f"Activation/pair metadata mismatch for {name}: "
            f"missing_metadata={missing_metadata[:5]} "
            f"missing_activations={missing_activations[:5]}"
        )
        raise ValueError(msg)

    return _DomainDataset(
        name=name,
        activations=np.asarray(payload.activations, dtype=np.float64),
        labels=np.asarray(payload.labels, dtype=str),
        pair_ids=np.asarray(payload.pair_ids, dtype=str),
        pair_groups=pair_groups,
        pair_metadata=pair_metadata,
    )


def _validate_shared_dim(source: _DomainDataset, target: _DomainDataset) -> None:
    if int(source.activations.shape[1]) != int(target.activations.shape[1]):
        raise ValueError("Source and target activation dimensions must match.")


def _bridge_fold(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
    held_out_group: str,
) -> dict[str, Any]:
    held_out_pairs = {
        pair_id
        for pair_id, group in held_out_dataset.pair_groups.items()
        if group == held_out_group
    }
    secondary_train_pairs = _unique_pair_ids(train_secondary) - held_out_pairs
    train_parts = (
        _training_part(train_primary, _unique_pair_ids(train_primary)),
        _training_part(train_secondary, secondary_train_pairs),
    )
    train_activations = np.concatenate([part[0] for part in train_parts], axis=0)
    train_labels = np.concatenate([part[1] for part in train_parts], axis=0)
    direction = train_direction_from_arrays(
        train_activations,
        labels=train_labels,
    ).direction
    evaluation = _evaluate_pairwise_projection(
        held_out_dataset,
        direction=direction,
        pair_ids=held_out_pairs,
    )
    return {
        "held_out_dataset": held_out_dataset.name,
        "held_out_group": held_out_group,
        "train_primary_dataset": train_primary.name,
        "train_secondary_dataset": train_secondary.name,
        "train_primary_pairs": len(_unique_pair_ids(train_primary)),
        "train_secondary_pairs": len(secondary_train_pairs),
        **evaluation,
    }


def _bridge_ablation_folds(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    groups = _groups(held_out_dataset)
    for held_out_group in groups:
        held_out_pairs = {
            pair_id
            for pair_id, group in held_out_dataset.pair_groups.items()
            if group == held_out_group
        }
        bridge_candidates = [group for group in groups if group != held_out_group]
        for bridge_count in range(len(bridge_candidates) + 1):
            for bridge_groups in combinations(bridge_candidates, bridge_count):
                bridge_group_set = set(bridge_groups)
                secondary_train_pairs = {
                    pair_id
                    for pair_id, group in train_secondary.pair_groups.items()
                    if group in bridge_group_set
                }
                train_parts = (
                    _training_part(train_primary, _unique_pair_ids(train_primary)),
                    _training_part(train_secondary, secondary_train_pairs),
                )
                train_activations = np.concatenate(
                    [part[0] for part in train_parts],
                    axis=0,
                )
                train_labels = np.concatenate([part[1] for part in train_parts], axis=0)
                direction = train_direction_from_arrays(
                    train_activations,
                    labels=train_labels,
                ).direction
                evaluation = _evaluate_pairwise_projection(
                    held_out_dataset,
                    direction=direction,
                    pair_ids=held_out_pairs,
                )
                folds.append(
                    {
                        "held_out_dataset": held_out_dataset.name,
                        "held_out_group": held_out_group,
                        "train_primary_dataset": train_primary.name,
                        "train_secondary_dataset": train_secondary.name,
                        "train_primary_pairs": len(_unique_pair_ids(train_primary)),
                        "train_secondary_pairs": len(secondary_train_pairs),
                        "bridge_group_count": bridge_count,
                        "bridge_groups": list(bridge_groups),
                        **evaluation,
                    }
                )
    return folds


def _pair_bridge_ablation_folds(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
    bridge_stratum_key: str,
    max_subsets_per_count: int,
    max_bridge_pair_count: int | None,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held_out_group in _groups(held_out_dataset):
        held_out_pairs = {
            pair_id
            for pair_id, group in held_out_dataset.pair_groups.items()
            if group == held_out_group
        }
        bridge_candidates = sorted(
            pair_id
            for pair_id, group in train_secondary.pair_groups.items()
            if group != held_out_group
        )
        candidate_limit = len(bridge_candidates)
        if max_bridge_pair_count is not None:
            candidate_limit = min(candidate_limit, max_bridge_pair_count)
        for bridge_pair_count in range(candidate_limit + 1):
            bridge_subsets, exhaustive = _pair_bridge_subsets(
                train_secondary,
                candidate_pairs=bridge_candidates,
                pair_count=bridge_pair_count,
                stratum_key=bridge_stratum_key,
                max_subsets_per_count=max_subsets_per_count,
            )
            for subset_index, bridge_pairs in enumerate(bridge_subsets):
                bridge_pair_set = set(bridge_pairs)
                train_parts = (
                    _training_part(train_primary, _unique_pair_ids(train_primary)),
                    _training_part(train_secondary, bridge_pair_set),
                )
                train_activations = np.concatenate(
                    [part[0] for part in train_parts],
                    axis=0,
                )
                train_labels = np.concatenate([part[1] for part in train_parts], axis=0)
                direction = train_direction_from_arrays(
                    train_activations,
                    labels=train_labels,
                ).direction
                evaluation = _evaluate_pairwise_projection(
                    held_out_dataset,
                    direction=direction,
                    pair_ids=held_out_pairs,
                )
                bridge_path_values = _bridge_path_values(
                    train_secondary,
                    bridge_pair_set,
                    stratum_key=bridge_stratum_key,
                )
                folds.append(
                    {
                        "held_out_dataset": held_out_dataset.name,
                        "held_out_group": held_out_group,
                        "train_primary_dataset": train_primary.name,
                        "train_secondary_dataset": train_secondary.name,
                        "train_primary_pairs": len(_unique_pair_ids(train_primary)),
                        "candidate_bridge_pairs": len(bridge_candidates),
                        "bridge_pair_count": bridge_pair_count,
                        "bridge_pairs": list(bridge_pairs),
                        "bridge_strata": _bridge_strata(
                            train_secondary,
                            bridge_pair_set,
                            stratum_key=bridge_stratum_key,
                        ),
                        "bridge_path_values": bridge_path_values,
                        "bridge_path_count": len(bridge_path_values),
                        "subset_index": subset_index,
                        "subset_count_for_pair_count": len(bridge_subsets),
                        "exhaustive_bridge_subsets": exhaustive,
                        **evaluation,
                    }
                )
    return folds


def _training_part(
    dataset: _DomainDataset,
    pair_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    mask = _prompt_mask(dataset, pair_ids)
    return dataset.activations[mask], dataset.labels[mask]


def _evaluate_pairwise_projection(
    dataset: _DomainDataset,
    *,
    direction: np.ndarray,
    pair_ids: set[str],
) -> dict[str, Any]:
    mask = _prompt_mask(dataset, pair_ids)
    projections = np.asarray(dataset.activations[mask] @ direction, dtype=np.float64)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(
        dataset.pair_ids[mask],
        dataset.labels[mask],
        projections,
        strict=True,
    ):
        grouped[str(pair_id)][str(label)].append(float(projection))

    pair_rows: list[dict[str, Any]] = []
    for pair_id, label_scores in sorted(grouped.items()):
        if "positive" not in label_scores or "negative" not in label_scores:
            continue
        positive_projection = float(np.mean(label_scores["positive"]))
        negative_projection = float(np.mean(label_scores["negative"]))
        margin = positive_projection - negative_projection
        pair_rows.append(
            {
                "pair_id": pair_id,
                "positive_projection": round(positive_projection, 6),
                "negative_projection": round(negative_projection, 6),
                "margin": round(margin, 6),
                "passed": margin > 0.0,
            }
        )
    margins = [float(row["margin"]) for row in pair_rows]
    failed_pairs = [row for row in pair_rows if not bool(row["passed"])]
    return {
        "test_pairs": len(pair_rows),
        "pairwise_accuracy": (
            round(sum(margin > 0.0 for margin in margins) / len(margins), 6)
            if margins
            else 0.0
        ),
        "mean_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        "min_margin": round(float(np.min(margins)), 6) if margins else 0.0,
        "max_margin": round(float(np.max(margins)), 6) if margins else 0.0,
        "failed_pair_count": len(failed_pairs),
        "failed_pairs": failed_pairs,
        "pair_margins": pair_rows,
    }


def _readiness(
    *,
    source_holdout_folds: Sequence[Mapping[str, Any]],
    target_holdout_folds: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    all_folds = [*source_holdout_folds, *target_holdout_folds]
    gates = [
        _gate("folds_present", float(len(all_folds)), 1.0),
        _gate(
            "min_pairwise_accuracy_per_fold",
            min(_fold_values(all_folds, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            "min_margin_per_fold",
            min(_fold_values(all_folds, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            "failed_pair_count",
            -float(sum(int(fold.get("failed_pair_count", 0)) for fold in all_folds)),
            0.0,
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "heldout_domain_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _summary(
    *,
    source_holdout_folds: Sequence[Mapping[str, Any]],
    target_holdout_folds: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    all_folds = [*source_holdout_folds, *target_holdout_folds]
    return {
        "readiness": str(readiness.get("status", "not_ready")),
        "ready_for_heldout_domain_claims": bool(readiness.get("ready", False)),
        "source_holdout_folds": len(source_holdout_folds),
        "source_holdout_min_accuracy": min(
            _fold_values(source_holdout_folds, "pairwise_accuracy"),
            default=0.0,
        ),
        "source_holdout_min_margin": min(
            _fold_values(source_holdout_folds, "min_margin"),
            default=0.0,
        ),
        "target_holdout_folds": len(target_holdout_folds),
        "target_holdout_min_accuracy": min(
            _fold_values(target_holdout_folds, "pairwise_accuracy"),
            default=0.0,
        ),
        "target_holdout_min_margin": min(
            _fold_values(target_holdout_folds, "min_margin"),
            default=0.0,
        ),
        "failed_pair_count": sum(
            int(fold.get("failed_pair_count", 0)) for fold in all_folds
        ),
    }


def _ablation_by_bridge_count(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bridge_counts = sorted({int(fold.get("bridge_group_count", 0)) for fold in folds})
    for bridge_count in bridge_counts:
        count_folds = [
            fold
            for fold in folds
            if int(fold.get("bridge_group_count", 0)) == bridge_count
        ]
        failed_folds = [
            fold
            for fold in count_folds
            if _fold_failed(
                fold,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            )
        ]
        rows.append(
            {
                "bridge_group_count": bridge_count,
                "folds": len(count_folds),
                "min_accuracy": min(
                    _fold_values(count_folds, "pairwise_accuracy"),
                    default=0.0,
                ),
                "min_margin": min(_fold_values(count_folds, "min_margin"), default=0.0),
                "failed_folds": len(failed_folds),
                "failed_pairs": sum(
                    int(fold.get("failed_pair_count", 0)) for fold in count_folds
                ),
                "ready": not failed_folds,
            }
        )
    return rows


def _minimal_bridge_readiness(
    *,
    source_by_count: Sequence[Mapping[str, Any]],
    target_by_count: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = [
        _gate("source_min_ready_bridge_count_exists", _exists_ready(source_by_count), 1.0),
        _gate("target_min_ready_bridge_count_exists", _exists_ready(target_by_count), 1.0),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "minimal_bridge_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _ablation_by_bridge_pair_count(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bridge_counts = sorted({int(fold.get("bridge_pair_count", 0)) for fold in folds})
    for bridge_count in bridge_counts:
        count_folds = [
            fold
            for fold in folds
            if int(fold.get("bridge_pair_count", 0)) == bridge_count
        ]
        failed_folds = [
            fold
            for fold in count_folds
            if _fold_failed(
                fold,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            )
        ]
        rows.append(
            {
                "bridge_pair_count": bridge_count,
                "evaluated_subsets": len(count_folds),
                "heldout_groups": len(
                    {str(fold.get("held_out_group", "")) for fold in count_folds}
                ),
                "min_accuracy": min(
                    _fold_values(count_folds, "pairwise_accuracy"),
                    default=0.0,
                ),
                "min_margin": min(_fold_values(count_folds, "min_margin"), default=0.0),
                "failed_subsets": len(failed_folds),
                "failed_pairs": sum(
                    int(fold.get("failed_pair_count", 0)) for fold in count_folds
                ),
                "min_bridge_path_count": min(
                    _fold_int_values(count_folds, "bridge_path_count"),
                    default=0,
                ),
                "max_bridge_path_count": max(
                    _fold_int_values(count_folds, "bridge_path_count"),
                    default=0,
                ),
                "exhaustive": all(
                    bool(fold.get("exhaustive_bridge_subsets", False))
                    for fold in count_folds
                ),
                "ready": not failed_folds,
            }
        )
    return rows


def _pair_bridge_readiness(
    *,
    source_by_count: Sequence[Mapping[str, Any]],
    target_by_count: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = [
        _gate("source_min_ready_pair_count_exists", _exists_ready(source_by_count), 1.0),
        _gate("target_min_ready_pair_count_exists", _exists_ready(target_by_count), 1.0),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "sampled_pair_bridge_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _exists_ready(rows: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if any(bool(row.get("ready", False)) for row in rows) else 0.0


def _min_ready_bridge_count(rows: Sequence[Mapping[str, Any]]) -> int | None:
    ready_counts = [
        int(row.get("bridge_group_count", 0))
        for row in rows
        if bool(row.get("ready", False))
    ]
    return min(ready_counts) if ready_counts else None


def _min_ready_bridge_pair_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_exhaustive: bool = False,
) -> int | None:
    ready_counts = [
        int(row.get("bridge_pair_count", 0))
        for row in rows
        if bool(row.get("ready", False))
        and (not require_exhaustive or bool(row.get("exhaustive", False)))
    ]
    return min(ready_counts) if ready_counts else None


def _failed_fold_count(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> int:
    return sum(
        1
        for fold in folds
        if _fold_failed(
            fold,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        )
    )


def _fold_failed(
    fold: Mapping[str, Any],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> bool:
    return (
        float(fold.get("pairwise_accuracy", 0.0)) < min_pairwise_accuracy
        or float(fold.get("min_margin", 0.0)) < min_margin
    )


def _groups(dataset: _DomainDataset) -> list[str]:
    return sorted(set(dataset.pair_groups.values()))


def _unique_pair_ids(dataset: _DomainDataset) -> set[str]:
    return {str(pair_id) for pair_id in dataset.pair_ids}


def _prompt_mask(dataset: _DomainDataset, pair_ids: set[str]) -> np.ndarray:
    return np.asarray([str(pair_id) in pair_ids for pair_id in dataset.pair_ids])


def _fold_values(folds: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(fold.get(key, 0.0)) for fold in folds]


def _fold_int_values(folds: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    return [int(fold.get(key, 0)) for fold in folds]


def _pair_bridge_subsets(
    dataset: _DomainDataset,
    *,
    candidate_pairs: Sequence[str],
    pair_count: int,
    stratum_key: str,
    max_subsets_per_count: int,
) -> tuple[list[tuple[str, ...]], bool]:
    candidates = sorted(candidate_pairs)
    if pair_count == 0:
        return [()], True
    if pair_count >= len(candidates):
        return [tuple(candidates)], True

    combination_total = _combination_count(len(candidates), pair_count)
    if combination_total <= max_subsets_per_count:
        return list(combinations(candidates, pair_count)), True

    grouped = _pairs_by_stratum(
        dataset,
        pair_ids=candidates,
        stratum_key=stratum_key,
    )
    subsets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    attempt_limit = max_subsets_per_count * max(4, len(candidates) * 2)
    for attempt in range(attempt_limit):
        subset = _round_robin_pair_subset(
            grouped,
            pair_count=pair_count,
            attempt=attempt,
        )
        if subset and subset not in seen:
            seen.add(subset)
            subsets.append(subset)
            if len(subsets) >= max_subsets_per_count:
                return subsets, False

    for attempt in range(attempt_limit):
        rotated = _rotate(candidates, attempt)
        subset = tuple(sorted(rotated[:pair_count]))
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
            if len(subsets) >= max_subsets_per_count:
                break
    return subsets, False


def _combination_count(item_count: int, choose_count: int) -> int:
    if choose_count < 0 or choose_count > item_count:
        return 0
    choose_count = min(choose_count, item_count - choose_count)
    numerator = 1
    denominator = 1
    for index in range(1, choose_count + 1):
        numerator *= item_count - choose_count + index
        denominator *= index
    return numerator // denominator


def _pairs_by_stratum(
    dataset: _DomainDataset,
    *,
    pair_ids: Sequence[str],
    stratum_key: str,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for pair_id in sorted(pair_ids):
        grouped[_pair_stratum(dataset, pair_id, stratum_key=stratum_key)].append(
            pair_id
        )
    return {stratum: sorted(ids) for stratum, ids in sorted(grouped.items())}


def _round_robin_pair_subset(
    grouped: Mapping[str, Sequence[str]],
    *,
    pair_count: int,
    attempt: int,
) -> tuple[str, ...]:
    strata = sorted(grouped)
    if not strata:
        return ()
    stratum_order = _rotate(strata, attempt % len(strata))
    pair_offset = attempt // len(strata)
    queues = {
        stratum: _rotate(list(grouped[stratum]), pair_offset)
        for stratum in stratum_order
    }
    selected: list[str] = []
    while len(selected) < pair_count:
        advanced = False
        for stratum in stratum_order:
            queue = queues[stratum]
            if not queue:
                continue
            selected.append(queue.pop(0))
            advanced = True
            if len(selected) >= pair_count:
                break
        if not advanced:
            break
    return tuple(sorted(selected))


def _rotate(items: Sequence[str], offset: int) -> list[str]:
    values = list(items)
    if not values:
        return []
    offset %= len(values)
    return [*values[offset:], *values[:offset]]


def _pair_stratum(
    dataset: _DomainDataset,
    pair_id: str,
    *,
    stratum_key: str,
) -> str:
    metadata = dataset.pair_metadata.get(pair_id, {})
    raw_value = str(metadata.get(stratum_key, "")).strip()
    if not raw_value:
        return "missing"
    parts = _split_metadata_values(raw_value)
    return ",".join(sorted(parts)) if parts else raw_value


def _bridge_strata(
    dataset: _DomainDataset,
    pair_ids: set[str],
    *,
    stratum_key: str,
) -> list[str]:
    return sorted(
        {_pair_stratum(dataset, pair_id, stratum_key=stratum_key) for pair_id in pair_ids}
    )


def _bridge_path_values(
    dataset: _DomainDataset,
    pair_ids: set[str],
    *,
    stratum_key: str,
) -> list[str]:
    values: set[str] = set()
    for pair_id in pair_ids:
        raw_value = dataset.pair_metadata.get(pair_id, {}).get(stratum_key, "")
        values.update(_split_metadata_values(raw_value))
    return sorted(values)


def _split_metadata_values(raw_value: object) -> list[str]:
    return [
        part.strip()
        for part in str(raw_value).split(",")
        if part.strip()
    ]


def _gate(gate_id: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "value": round(value, 6),
        "threshold": round(threshold, 6),
        "passed": value + _EPSILON >= threshold,
    }


def _fold_table(raw_folds: object) -> list[str]:
    folds = [_mapping(fold) for fold in _sequence(raw_folds)]
    lines = [
        "| Held-out dataset | Held-out group | Primary train pairs | Secondary train pairs | Test pairs | Accuracy | Mean margin | Min margin | Failed pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in folds:
        lines.append(
            "| "
            f"`{fold.get('held_out_dataset', '')}` | "
            f"`{fold.get('held_out_group', '')}` | "
            f"{int(fold.get('train_primary_pairs', 0))} | "
            f"{int(fold.get('train_secondary_pairs', 0))} | "
            f"{int(fold.get('test_pairs', 0))} | "
            f"{float(fold.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(fold.get('mean_margin', 0.0)):+.3f} | "
            f"{float(fold.get('min_margin', 0.0)):+.3f} | "
            f"{int(fold.get('failed_pair_count', 0))} |"
        )
    return lines


def _failed_pair_table(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Held-out dataset | Held-out group | Pair | Margin | Positive projection | Negative projection |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for fold in [
        *(_mapping(item) for item in _sequence(report.get("target_holdout_folds"))),
        *(_mapping(item) for item in _sequence(report.get("source_holdout_folds"))),
    ]:
        for raw_pair in _sequence(fold.get("failed_pairs")):
            pair = _mapping(raw_pair)
            lines.append(
                "| "
                f"`{fold.get('held_out_dataset', '')}` | "
                f"`{fold.get('held_out_group', '')}` | "
                f"`{pair.get('pair_id', '')}` | "
                f"{float(pair.get('margin', 0.0)):+.3f} | "
                f"{float(pair.get('positive_projection', 0.0)):+.3f} | "
                f"{float(pair.get('negative_projection', 0.0)):+.3f} |"
            )
    if len(lines) == 2:
        return ["No failed pairs."]
    return lines


def _ablation_summary_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    lines = [
        "| Bridge groups | Folds | Min accuracy | Min margin | Failed folds | Failed pairs | Ready |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row.get('bridge_group_count', 0))} | "
            f"{int(row.get('folds', 0))} | "
            f"{float(row.get('min_accuracy', 0.0)):.3f} | "
            f"{float(row.get('min_margin', 0.0)):+.3f} | "
            f"{int(row.get('failed_folds', 0))} | "
            f"{int(row.get('failed_pairs', 0))} | "
            f"{bool(row.get('ready', False))} |"
        )
    return lines


def _pair_ablation_summary_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    lines = [
        "| Bridge pairs | Evaluated subsets | Min accuracy | Min margin | Failed subsets | Path coverage | Exhaustive | Ready |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        min_path_count = int(row.get("min_bridge_path_count", 0))
        max_path_count = int(row.get("max_bridge_path_count", 0))
        path_range = (
            str(min_path_count)
            if min_path_count == max_path_count
            else f"{min_path_count}-{max_path_count}"
        )
        lines.append(
            "| "
            f"{int(row.get('bridge_pair_count', 0))} | "
            f"{int(row.get('evaluated_subsets', 0))} | "
            f"{float(row.get('min_accuracy', 0.0)):.3f} | "
            f"{float(row.get('min_margin', 0.0)):+.3f} | "
            f"{int(row.get('failed_subsets', 0))} | "
            f"{path_range} | "
            f"{bool(row.get('exhaustive', False))} | "
            f"{bool(row.get('ready', False))} |"
        )
    return lines


def _failed_ablation_fold_table(report: Mapping[str, Any]) -> list[str]:
    inputs = _mapping(report.get("inputs"))
    min_pairwise_accuracy = float(inputs.get("min_pairwise_accuracy", 1.0))
    min_margin = float(inputs.get("min_margin", 0.0))
    lines = [
        "| Held-out dataset | Held-out group | Bridge groups | Bridge source families | Accuracy | Min margin | Failed pairs |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for fold in [
        *(_mapping(item) for item in _sequence(report.get("target_ablation_folds"))),
        *(_mapping(item) for item in _sequence(report.get("source_ablation_folds"))),
    ]:
        if not _fold_failed(
            fold,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ):
            continue
        lines.append(
            "| "
            f"`{fold.get('held_out_dataset', '')}` | "
            f"`{fold.get('held_out_group', '')}` | "
            f"{int(fold.get('bridge_group_count', 0))} | "
            f"`{', '.join(str(group) for group in _sequence(fold.get('bridge_groups')))}` | "
            f"{float(fold.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(fold.get('min_margin', 0.0)):+.3f} | "
            f"{int(fold.get('failed_pair_count', 0))} |"
        )
    if len(lines) == 2:
        return ["No failed folds."]
    return lines


def _failed_pair_bridge_fold_table(
    report: Mapping[str, Any],
    *,
    limit: int = 80,
) -> list[str]:
    inputs = _mapping(report.get("inputs"))
    min_pairwise_accuracy = float(inputs.get("min_pairwise_accuracy", 1.0))
    min_margin = float(inputs.get("min_margin", 0.0))
    failed_folds = [
        fold
        for fold in [
            *(
                _mapping(item)
                for item in _sequence(report.get("target_pair_ablation_folds"))
            ),
            *(
                _mapping(item)
                for item in _sequence(report.get("source_pair_ablation_folds"))
            ),
        ]
        if _fold_failed(
            fold,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        )
    ]
    if not failed_folds:
        return ["No failed sampled folds."]

    lines = [
        f"Showing first {min(limit, len(failed_folds))} of "
        f"{len(failed_folds)} failed sampled folds.",
        "",
        "| Held-out dataset | Held-out group | Bridge pairs | Subset | Path count | Accuracy | Min margin | Failed pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in failed_folds[:limit]:
        lines.append(
            "| "
            f"`{fold.get('held_out_dataset', '')}` | "
            f"`{fold.get('held_out_group', '')}` | "
            f"{int(fold.get('bridge_pair_count', 0))} | "
            f"{int(fold.get('subset_index', 0))} | "
            f"{int(fold.get('bridge_path_count', 0))} | "
            f"{float(fold.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(fold.get('min_margin', 0.0)):+.3f} | "
            f"{int(fold.get('failed_pair_count', 0))} |"
        )
    return lines


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []
