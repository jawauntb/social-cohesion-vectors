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
_FRESH_BRIDGE_EVALUATION_KEYS = (
    "on_source",
    "on_target",
    "on_fresh_source",
    "on_fresh_target",
)


@dataclass(frozen=True)
class _DomainDataset:
    name: str
    activations: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    pair_groups: Mapping[str, str]
    pair_metadata: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class _BridgeDirectionVectorFold:
    fold_id: str
    fold_kind: str
    held_out_group: str
    direction: np.ndarray
    report: dict[str, Any]


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


def run_bridge_set_sufficiency_audit_from_files(
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
    composition_keys: Sequence[str] = ("source", "primary_fault_class"),
    bridge_pair_count: int = 6,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Evaluate intentionally constructed bridge-pair sets."""

    if bridge_pair_count < 1:
        raise ValueError("bridge_pair_count must be at least 1.")

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

    target_bridge_set_folds = _bridge_set_sufficiency_folds(
        train_primary=source,
        train_secondary=target,
        held_out_dataset=target,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
    )
    source_bridge_set_folds = _bridge_set_sufficiency_folds(
        train_primary=target,
        train_secondary=source,
        held_out_dataset=source,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
    )
    readiness = _bridge_set_readiness(
        source_bridge_set_folds=source_bridge_set_folds,
        target_bridge_set_folds=target_bridge_set_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "bridge_set_sufficiency_audit",
        "description": (
            "Constructs one fixed-size same-domain bridge-pair set per held-out "
            "source family using procedural-path and composition coverage, then "
            "evaluates held-out source-family transfer."
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
            "composition_keys": list(composition_keys),
            "bridge_pair_count": int(bridge_pair_count),
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "source_groups": len(_groups(source)),
            "target_groups": len(_groups(target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": _bridge_set_summary(
            source_bridge_set_folds=source_bridge_set_folds,
            target_bridge_set_folds=target_bridge_set_folds,
            readiness=readiness,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ),
        "readiness": readiness,
        "source_bridge_set_folds": source_bridge_set_folds,
        "target_bridge_set_folds": target_bridge_set_folds,
        "interpretation_guardrail": (
            "A bridge-set sufficiency pass supports a text-benchmark activation "
            "diagnostic for intentionally constructed bridge sets. It does not "
            "establish a human, neural, clinical, deployment, or causal "
            "steering claim."
        ),
    }


def run_bridge_direction_comparison_from_files(
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
    composition_keys: Sequence[str] = ("source", "primary_fault_class"),
    bridge_pair_count: int = 6,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
    min_direction_cosine: float = 0.0,
) -> dict[str, Any]:
    """Compare constructed bridge directions with source-only and joint directions."""

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

    source_direction = _direction_from_training_parts(
        _training_part(source, _unique_pair_ids(source)),
    )
    target_direction = _direction_from_training_parts(
        _training_part(target, _unique_pair_ids(target)),
    )
    joint_direction = _direction_from_training_parts(
        _training_part(source, _unique_pair_ids(source)),
        _training_part(target, _unique_pair_ids(target)),
    )
    target_bridge_direction_folds = _bridge_direction_comparison_folds(
        train_primary=source,
        train_secondary=target,
        held_out_dataset=target,
        source_dataset=source,
        target_dataset=target,
        joint_direction=joint_direction,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
    )
    source_bridge_direction_folds = _bridge_direction_comparison_folds(
        train_primary=target,
        train_secondary=source,
        held_out_dataset=source,
        source_dataset=source,
        target_dataset=target,
        joint_direction=joint_direction,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
    )
    readiness = _bridge_direction_comparison_readiness(
        source_bridge_direction_folds=source_bridge_direction_folds,
        target_bridge_direction_folds=target_bridge_direction_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
        min_direction_cosine=min_direction_cosine,
    )
    source_only_on_source = _evaluate_pairwise_projection(
        source,
        direction=source_direction,
        pair_ids=_unique_pair_ids(source),
    )
    source_only_on_target = _evaluate_pairwise_projection(
        target,
        direction=source_direction,
        pair_ids=_unique_pair_ids(target),
    )
    target_only_on_source = _evaluate_pairwise_projection(
        source,
        direction=target_direction,
        pair_ids=_unique_pair_ids(source),
    )
    target_only_on_target = _evaluate_pairwise_projection(
        target,
        direction=target_direction,
        pair_ids=_unique_pair_ids(target),
    )
    joint_on_source = _evaluate_pairwise_projection(
        source,
        direction=joint_direction,
        pair_ids=_unique_pair_ids(source),
    )
    joint_on_target = _evaluate_pairwise_projection(
        target,
        direction=joint_direction,
        pair_ids=_unique_pair_ids(target),
    )
    return {
        "experiment": "bridge_direction_comparison",
        "description": (
            "Compares source-only, target-only, full joint, and intentionally "
            "constructed bridge-set directions in one activation space."
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
            "composition_keys": list(composition_keys),
            "bridge_pair_count": int(bridge_pair_count),
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
            "min_direction_cosine": float(min_direction_cosine),
        },
        "summary": _bridge_direction_comparison_summary(
            source_bridge_direction_folds=source_bridge_direction_folds,
            target_bridge_direction_folds=target_bridge_direction_folds,
            readiness=readiness,
            source_only_on_target=source_only_on_target,
            target_only_on_source=target_only_on_source,
            joint_on_source=joint_on_source,
            joint_on_target=joint_on_target,
        ),
        "readiness": readiness,
        "source_only_on_source": source_only_on_source,
        "source_only_on_target": source_only_on_target,
        "target_only_on_source": target_only_on_source,
        "target_only_on_target": target_only_on_target,
        "joint_on_source": joint_on_source,
        "joint_on_target": joint_on_target,
        "source_bridge_direction_folds": source_bridge_direction_folds,
        "target_bridge_direction_folds": target_bridge_direction_folds,
        "interpretation_guardrail": (
            "Constructed bridge direction comparison supports a text-benchmark "
            "activation diagnostic. It does not establish a human, neural, "
            "clinical, deployment, or causal steering claim."
        ),
    }


def run_fresh_generated_bridge_diagnostic_from_files(
    *,
    source_activation_npz: str | Path,
    source_pairs_path: str | Path,
    target_activation_npz: str | Path,
    target_pairs_path: str | Path,
    fresh_source_activation_npz: str | Path,
    fresh_source_pairs_path: str | Path,
    fresh_target_activation_npz: str | Path,
    fresh_target_pairs_path: str | Path,
    source_name: str = "source",
    target_name: str = "target",
    fresh_source_name: str = "fresh_source",
    fresh_target_name: str = "fresh_target",
    source_group_key: str = "source",
    target_group_key: str = "source",
    bridge_stratum_key: str = "slack_options_tested",
    composition_keys: Sequence[str] = ("source", "primary_fault_class"),
    bridge_pair_count: int = 6,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Diagnose how fresh generated prompts relate to bridge directions."""

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
    fresh_source = _load_domain_dataset(
        name=fresh_source_name,
        activation_npz=fresh_source_activation_npz,
        pairs_path=fresh_source_pairs_path,
        group_key=source_group_key,
    )
    fresh_target = _load_domain_dataset(
        name=fresh_target_name,
        activation_npz=fresh_target_activation_npz,
        pairs_path=fresh_target_pairs_path,
        group_key=target_group_key,
    )
    _validate_shared_dim(source, target)
    _validate_shared_dim(source, fresh_source)
    _validate_shared_dim(target, fresh_target)

    source_direction = _direction_from_training_parts(
        _training_part(source, _unique_pair_ids(source)),
    )
    fresh_source_direction = _direction_from_training_parts(
        _training_part(fresh_source, _unique_pair_ids(fresh_source)),
    )
    source_fresh_joint_direction = _direction_from_training_parts(
        _training_part(source, _unique_pair_ids(source)),
        _training_part(fresh_source, _unique_pair_ids(fresh_source)),
    )
    original_joint_direction = _direction_from_training_parts(
        _training_part(source, _unique_pair_ids(source)),
        _training_part(target, _unique_pair_ids(target)),
    )
    reference_directions = {
        "source_only": source_direction,
        "fresh_source_only": fresh_source_direction,
        "source_fresh_joint": source_fresh_joint_direction,
        "original_source_target_joint": original_joint_direction,
    }
    target_bridge_folds = _bridge_direction_vector_folds(
        train_primary=source,
        train_secondary=target,
        held_out_dataset=target,
        source_dataset=source,
        target_dataset=target,
        joint_direction=original_joint_direction,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
        fold_kind="target_bridge",
    )
    source_bridge_folds = _bridge_direction_vector_folds(
        train_primary=target,
        train_secondary=source,
        held_out_dataset=source,
        source_dataset=source,
        target_dataset=target,
        joint_direction=original_joint_direction,
        bridge_stratum_key=bridge_stratum_key,
        composition_keys=composition_keys,
        bridge_pair_count=bridge_pair_count,
        fold_kind="source_bridge",
    )
    diagnostic_rows = [
        _fresh_bridge_direction_row(
            direction_id="source_only",
            direction_family="baseline",
            direction=source_direction,
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
            reference_directions=reference_directions,
        ),
        _fresh_bridge_direction_row(
            direction_id="fresh_source_only",
            direction_family="baseline",
            direction=fresh_source_direction,
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
            reference_directions=reference_directions,
        ),
        _fresh_bridge_direction_row(
            direction_id="source_fresh_joint",
            direction_family="baseline",
            direction=source_fresh_joint_direction,
            source=source,
            target=target,
            fresh_source=fresh_source,
            fresh_target=fresh_target,
            reference_directions=reference_directions,
        ),
        *[
            _fresh_bridge_direction_row(
                direction_id=fold.fold_id,
                direction_family="constructed_bridge",
                direction=fold.direction,
                source=source,
                target=target,
                fresh_source=fresh_source,
                fresh_target=fresh_target,
                reference_directions=reference_directions,
                bridge_fold=fold.report,
            )
            for fold in [*target_bridge_folds, *source_bridge_folds]
        ],
    ]
    readiness = _fresh_generated_bridge_readiness(
        diagnostic_rows,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
    return {
        "experiment": "fresh_generated_bridge_diagnostic",
        "description": (
            "Compares source-only, fresh-source-only, source+fresh-source, and "
            "constructed bridge directions inside one activation space, then "
            "evaluates each direction on original and fresh generated/control "
            "prompt slices."
        ),
        "inputs": {
            "source_name": source_name,
            "target_name": target_name,
            "fresh_source_name": fresh_source_name,
            "fresh_target_name": fresh_target_name,
            "source_activation_npz": str(source_activation_npz),
            "source_pairs_path": str(source_pairs_path),
            "target_activation_npz": str(target_activation_npz),
            "target_pairs_path": str(target_pairs_path),
            "fresh_source_activation_npz": str(fresh_source_activation_npz),
            "fresh_source_pairs_path": str(fresh_source_pairs_path),
            "fresh_target_activation_npz": str(fresh_target_activation_npz),
            "fresh_target_pairs_path": str(fresh_target_pairs_path),
            "source_group_key": source_group_key,
            "target_group_key": target_group_key,
            "bridge_stratum_key": bridge_stratum_key,
            "composition_keys": list(composition_keys),
            "bridge_pair_count": int(bridge_pair_count),
            "activation_dim": int(source.activations.shape[1]),
            "source_pairs": len(_unique_pair_ids(source)),
            "target_pairs": len(_unique_pair_ids(target)),
            "fresh_source_pairs": len(_unique_pair_ids(fresh_source)),
            "fresh_target_pairs": len(_unique_pair_ids(fresh_target)),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
        },
        "summary": _fresh_generated_bridge_summary(
            diagnostic_rows,
            readiness=readiness,
        ),
        "readiness": readiness,
        "direction_evaluations": diagnostic_rows,
        "interpretation_guardrail": (
            "Fresh generated bridge diagnostics support only a text-benchmark "
            "activation diagnostic over generated and hand-authored prompts. "
            "They do not establish human behavioral, neural, clinical, "
            "deployment, or real-world social-effect claims."
        ),
    }


def run_cross_model_bridge_transport_from_files(
    *,
    model_a_source_activation_npz: str | Path,
    model_a_source_pairs_path: str | Path,
    model_a_target_activation_npz: str | Path,
    model_a_target_pairs_path: str | Path,
    model_b_source_activation_npz: str | Path,
    model_b_source_pairs_path: str | Path,
    model_b_target_activation_npz: str | Path,
    model_b_target_pairs_path: str | Path,
    model_a_fresh_source_activation_npz: str | Path | None = None,
    model_a_fresh_source_pairs_path: str | Path | None = None,
    model_a_fresh_target_activation_npz: str | Path | None = None,
    model_a_fresh_target_pairs_path: str | Path | None = None,
    model_b_fresh_source_activation_npz: str | Path | None = None,
    model_b_fresh_source_pairs_path: str | Path | None = None,
    model_b_fresh_target_activation_npz: str | Path | None = None,
    model_b_fresh_target_pairs_path: str | Path | None = None,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    source_name: str = "source",
    target_name: str = "target",
    fresh_source_name: str = "fresh_source",
    fresh_target_name: str = "fresh_target",
    source_group_key: str = "source",
    target_group_key: str = "source",
    bridge_stratum_key: str = "slack_options_tested",
    composition_keys: Sequence[str] = ("source", "primary_fault_class"),
    bridge_pair_count: int = 6,
    knn_k: int = 10,
    ridge_alpha: float = 1e-3,
    min_pairwise_accuracy: float = 1.0,
    min_margin: float = 0.0,
    min_mapped_direction_cosine: float = 0.0,
) -> dict[str, Any]:
    """Map constructed bridge directions between two aligned model spaces."""

    if bridge_pair_count < 1:
        raise ValueError("bridge_pair_count must be at least 1.")

    model_a_source = _load_domain_dataset(
        name=f"{model_a_name}_{source_name}",
        activation_npz=model_a_source_activation_npz,
        pairs_path=model_a_source_pairs_path,
        group_key=source_group_key,
    )
    model_a_target = _load_domain_dataset(
        name=f"{model_a_name}_{target_name}",
        activation_npz=model_a_target_activation_npz,
        pairs_path=model_a_target_pairs_path,
        group_key=target_group_key,
    )
    model_b_source = _load_domain_dataset(
        name=f"{model_b_name}_{source_name}",
        activation_npz=model_b_source_activation_npz,
        pairs_path=model_b_source_pairs_path,
        group_key=source_group_key,
    )
    model_b_target = _load_domain_dataset(
        name=f"{model_b_name}_{target_name}",
        activation_npz=model_b_target_activation_npz,
        pairs_path=model_b_target_pairs_path,
        group_key=target_group_key,
    )
    _validate_shared_dim(model_a_source, model_a_target)
    _validate_shared_dim(model_b_source, model_b_target)
    _validate_matching_pair_ids(model_a_source, model_b_source, label=source_name)
    _validate_matching_pair_ids(model_a_target, model_b_target, label=target_name)
    model_a_fresh_source = _load_optional_domain_dataset(
        name=f"{model_a_name}_{fresh_source_name}",
        activation_npz=model_a_fresh_source_activation_npz,
        pairs_path=model_a_fresh_source_pairs_path,
        group_key=source_group_key,
    )
    model_b_fresh_source = _load_optional_domain_dataset(
        name=f"{model_b_name}_{fresh_source_name}",
        activation_npz=model_b_fresh_source_activation_npz,
        pairs_path=model_b_fresh_source_pairs_path,
        group_key=source_group_key,
    )
    model_a_fresh_target = _load_optional_domain_dataset(
        name=f"{model_a_name}_{fresh_target_name}",
        activation_npz=model_a_fresh_target_activation_npz,
        pairs_path=model_a_fresh_target_pairs_path,
        group_key=target_group_key,
    )
    model_b_fresh_target = _load_optional_domain_dataset(
        name=f"{model_b_name}_{fresh_target_name}",
        activation_npz=model_b_fresh_target_activation_npz,
        pairs_path=model_b_fresh_target_pairs_path,
        group_key=target_group_key,
    )
    if model_a_fresh_source is not None and model_b_fresh_source is not None:
        _validate_matching_pair_ids(
            model_a_fresh_source,
            model_b_fresh_source,
            label=fresh_source_name,
        )
        _validate_shared_dim(model_a_source, model_a_fresh_source)
        _validate_shared_dim(model_b_source, model_b_fresh_source)
    if model_a_fresh_target is not None and model_b_fresh_target is not None:
        _validate_matching_pair_ids(
            model_a_fresh_target,
            model_b_fresh_target,
            label=fresh_target_name,
        )
        _validate_shared_dim(model_a_target, model_a_fresh_target)
        _validate_shared_dim(model_b_target, model_b_fresh_target)

    model_a_joint_direction = _direction_from_training_parts(
        _training_part(model_a_source, _unique_pair_ids(model_a_source)),
        _training_part(model_a_target, _unique_pair_ids(model_a_target)),
    )
    model_b_joint_direction = _direction_from_training_parts(
        _training_part(model_b_source, _unique_pair_ids(model_b_source)),
        _training_part(model_b_target, _unique_pair_ids(model_b_target)),
    )
    model_a_folds = [
        *_bridge_direction_vector_folds(
            train_primary=model_a_source,
            train_secondary=model_a_target,
            held_out_dataset=model_a_target,
            source_dataset=model_a_source,
            target_dataset=model_a_target,
            joint_direction=model_a_joint_direction,
            bridge_stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
            bridge_pair_count=bridge_pair_count,
            fold_kind="target_bridge",
        ),
        *_bridge_direction_vector_folds(
            train_primary=model_a_target,
            train_secondary=model_a_source,
            held_out_dataset=model_a_source,
            source_dataset=model_a_source,
            target_dataset=model_a_target,
            joint_direction=model_a_joint_direction,
            bridge_stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
            bridge_pair_count=bridge_pair_count,
            fold_kind="source_bridge",
        ),
    ]
    model_b_folds = [
        *_bridge_direction_vector_folds(
            train_primary=model_b_source,
            train_secondary=model_b_target,
            held_out_dataset=model_b_target,
            source_dataset=model_b_source,
            target_dataset=model_b_target,
            joint_direction=model_b_joint_direction,
            bridge_stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
            bridge_pair_count=bridge_pair_count,
            fold_kind="target_bridge",
        ),
        *_bridge_direction_vector_folds(
            train_primary=model_b_target,
            train_secondary=model_b_source,
            held_out_dataset=model_b_source,
            source_dataset=model_b_source,
            target_dataset=model_b_target,
            joint_direction=model_b_joint_direction,
            bridge_stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
            bridge_pair_count=bridge_pair_count,
            fold_kind="source_bridge",
        ),
    ]
    (
        aligned_a,
        aligned_b,
        aligned_pair_ids,
        alignment_inputs,
    ) = _combined_aligned_activation_matrices(
        model_a_source_activation_npz=model_a_source_activation_npz,
        model_b_source_activation_npz=model_b_source_activation_npz,
        model_a_target_activation_npz=model_a_target_activation_npz,
        model_b_target_activation_npz=model_b_target_activation_npz,
    )
    alignment = _cross_model_alignment_metrics(aligned_a, aligned_b, knn_k=knn_k)
    model_a_to_b = _transport_bridge_direction_folds(
        source_model_name=model_a_name,
        target_model_name=model_b_name,
        source_folds=model_a_folds,
        target_folds=model_b_folds,
        aligned_source_activations=aligned_a,
        aligned_target_activations=aligned_b,
        aligned_pair_ids=aligned_pair_ids,
        ridge_alpha=ridge_alpha,
        target_source_dataset=model_b_source,
        target_target_dataset=model_b_target,
        target_fresh_source_dataset=model_b_fresh_source,
        target_fresh_target_dataset=model_b_fresh_target,
    )
    model_b_to_a = _transport_bridge_direction_folds(
        source_model_name=model_b_name,
        target_model_name=model_a_name,
        source_folds=model_b_folds,
        target_folds=model_a_folds,
        aligned_source_activations=aligned_b,
        aligned_target_activations=aligned_a,
        aligned_pair_ids=aligned_pair_ids,
        ridge_alpha=ridge_alpha,
        target_source_dataset=model_a_source,
        target_target_dataset=model_a_target,
        target_fresh_source_dataset=model_a_fresh_source,
        target_fresh_target_dataset=model_a_fresh_target,
    )
    readiness = _cross_model_bridge_transport_readiness(
        model_a_to_b=model_a_to_b,
        model_b_to_a=model_b_to_a,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
        min_mapped_direction_cosine=min_mapped_direction_cosine,
    )
    return {
        "experiment": "cross_model_bridge_transport_audit",
        "description": (
            "Constructs bridge directions in two aligned model activation spaces, "
            "maps each direction into the other space, and evaluates whether the "
            "mapped direction separates both full benchmarks there."
        ),
        "inputs": {
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "source_name": source_name,
            "target_name": target_name,
            "fresh_source_name": fresh_source_name,
            "fresh_target_name": fresh_target_name,
            "model_a_source_activation_npz": str(model_a_source_activation_npz),
            "model_a_source_pairs_path": str(model_a_source_pairs_path),
            "model_a_target_activation_npz": str(model_a_target_activation_npz),
            "model_a_target_pairs_path": str(model_a_target_pairs_path),
            "model_b_source_activation_npz": str(model_b_source_activation_npz),
            "model_b_source_pairs_path": str(model_b_source_pairs_path),
            "model_b_target_activation_npz": str(model_b_target_activation_npz),
            "model_b_target_pairs_path": str(model_b_target_pairs_path),
            "model_a_fresh_source_activation_npz": _optional_path_string(
                model_a_fresh_source_activation_npz
            ),
            "model_a_fresh_source_pairs_path": _optional_path_string(
                model_a_fresh_source_pairs_path
            ),
            "model_a_fresh_target_activation_npz": _optional_path_string(
                model_a_fresh_target_activation_npz
            ),
            "model_a_fresh_target_pairs_path": _optional_path_string(
                model_a_fresh_target_pairs_path
            ),
            "model_b_fresh_source_activation_npz": _optional_path_string(
                model_b_fresh_source_activation_npz
            ),
            "model_b_fresh_source_pairs_path": _optional_path_string(
                model_b_fresh_source_pairs_path
            ),
            "model_b_fresh_target_activation_npz": _optional_path_string(
                model_b_fresh_target_activation_npz
            ),
            "model_b_fresh_target_pairs_path": _optional_path_string(
                model_b_fresh_target_pairs_path
            ),
            "source_group_key": source_group_key,
            "target_group_key": target_group_key,
            "bridge_stratum_key": bridge_stratum_key,
            "composition_keys": list(composition_keys),
            "bridge_pair_count": int(bridge_pair_count),
            "model_a_dim": int(model_a_source.activations.shape[1]),
            "model_b_dim": int(model_b_source.activations.shape[1]),
            "knn_k": int(knn_k),
            "ridge_alpha": float(ridge_alpha),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "min_margin": float(min_margin),
            "min_mapped_direction_cosine": float(min_mapped_direction_cosine),
            **alignment_inputs,
        },
        "summary": _cross_model_bridge_transport_summary(
            readiness=readiness,
            alignment=alignment,
            model_a_to_b=model_a_to_b,
            model_b_to_a=model_b_to_a,
        ),
        "readiness": readiness,
        "alignment": alignment,
        "model_a_bridge_directions": [fold.report for fold in model_a_folds],
        "model_b_bridge_directions": [fold.report for fold in model_b_folds],
        "model_a_to_b_transport": model_a_to_b,
        "model_b_to_a_transport": model_b_to_a,
        "interpretation_guardrail": (
            "Cross-model bridge transport supports a text-benchmark activation "
            "diagnostic across aligned model spaces. It does not establish a "
            "human, neural, clinical, deployment, or causal steering claim."
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


def save_bridge_set_sufficiency_audit_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown bridge-set sufficiency reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_bridge_set_sufficiency_audit_markdown(report),
        encoding="utf-8",
    )


def save_bridge_direction_comparison_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown bridge direction comparison reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_bridge_direction_comparison_markdown(report),
        encoding="utf-8",
    )


def save_fresh_generated_bridge_diagnostic_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown fresh generated bridge diagnostic reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_fresh_generated_bridge_diagnostic_markdown(report),
        encoding="utf-8",
    )


def save_cross_model_bridge_transport_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown cross-model bridge transport reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_cross_model_bridge_transport_markdown(report),
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


def render_bridge_set_sufficiency_audit_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a bridge-set sufficiency audit as Markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Bridge Set Sufficiency Audit",
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
            f"- Composition keys: "
            f"`{', '.join(str(key) for key in _sequence(inputs.get('composition_keys')))}`",
            f"- Bridge pair count: {int(inputs.get('bridge_pair_count', 0))}",
            f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
            f"- Source pairs/groups: {int(inputs.get('source_pairs', 0))}/"
            f"{int(inputs.get('source_groups', 0))}",
            f"- Target pairs/groups: {int(inputs.get('target_pairs', 0))}/"
            f"{int(inputs.get('target_groups', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for bridge-set claims: "
            f"{bool(summary.get('ready_for_bridge_set_claims', False))}",
            f"- Source min accuracy/margin: "
            f"{float(summary.get('source_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('source_min_margin', 0.0)):+.3f}",
            f"- Target min accuracy/margin: "
            f"{float(summary.get('target_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('target_min_margin', 0.0)):+.3f}",
            f"- Source path-complete folds: "
            f"{int(summary.get('source_path_complete_folds', 0))}/"
            f"{int(summary.get('source_fold_count', 0))}",
            f"- Target path-complete folds: "
            f"{int(summary.get('target_path_complete_folds', 0))}/"
            f"{int(summary.get('target_fold_count', 0))}",
            f"- Failed fold count: {int(summary.get('failed_fold_count', 0))}",
            "",
            "## Target Bridge Sets",
            "",
            *_bridge_set_fold_table(report.get("target_bridge_set_folds")),
            "",
            "## Source Bridge Sets",
            "",
            *_bridge_set_fold_table(report.get("source_bridge_set_folds")),
            "",
            "## Failed Bridge Sets",
            "",
            *_failed_bridge_set_table(report),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def render_bridge_direction_comparison_markdown(report: Mapping[str, Any]) -> str:
    """Render a constructed bridge direction comparison report."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Bridge Direction Comparison",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Source benchmark: `{inputs.get('source_name', '')}`",
            f"- Target benchmark: `{inputs.get('target_name', '')}`",
            f"- Bridge pair count: {int(inputs.get('bridge_pair_count', 0))}",
            f"- Bridge stratum key: `{inputs.get('bridge_stratum_key', '')}`",
            f"- Composition keys: "
            f"`{', '.join(str(key) for key in _sequence(inputs.get('composition_keys')))}`",
            f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for constructed bridge direction claims: "
            f"{bool(summary.get('ready_for_constructed_bridge_direction_claims', False))}",
            f"- Constructed min joint cosine: "
            f"{float(summary.get('constructed_min_joint_cosine', 0.0)):+.3f}",
            f"- Constructed source min accuracy/margin: "
            f"{float(summary.get('constructed_source_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('constructed_source_min_margin', 0.0)):+.3f}",
            f"- Constructed target min accuracy/margin: "
            f"{float(summary.get('constructed_target_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('constructed_target_min_margin', 0.0)):+.3f}",
            f"- Source-only on target accuracy/min margin: "
            f"{float(summary.get('source_only_on_target_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('source_only_on_target_min_margin', 0.0)):+.3f}",
            f"- Target-only on source accuracy/min margin: "
            f"{float(summary.get('target_only_on_source_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('target_only_on_source_min_margin', 0.0)):+.3f}",
            f"- Joint source/target min margins: "
            f"{float(summary.get('joint_on_source_min_margin', 0.0)):+.3f}/"
            f"{float(summary.get('joint_on_target_min_margin', 0.0)):+.3f}",
            "",
            "## Baseline Directions",
            "",
            "| Direction | Evaluation set | Accuracy | Min margin | Failed pairs |",
            "| --- | --- | ---: | ---: | ---: |",
            _compact_eval_row("source-only", "source", report.get("source_only_on_source")),
            _compact_eval_row("source-only", "target", report.get("source_only_on_target")),
            _compact_eval_row("target-only", "source", report.get("target_only_on_source")),
            _compact_eval_row("target-only", "target", report.get("target_only_on_target")),
            _compact_eval_row("joint", "source", report.get("joint_on_source")),
            _compact_eval_row("joint", "target", report.get("joint_on_target")),
            "",
            "## Target Bridge Directions",
            "",
            *_bridge_direction_fold_table(
                report.get("target_bridge_direction_folds")
            ),
            "",
            "## Source Bridge Directions",
            "",
            *_bridge_direction_fold_table(
                report.get("source_bridge_direction_folds")
            ),
            "",
            "## Failed Constructed Directions",
            "",
            *_failed_bridge_direction_table(report),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def render_fresh_generated_bridge_diagnostic_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a fresh generated bridge diagnostic report."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    return "\n".join(
        [
            "# Fresh Generated Bridge Diagnostic",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Source benchmark: `{inputs.get('source_name', '')}`",
            f"- Target benchmark: `{inputs.get('target_name', '')}`",
            f"- Fresh source benchmark: `{inputs.get('fresh_source_name', '')}`",
            f"- Fresh target benchmark: `{inputs.get('fresh_target_name', '')}`",
            f"- Bridge pair count: {int(inputs.get('bridge_pair_count', 0))}",
            f"- Bridge stratum key: `{inputs.get('bridge_stratum_key', '')}`",
            f"- Composition keys: "
            f"`{', '.join(str(key) for key in _sequence(inputs.get('composition_keys')))}`",
            f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for fresh generated bridge claims: "
            f"{bool(summary.get('ready_for_fresh_generated_bridge_claims', False))}",
            f"- Constructed direction count: "
            f"{int(summary.get('constructed_direction_count', 0))}",
            f"- Constructed fresh source min accuracy/margin: "
            f"{float(summary.get('constructed_fresh_source_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('constructed_fresh_source_min_margin', 0.0)):+.3f}",
            f"- Constructed fresh target min accuracy/margin: "
            f"{float(summary.get('constructed_fresh_target_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('constructed_fresh_target_min_margin', 0.0)):+.3f}",
            f"- Source-only fresh source min margin: "
            f"{float(summary.get('source_only_fresh_source_min_margin', 0.0)):+.3f}",
            f"- Fresh-source-only source min margin: "
            f"{float(summary.get('fresh_source_only_source_min_margin', 0.0)):+.3f}",
            f"- Source+fresh-source fresh target min margin: "
            f"{float(summary.get('source_fresh_joint_fresh_target_min_margin', 0.0)):+.3f}",
            f"- Failed pair evaluations: "
            f"{int(summary.get('failed_pair_evaluation_count', 0))}",
            "",
            "## Direction Evaluations",
            "",
            *_fresh_bridge_direction_table(report.get("direction_evaluations")),
            "",
            "## Failed Pairs",
            "",
            *_failed_fresh_bridge_pair_table(report),
            "",
            "## Interpretation Guardrail",
            "",
            str(report.get("interpretation_guardrail", "")),
            "",
        ]
    )


def render_cross_model_bridge_transport_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a cross-model bridge transport report."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    alignment = _mapping(report.get("alignment"))
    return "\n".join(
        [
            "# Cross-Model Bridge Transport Audit",
            "",
            str(report.get("description", "")),
            "",
            "## Inputs",
            "",
            f"- Model A: `{inputs.get('model_a_name', '')}`",
            f"- Model B: `{inputs.get('model_b_name', '')}`",
            f"- Source benchmark: `{inputs.get('source_name', '')}`",
            f"- Target benchmark: `{inputs.get('target_name', '')}`",
            *_fresh_transport_input_lines(inputs),
            f"- Bridge pair count: {int(inputs.get('bridge_pair_count', 0))}",
            f"- Bridge stratum key: `{inputs.get('bridge_stratum_key', '')}`",
            f"- Composition keys: "
            f"`{', '.join(str(key) for key in _sequence(inputs.get('composition_keys')))}`",
            f"- Model dims: {int(inputs.get('model_a_dim', 0))}/"
            f"{int(inputs.get('model_b_dim', 0))}",
            f"- Shared aligned samples: "
            f"{int(inputs.get('shared_aligned_samples', 0))}",
            "",
            "## Alignment",
            "",
            f"- Combined linear CKA: {float(alignment.get('linear_cka', 0.0)):.3f}",
            f"- Combined mutual kNN overlap: "
            f"{float(alignment.get('mutual_knn_overlap', 0.0)):.3f}",
            f"- Source shared samples: "
            f"{int(inputs.get('source_shared_samples', 0))}",
            f"- Target shared samples: "
            f"{int(inputs.get('target_shared_samples', 0))}",
            "",
            "## Summary",
            "",
            f"- Readiness: `{summary.get('readiness', 'not_ready')}`",
            f"- Ready for cross-model bridge transport claims: "
            f"{bool(summary.get('ready_for_cross_model_bridge_transport_claims', False))}",
            f"- Model A -> B min cosine: "
            f"{float(summary.get('model_a_to_b_min_bridge_cosine', 0.0)):+.3f}",
            f"- Model A -> B source min accuracy/margin: "
            f"{float(summary.get('model_a_to_b_source_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_a_to_b_source_min_margin', 0.0)):+.3f}",
            f"- Model A -> B target min accuracy/margin: "
            f"{float(summary.get('model_a_to_b_target_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_a_to_b_target_min_margin', 0.0)):+.3f}",
            f"- Model A -> B leave-held-out min accuracy/margin: "
            f"{float(summary.get('model_a_to_b_leave_heldout_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_a_to_b_leave_heldout_min_margin', 0.0)):+.3f}",
            *_fresh_transport_summary_lines(summary, "model_a_to_b"),
            f"- Model B -> A min cosine: "
            f"{float(summary.get('model_b_to_a_min_bridge_cosine', 0.0)):+.3f}",
            f"- Model B -> A source min accuracy/margin: "
            f"{float(summary.get('model_b_to_a_source_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_b_to_a_source_min_margin', 0.0)):+.3f}",
            f"- Model B -> A target min accuracy/margin: "
            f"{float(summary.get('model_b_to_a_target_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_b_to_a_target_min_margin', 0.0)):+.3f}",
            f"- Model B -> A leave-held-out min accuracy/margin: "
            f"{float(summary.get('model_b_to_a_leave_heldout_min_accuracy', 0.0)):.3f}/"
            f"{float(summary.get('model_b_to_a_leave_heldout_min_margin', 0.0)):+.3f}",
            *_fresh_transport_summary_lines(summary, "model_b_to_a"),
            f"- Failed transported directions: "
            f"{int(summary.get('failed_transported_direction_count', 0))}",
            "",
            "## Model A -> B Transport",
            "",
            *_transport_bridge_direction_table(report.get("model_a_to_b_transport")),
            "",
            "## Model B -> A Transport",
            "",
            *_transport_bridge_direction_table(report.get("model_b_to_a_transport")),
            "",
            "## Failed Transported Directions",
            "",
            *_failed_transport_bridge_direction_table(report),
            "",
            "## Failed Transported Pairs",
            "",
            *_failed_transport_pair_table(report),
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


def _load_optional_domain_dataset(
    *,
    name: str,
    activation_npz: str | Path | None,
    pairs_path: str | Path | None,
    group_key: str,
) -> _DomainDataset | None:
    if activation_npz is None and pairs_path is None:
        return None
    if activation_npz is None or pairs_path is None:
        raise ValueError(
            f"Fresh dataset {name} requires both activation_npz and pairs_path."
        )
    return _load_domain_dataset(
        name=name,
        activation_npz=activation_npz,
        pairs_path=pairs_path,
        group_key=group_key,
    )


def _optional_path_string(path: str | Path | None) -> str | None:
    return str(path) if path is not None else None


def _validate_shared_dim(source: _DomainDataset, target: _DomainDataset) -> None:
    if int(source.activations.shape[1]) != int(target.activations.shape[1]):
        raise ValueError("Source and target activation dimensions must match.")


def _validate_matching_pair_ids(
    left: _DomainDataset,
    right: _DomainDataset,
    *,
    label: str,
) -> None:
    left_pairs = _unique_pair_ids(left)
    right_pairs = _unique_pair_ids(right)
    if left_pairs != right_pairs:
        msg = (
            f"Cross-model {label} pair ids must match: "
            f"left_only={sorted(left_pairs - right_pairs)[:5]} "
            f"right_only={sorted(right_pairs - left_pairs)[:5]}"
        )
        raise ValueError(msg)


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


def _bridge_set_sufficiency_folds(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
    bridge_stratum_key: str,
    composition_keys: Sequence[str],
    bridge_pair_count: int,
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
        if bridge_pair_count > len(bridge_candidates):
            msg = (
                f"Cannot choose {bridge_pair_count} bridge pairs for "
                f"{held_out_group}; only {len(bridge_candidates)} candidates exist."
            )
            raise ValueError(msg)
        bridge_pairs = _construct_bridge_pair_set(
            train_secondary,
            candidate_pairs=bridge_candidates,
            pair_count=bridge_pair_count,
            stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
        )
        bridge_pair_set = set(bridge_pairs)
        train_parts = (
            _training_part(train_primary, _unique_pair_ids(train_primary)),
            _training_part(train_secondary, bridge_pair_set),
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
        bridge_path_values = _bridge_path_values(
            train_secondary,
            bridge_pair_set,
            stratum_key=bridge_stratum_key,
        )
        candidate_path_values = _bridge_path_values(
            train_secondary,
            set(bridge_candidates),
            stratum_key=bridge_stratum_key,
        )
        missing_path_values = sorted(set(candidate_path_values) - set(bridge_path_values))
        folds.append(
            {
                "held_out_dataset": held_out_dataset.name,
                "held_out_group": held_out_group,
                "train_primary_dataset": train_primary.name,
                "train_secondary_dataset": train_secondary.name,
                "train_primary_pairs": len(_unique_pair_ids(train_primary)),
                "candidate_bridge_pairs": len(bridge_candidates),
                "bridge_pair_count": len(bridge_pairs),
                "bridge_pairs": list(bridge_pairs),
                "bridge_strata": _bridge_strata(
                    train_secondary,
                    bridge_pair_set,
                    stratum_key=bridge_stratum_key,
                ),
                "bridge_path_values": bridge_path_values,
                "candidate_path_values": candidate_path_values,
                "missing_path_values": missing_path_values,
                "path_complete": not missing_path_values,
                "coverage_values": _coverage_values(
                    train_secondary,
                    bridge_pair_set,
                    composition_keys=composition_keys,
                ),
                **evaluation,
            }
        )
    return folds


def _bridge_direction_comparison_folds(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
    source_dataset: _DomainDataset,
    target_dataset: _DomainDataset,
    joint_direction: np.ndarray,
    bridge_stratum_key: str,
    composition_keys: Sequence[str],
    bridge_pair_count: int,
) -> list[dict[str, Any]]:
    return [
        fold.report
        for fold in _bridge_direction_vector_folds(
            train_primary=train_primary,
            train_secondary=train_secondary,
            held_out_dataset=held_out_dataset,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            joint_direction=joint_direction,
            bridge_stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
            bridge_pair_count=bridge_pair_count,
            fold_kind=held_out_dataset.name,
        )
    ]


def _bridge_direction_vector_folds(
    *,
    train_primary: _DomainDataset,
    train_secondary: _DomainDataset,
    held_out_dataset: _DomainDataset,
    source_dataset: _DomainDataset,
    target_dataset: _DomainDataset,
    joint_direction: np.ndarray,
    bridge_stratum_key: str,
    composition_keys: Sequence[str],
    bridge_pair_count: int,
    fold_kind: str,
) -> list[_BridgeDirectionVectorFold]:
    folds: list[_BridgeDirectionVectorFold] = []
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
        if bridge_pair_count > len(bridge_candidates):
            msg = (
                f"Cannot choose {bridge_pair_count} bridge pairs for "
                f"{held_out_group}; only {len(bridge_candidates)} candidates exist."
            )
            raise ValueError(msg)
        bridge_pairs = _construct_bridge_pair_set(
            train_secondary,
            candidate_pairs=bridge_candidates,
            pair_count=bridge_pair_count,
            stratum_key=bridge_stratum_key,
            composition_keys=composition_keys,
        )
        bridge_pair_set = set(bridge_pairs)
        direction = _direction_from_training_parts(
            _training_part(train_primary, _unique_pair_ids(train_primary)),
            _training_part(train_secondary, bridge_pair_set),
        )
        bridge_path_values = _bridge_path_values(
            train_secondary,
            bridge_pair_set,
            stratum_key=bridge_stratum_key,
        )
        candidate_path_values = _bridge_path_values(
            train_secondary,
            set(bridge_candidates),
            stratum_key=bridge_stratum_key,
        )
        missing_path_values = sorted(set(candidate_path_values) - set(bridge_path_values))
        fold_id = f"{fold_kind}:{held_out_group}"
        report = {
            "fold_id": fold_id,
            "fold_kind": fold_kind,
            "held_out_dataset": held_out_dataset.name,
            "held_out_group": held_out_group,
            "held_out_pairs": sorted(held_out_pairs),
            "train_primary_dataset": train_primary.name,
            "train_secondary_dataset": train_secondary.name,
            "bridge_pair_count": len(bridge_pairs),
            "bridge_pairs": list(bridge_pairs),
            "bridge_path_values": bridge_path_values,
            "candidate_path_values": candidate_path_values,
            "missing_path_values": missing_path_values,
            "path_complete": not missing_path_values,
            "coverage_values": _coverage_values(
                train_secondary,
                bridge_pair_set,
                composition_keys=composition_keys,
            ),
            "joint_direction_cosine": round(
                _direction_cosine(direction, joint_direction),
                6,
            ),
            "held_out_evaluation": _evaluate_pairwise_projection(
                held_out_dataset,
                direction=direction,
                pair_ids=held_out_pairs,
            ),
            "on_source": _evaluate_pairwise_projection(
                source_dataset,
                direction=direction,
                pair_ids=_unique_pair_ids(source_dataset),
            ),
            "on_target": _evaluate_pairwise_projection(
                target_dataset,
                direction=direction,
                pair_ids=_unique_pair_ids(target_dataset),
            ),
        }
        folds.append(
            _BridgeDirectionVectorFold(
                fold_id=fold_id,
                fold_kind=fold_kind,
                held_out_group=held_out_group,
                direction=direction,
                report=report,
            )
        )
    return folds


def _fresh_bridge_direction_row(
    *,
    direction_id: str,
    direction_family: str,
    direction: np.ndarray,
    source: _DomainDataset,
    target: _DomainDataset,
    fresh_source: _DomainDataset,
    fresh_target: _DomainDataset,
    reference_directions: Mapping[str, np.ndarray],
    bridge_fold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "direction_id": direction_id,
        "direction_family": direction_family,
        "on_source": _evaluate_pairwise_projection(
            source,
            direction=direction,
            pair_ids=_unique_pair_ids(source),
        ),
        "on_target": _evaluate_pairwise_projection(
            target,
            direction=direction,
            pair_ids=_unique_pair_ids(target),
        ),
        "on_fresh_source": _evaluate_pairwise_projection(
            fresh_source,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_source),
        ),
        "on_fresh_target": _evaluate_pairwise_projection(
            fresh_target,
            direction=direction,
            pair_ids=_unique_pair_ids(fresh_target),
        ),
        "cosine_to_source_only": round(
            _direction_cosine(direction, reference_directions["source_only"]),
            6,
        ),
        "cosine_to_fresh_source_only": round(
            _direction_cosine(direction, reference_directions["fresh_source_only"]),
            6,
        ),
        "cosine_to_source_fresh_joint": round(
            _direction_cosine(direction, reference_directions["source_fresh_joint"]),
            6,
        ),
        "cosine_to_original_source_target_joint": round(
            _direction_cosine(
                direction,
                reference_directions["original_source_target_joint"],
            ),
            6,
        ),
    }
    if bridge_fold is not None:
        row["bridge_fold"] = dict(bridge_fold)
    return row


def _combined_aligned_activation_matrices(
    *,
    model_a_source_activation_npz: str | Path,
    model_b_source_activation_npz: str | Path,
    model_a_target_activation_npz: str | Path,
    model_b_target_activation_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_a, source_b, source_pair_ids, source_summary = _aligned_activation_matrices(
        model_a_source_activation_npz,
        model_b_source_activation_npz,
    )
    target_a, target_b, target_pair_ids, target_summary = _aligned_activation_matrices(
        model_a_target_activation_npz,
        model_b_target_activation_npz,
    )
    aligned_a = np.concatenate([source_a, target_a], axis=0)
    aligned_b = np.concatenate([source_b, target_b], axis=0)
    aligned_pair_ids = np.concatenate([source_pair_ids, target_pair_ids], axis=0)
    return (
        aligned_a,
        aligned_b,
        aligned_pair_ids,
        {
            "source_shared_samples": int(source_summary["shared_samples"]),
            "source_shared_pairs": int(source_summary["shared_pairs"]),
            "target_shared_samples": int(target_summary["shared_samples"]),
            "target_shared_pairs": int(target_summary["shared_pairs"]),
            "shared_aligned_samples": int(aligned_a.shape[0]),
        },
    )


def _aligned_activation_matrices(
    left_path: str | Path,
    right_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    left = _load_activation_arrays_with_sample_ids(left_path)
    right = _load_activation_arrays_with_sample_ids(right_path)
    right_by_sample = {
        sample_id: index for index, sample_id in enumerate(right["sample_ids"])
    }
    left_rows: list[int] = []
    right_rows: list[int] = []
    for left_index, sample_id in enumerate(left["sample_ids"]):
        right_index = right_by_sample.get(sample_id)
        if right_index is None:
            continue
        if left["pair_ids"][left_index] != right["pair_ids"][right_index]:
            continue
        if left["labels"][left_index] != right["labels"][right_index]:
            continue
        left_rows.append(left_index)
        right_rows.append(right_index)
    if not left_rows:
        raise ValueError("Cross-model activation payloads have no aligned samples.")
    left_row_array = np.asarray(left_rows, dtype=np.int64)
    aligned_left = left["activations"][left_row_array]
    aligned_right = right["activations"][np.asarray(right_rows, dtype=np.int64)]
    aligned_pair_ids = np.asarray(left["pair_ids"])[left_row_array]
    shared_pairs = {str(pair_id) for pair_id in aligned_pair_ids}
    return (
        aligned_left,
        aligned_right,
        np.asarray(aligned_pair_ids, dtype=str),
        {
            "shared_samples": len(left_rows),
            "shared_pairs": len(shared_pairs),
        },
    )


def _load_activation_arrays_with_sample_ids(path: str | Path) -> dict[str, np.ndarray]:
    activation_path = Path(path)
    with np.load(activation_path, allow_pickle=False) as data:
        activations = np.asarray(data["activations"], dtype=np.float64)
        pair_ids = np.asarray(data["pair_ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=str)
        if "sample_ids" in data:
            sample_ids = np.asarray(data["sample_ids"], dtype=str)
        elif "ids" in data:
            sample_ids = np.asarray(data["ids"], dtype=str)
        else:
            sample_ids = _fallback_sample_ids(pair_ids, labels)
    if activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional matrix.")
    for name, values in (
        ("pair_ids", pair_ids),
        ("labels", labels),
        ("sample_ids", sample_ids),
    ):
        if len(values) != activations.shape[0]:
            raise ValueError(f"{name} length does not match activations.")
    return {
        "activations": activations,
        "pair_ids": pair_ids,
        "labels": labels,
        "sample_ids": sample_ids,
    }


def _fallback_sample_ids(pair_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    sample_ids: list[str] = []
    for pair_id, label in zip(pair_ids.astype(str), labels.astype(str), strict=True):
        key = (pair_id, label)
        index = counts[key]
        counts[key] += 1
        sample_ids.append(f"{pair_id}:{label}:{index}")
    return np.asarray(sample_ids, dtype=str)


def _transport_bridge_direction_folds(
    *,
    source_model_name: str,
    target_model_name: str,
    source_folds: Sequence[_BridgeDirectionVectorFold],
    target_folds: Sequence[_BridgeDirectionVectorFold],
    aligned_source_activations: np.ndarray,
    aligned_target_activations: np.ndarray,
    aligned_pair_ids: np.ndarray,
    ridge_alpha: float,
    target_source_dataset: _DomainDataset,
    target_target_dataset: _DomainDataset,
    target_fresh_source_dataset: _DomainDataset | None,
    target_fresh_target_dataset: _DomainDataset | None,
) -> list[dict[str, Any]]:
    target_by_fold_id = {fold.fold_id: fold for fold in target_folds}
    rows: list[dict[str, Any]] = []
    for fold in source_folds:
        target_fold = target_by_fold_id.get(fold.fold_id)
        if target_fold is None:
            raise ValueError(f"Missing target-model bridge fold {fold.fold_id}.")
        mapped_direction = _ridge_mapped_direction(
            source_activations=aligned_source_activations,
            target_activations=aligned_target_activations,
            source_direction=fold.direction,
            ridge_alpha=ridge_alpha,
        )
        held_out_pair_ids = set(str(pair_id) for pair_id in fold.report["held_out_pairs"])
        leave_out_direction = _leave_held_out_mapped_direction(
            source_direction=fold.direction,
            aligned_source_activations=aligned_source_activations,
            aligned_target_activations=aligned_target_activations,
            aligned_pair_ids=aligned_pair_ids,
            held_out_pair_ids=held_out_pair_ids,
            ridge_alpha=ridge_alpha,
        )
        leave_out_dataset = (
            target_target_dataset
            if fold.fold_kind == "target_bridge"
            else target_source_dataset
        )
        row = {
            "fold_id": fold.fold_id,
            "fold_kind": fold.fold_kind,
            "held_out_group": fold.held_out_group,
            "held_out_pairs": sorted(held_out_pair_ids),
            "source_model": source_model_name,
            "target_model": target_model_name,
            "mapped_direction_cosine_to_target_bridge": round(
                _direction_cosine(mapped_direction, target_fold.direction),
                6,
            ),
            "leave_held_out_map_cosine_to_target_bridge": round(
                _direction_cosine(leave_out_direction, target_fold.direction),
                6,
            ),
            "source_model_on_source": fold.report["on_source"],
            "source_model_on_target": fold.report["on_target"],
            "target_model_bridge_on_source": target_fold.report["on_source"],
            "target_model_bridge_on_target": target_fold.report["on_target"],
            "mapped_on_target_model_source": _evaluate_pairwise_projection(
                target_source_dataset,
                direction=mapped_direction,
                pair_ids=_unique_pair_ids(target_source_dataset),
            ),
            "mapped_on_target_model_target": _evaluate_pairwise_projection(
                target_target_dataset,
                direction=mapped_direction,
                pair_ids=_unique_pair_ids(target_target_dataset),
            ),
            "leave_held_out_map_evaluation": _evaluate_pairwise_projection(
                leave_out_dataset,
                direction=leave_out_direction,
                pair_ids=held_out_pair_ids,
            ),
        }
        if target_fresh_source_dataset is not None:
            row["mapped_on_target_model_fresh_source"] = _evaluate_pairwise_projection(
                target_fresh_source_dataset,
                direction=mapped_direction,
                pair_ids=_unique_pair_ids(target_fresh_source_dataset),
            )
        if target_fresh_target_dataset is not None:
            row["mapped_on_target_model_fresh_target"] = _evaluate_pairwise_projection(
                target_fresh_target_dataset,
                direction=mapped_direction,
                pair_ids=_unique_pair_ids(target_fresh_target_dataset),
            )
        rows.append(row)
    return rows


def _leave_held_out_mapped_direction(
    *,
    source_direction: np.ndarray,
    aligned_source_activations: np.ndarray,
    aligned_target_activations: np.ndarray,
    aligned_pair_ids: np.ndarray,
    held_out_pair_ids: set[str],
    ridge_alpha: float,
) -> np.ndarray:
    train_mask = np.asarray(
        [str(pair_id) not in held_out_pair_ids for pair_id in aligned_pair_ids],
        dtype=bool,
    )
    if int(train_mask.sum()) < 2:
        return np.zeros(aligned_target_activations.shape[1], dtype=np.float64)
    return _ridge_mapped_direction(
        source_activations=aligned_source_activations[train_mask],
        target_activations=aligned_target_activations[train_mask],
        source_direction=source_direction,
        ridge_alpha=ridge_alpha,
    )


def _training_part(
    dataset: _DomainDataset,
    pair_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    mask = _prompt_mask(dataset, pair_ids)
    return dataset.activations[mask], dataset.labels[mask]


def _direction_from_training_parts(
    *parts: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    train_activations = np.concatenate([part[0] for part in parts], axis=0)
    train_labels = np.concatenate([part[1] for part in parts], axis=0)
    return train_direction_from_arrays(train_activations, labels=train_labels).direction


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


def _bridge_set_readiness(
    *,
    source_bridge_set_folds: Sequence[Mapping[str, Any]],
    target_bridge_set_folds: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    all_folds = [*source_bridge_set_folds, *target_bridge_set_folds]
    failed_folds = _failed_fold_count(
        all_folds,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_margin=min_margin,
    )
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
        _gate("failed_fold_count", -float(failed_folds), 0.0),
        _gate(
            "path_complete_folds",
            float(sum(bool(fold.get("path_complete", False)) for fold in all_folds)),
            float(len(all_folds)),
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "bridge_set_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _bridge_set_summary(
    *,
    source_bridge_set_folds: Sequence[Mapping[str, Any]],
    target_bridge_set_folds: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    all_folds = [*source_bridge_set_folds, *target_bridge_set_folds]
    return {
        "readiness": str(readiness.get("status", "not_ready")),
        "ready_for_bridge_set_claims": bool(readiness.get("ready", False)),
        "source_fold_count": len(source_bridge_set_folds),
        "target_fold_count": len(target_bridge_set_folds),
        "source_min_accuracy": min(
            _fold_values(source_bridge_set_folds, "pairwise_accuracy"),
            default=0.0,
        ),
        "source_min_margin": min(
            _fold_values(source_bridge_set_folds, "min_margin"),
            default=0.0,
        ),
        "target_min_accuracy": min(
            _fold_values(target_bridge_set_folds, "pairwise_accuracy"),
            default=0.0,
        ),
        "target_min_margin": min(
            _fold_values(target_bridge_set_folds, "min_margin"),
            default=0.0,
        ),
        "source_path_complete_folds": sum(
            bool(fold.get("path_complete", False)) for fold in source_bridge_set_folds
        ),
        "target_path_complete_folds": sum(
            bool(fold.get("path_complete", False)) for fold in target_bridge_set_folds
        ),
        "failed_fold_count": _failed_fold_count(
            all_folds,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ),
        "failed_pair_count": sum(
            int(fold.get("failed_pair_count", 0)) for fold in all_folds
        ),
    }


def _bridge_direction_comparison_readiness(
    *,
    source_bridge_direction_folds: Sequence[Mapping[str, Any]],
    target_bridge_direction_folds: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
    min_direction_cosine: float,
) -> dict[str, Any]:
    all_folds = [*source_bridge_direction_folds, *target_bridge_direction_folds]
    source_evaluations = [_mapping(fold.get("on_source")) for fold in all_folds]
    target_evaluations = [_mapping(fold.get("on_target")) for fold in all_folds]
    gates = [
        _gate("folds_present", float(len(all_folds)), 1.0),
        _gate(
            "constructed_source_min_accuracy",
            min(_fold_values(source_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            "constructed_source_min_margin",
            min(_fold_values(source_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            "constructed_target_min_accuracy",
            min(_fold_values(target_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            "constructed_target_min_margin",
            min(_fold_values(target_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            "constructed_min_joint_cosine",
            min(_fold_values(all_folds, "joint_direction_cosine"), default=0.0),
            min_direction_cosine,
        ),
        _gate(
            "path_complete_folds",
            float(sum(bool(fold.get("path_complete", False)) for fold in all_folds)),
            float(len(all_folds)),
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "constructed_bridge_direction_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _bridge_direction_comparison_summary(
    *,
    source_bridge_direction_folds: Sequence[Mapping[str, Any]],
    target_bridge_direction_folds: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    source_only_on_target: Mapping[str, Any],
    target_only_on_source: Mapping[str, Any],
    joint_on_source: Mapping[str, Any],
    joint_on_target: Mapping[str, Any],
) -> dict[str, Any]:
    all_folds = [*source_bridge_direction_folds, *target_bridge_direction_folds]
    source_evaluations = [_mapping(fold.get("on_source")) for fold in all_folds]
    target_evaluations = [_mapping(fold.get("on_target")) for fold in all_folds]
    return {
        "readiness": str(readiness.get("status", "not_ready")),
        "ready_for_constructed_bridge_direction_claims": bool(
            readiness.get("ready", False)
        ),
        "constructed_direction_count": len(all_folds),
        "constructed_min_joint_cosine": min(
            _fold_values(all_folds, "joint_direction_cosine"),
            default=0.0,
        ),
        "constructed_source_min_accuracy": min(
            _fold_values(source_evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        "constructed_source_min_margin": min(
            _fold_values(source_evaluations, "min_margin"),
            default=0.0,
        ),
        "constructed_target_min_accuracy": min(
            _fold_values(target_evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        "constructed_target_min_margin": min(
            _fold_values(target_evaluations, "min_margin"),
            default=0.0,
        ),
        "source_only_on_target_accuracy": float(
            source_only_on_target.get("pairwise_accuracy", 0.0)
        ),
        "source_only_on_target_min_margin": float(
            source_only_on_target.get("min_margin", 0.0)
        ),
        "source_only_on_target_failed_pairs": int(
            source_only_on_target.get("failed_pair_count", 0)
        ),
        "target_only_on_source_accuracy": float(
            target_only_on_source.get("pairwise_accuracy", 0.0)
        ),
        "target_only_on_source_min_margin": float(
            target_only_on_source.get("min_margin", 0.0)
        ),
        "target_only_on_source_failed_pairs": int(
            target_only_on_source.get("failed_pair_count", 0)
        ),
        "joint_on_source_accuracy": float(joint_on_source.get("pairwise_accuracy", 0.0)),
        "joint_on_source_min_margin": float(joint_on_source.get("min_margin", 0.0)),
        "joint_on_target_accuracy": float(joint_on_target.get("pairwise_accuracy", 0.0)),
        "joint_on_target_min_margin": float(joint_on_target.get("min_margin", 0.0)),
    }


def _fresh_generated_bridge_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> dict[str, Any]:
    constructed_rows = _direction_rows_by_family(rows, "constructed_bridge")
    fresh_source_evaluations = [
        _mapping(row.get("on_fresh_source")) for row in constructed_rows
    ]
    fresh_target_evaluations = [
        _mapping(row.get("on_fresh_target")) for row in constructed_rows
    ]
    gates = [
        _gate("source_only_direction_present", _direction_present(rows, "source_only"), 1.0),
        _gate(
            "fresh_source_only_direction_present",
            _direction_present(rows, "fresh_source_only"),
            1.0,
        ),
        _gate(
            "source_fresh_joint_direction_present",
            _direction_present(rows, "source_fresh_joint"),
            1.0,
        ),
        _gate("constructed_direction_count", float(len(constructed_rows)), 1.0),
        _gate(
            "constructed_fresh_source_min_accuracy",
            min(_fold_values(fresh_source_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            "constructed_fresh_source_min_margin",
            min(_fold_values(fresh_source_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            "constructed_fresh_target_min_accuracy",
            min(_fold_values(fresh_target_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            "constructed_fresh_target_min_margin",
            min(_fold_values(fresh_target_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "fresh_generated_bridge_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _fresh_generated_bridge_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    constructed_rows = _direction_rows_by_family(rows, "constructed_bridge")
    return {
        "readiness": str(readiness.get("status", "not_ready")),
        "ready_for_fresh_generated_bridge_claims": bool(
            readiness.get("ready", False)
        ),
        "direction_count": len(rows),
        "constructed_direction_count": len(constructed_rows),
        "constructed_fresh_source_min_accuracy": _min_eval_metric(
            constructed_rows,
            "on_fresh_source",
            "pairwise_accuracy",
        ),
        "constructed_fresh_source_min_margin": _min_eval_metric(
            constructed_rows,
            "on_fresh_source",
            "min_margin",
        ),
        "constructed_fresh_source_failed_pairs": _sum_eval_metric(
            constructed_rows,
            "on_fresh_source",
            "failed_pair_count",
        ),
        "constructed_fresh_target_min_accuracy": _min_eval_metric(
            constructed_rows,
            "on_fresh_target",
            "pairwise_accuracy",
        ),
        "constructed_fresh_target_min_margin": _min_eval_metric(
            constructed_rows,
            "on_fresh_target",
            "min_margin",
        ),
        "constructed_fresh_target_failed_pairs": _sum_eval_metric(
            constructed_rows,
            "on_fresh_target",
            "failed_pair_count",
        ),
        "source_only_fresh_source_min_margin": _direction_eval_metric(
            rows,
            "source_only",
            "on_fresh_source",
            "min_margin",
        ),
        "fresh_source_only_source_min_margin": _direction_eval_metric(
            rows,
            "fresh_source_only",
            "on_source",
            "min_margin",
        ),
        "source_fresh_joint_fresh_source_min_margin": _direction_eval_metric(
            rows,
            "source_fresh_joint",
            "on_fresh_source",
            "min_margin",
        ),
        "source_fresh_joint_fresh_target_min_margin": _direction_eval_metric(
            rows,
            "source_fresh_joint",
            "on_fresh_target",
            "min_margin",
        ),
        "failed_pair_evaluation_count": _fresh_bridge_failed_pair_count(rows),
    }


def _cross_model_bridge_transport_readiness(
    *,
    model_a_to_b: Sequence[Mapping[str, Any]],
    model_b_to_a: Sequence[Mapping[str, Any]],
    min_pairwise_accuracy: float,
    min_margin: float,
    min_mapped_direction_cosine: float,
) -> dict[str, Any]:
    gates = [
        _gate("model_a_to_b_folds_present", float(len(model_a_to_b)), 1.0),
        _gate("model_b_to_a_folds_present", float(len(model_b_to_a)), 1.0),
        *_transport_readiness_gates(
            "model_a_to_b",
            model_a_to_b,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
            min_mapped_direction_cosine=min_mapped_direction_cosine,
        ),
        *_transport_readiness_gates(
            "model_b_to_a",
            model_b_to_a,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
            min_mapped_direction_cosine=min_mapped_direction_cosine,
        ),
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "cross_model_bridge_transport_ready" if ready else "not_ready",
        "ready": ready,
        "gates": gates,
    }


def _transport_readiness_gates(
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
    min_mapped_direction_cosine: float,
) -> list[dict[str, Any]]:
    source_evaluations = [
        _mapping(row.get("mapped_on_target_model_source")) for row in rows
    ]
    target_evaluations = [
        _mapping(row.get("mapped_on_target_model_target")) for row in rows
    ]
    leave_held_out_evaluations = [
        _mapping(row.get("leave_held_out_map_evaluation")) for row in rows
    ]
    gates = [
        _gate(
            f"{prefix}_source_min_accuracy",
            min(_fold_values(source_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            f"{prefix}_source_min_margin",
            min(_fold_values(source_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            f"{prefix}_target_min_accuracy",
            min(_fold_values(target_evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            f"{prefix}_target_min_margin",
            min(_fold_values(target_evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
        _gate(
            f"{prefix}_min_bridge_cosine",
            min(
                _fold_values(rows, "mapped_direction_cosine_to_target_bridge"),
                default=0.0,
            ),
            min_mapped_direction_cosine,
        ),
        _gate(
            f"{prefix}_leave_heldout_min_accuracy",
            min(
                _fold_values(leave_held_out_evaluations, "pairwise_accuracy"),
                default=0.0,
            ),
            min_pairwise_accuracy,
        ),
        _gate(
            f"{prefix}_leave_heldout_min_margin",
            min(
                _fold_values(leave_held_out_evaluations, "min_margin"),
                default=0.0,
            ),
            min_margin,
        ),
        _gate(
            f"{prefix}_leave_heldout_min_bridge_cosine",
            min(
                _fold_values(rows, "leave_held_out_map_cosine_to_target_bridge"),
                default=0.0,
            ),
            min_mapped_direction_cosine,
        ),
    ]
    gates.extend(
        _optional_transport_eval_gates(
            prefix,
            "fresh_source",
            rows,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        )
    )
    gates.extend(
        _optional_transport_eval_gates(
            prefix,
            "fresh_target",
            rows,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        )
    )
    return gates


def _optional_transport_eval_gates(
    prefix: str,
    suffix: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    key = f"mapped_on_target_model_{suffix}"
    evaluations = [_mapping(row.get(key)) for row in rows if row.get(key) is not None]
    if not evaluations:
        return []
    return [
        _gate(
            f"{prefix}_{suffix}_min_accuracy",
            min(_fold_values(evaluations, "pairwise_accuracy"), default=0.0),
            min_pairwise_accuracy,
        ),
        _gate(
            f"{prefix}_{suffix}_min_margin",
            min(_fold_values(evaluations, "min_margin"), default=0.0),
            min_margin,
        ),
    ]


def _cross_model_bridge_transport_summary(
    *,
    readiness: Mapping[str, Any],
    alignment: Mapping[str, Any],
    model_a_to_b: Sequence[Mapping[str, Any]],
    model_b_to_a: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "readiness": str(readiness.get("status", "not_ready")),
        "ready_for_cross_model_bridge_transport_claims": bool(
            readiness.get("ready", False)
        ),
        "linear_cka": float(alignment.get("linear_cka", 0.0)),
        "mutual_knn_overlap": float(alignment.get("mutual_knn_overlap", 0.0)),
        "model_a_to_b_direction_count": len(model_a_to_b),
        "model_b_to_a_direction_count": len(model_b_to_a),
        **_transport_direction_summary("model_a_to_b", model_a_to_b),
        **_transport_direction_summary("model_b_to_a", model_b_to_a),
        "failed_transported_direction_count": (
            _failed_transport_direction_count(model_a_to_b)
            + _failed_transport_direction_count(model_b_to_a)
        ),
    }


def _transport_direction_summary(
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_evaluations = [
        _mapping(row.get("mapped_on_target_model_source")) for row in rows
    ]
    target_evaluations = [
        _mapping(row.get("mapped_on_target_model_target")) for row in rows
    ]
    leave_held_out_evaluations = [
        _mapping(row.get("leave_held_out_map_evaluation")) for row in rows
    ]
    summary = {
        f"{prefix}_min_bridge_cosine": min(
            _fold_values(rows, "mapped_direction_cosine_to_target_bridge"),
            default=0.0,
        ),
        f"{prefix}_leave_heldout_min_bridge_cosine": min(
            _fold_values(rows, "leave_held_out_map_cosine_to_target_bridge"),
            default=0.0,
        ),
        f"{prefix}_source_min_accuracy": min(
            _fold_values(source_evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        f"{prefix}_source_min_margin": min(
            _fold_values(source_evaluations, "min_margin"),
            default=0.0,
        ),
        f"{prefix}_target_min_accuracy": min(
            _fold_values(target_evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        f"{prefix}_target_min_margin": min(
            _fold_values(target_evaluations, "min_margin"),
            default=0.0,
        ),
        f"{prefix}_leave_heldout_min_accuracy": min(
            _fold_values(leave_held_out_evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        f"{prefix}_leave_heldout_min_margin": min(
            _fold_values(leave_held_out_evaluations, "min_margin"),
            default=0.0,
        ),
        f"{prefix}_failed_direction_count": _failed_transport_direction_count(rows),
    }
    summary.update(_optional_transport_eval_summary(prefix, "fresh_source", rows))
    summary.update(_optional_transport_eval_summary(prefix, "fresh_target", rows))
    return summary


def _optional_transport_eval_summary(
    prefix: str,
    suffix: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    key = f"mapped_on_target_model_{suffix}"
    evaluations = [_mapping(row.get(key)) for row in rows if row.get(key) is not None]
    if not evaluations:
        return {}
    return {
        f"{prefix}_{suffix}_min_accuracy": min(
            _fold_values(evaluations, "pairwise_accuracy"),
            default=0.0,
        ),
        f"{prefix}_{suffix}_min_margin": min(
            _fold_values(evaluations, "min_margin"),
            default=0.0,
        ),
        f"{prefix}_{suffix}_failed_pair_count": sum(
            int(evaluation.get("failed_pair_count", 0)) for evaluation in evaluations
        ),
    }


def _failed_transport_direction_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if int(
            _mapping(row.get("mapped_on_target_model_source")).get(
                "failed_pair_count",
                0,
            )
        )
        or int(
            _mapping(row.get("mapped_on_target_model_target")).get(
                "failed_pair_count",
                0,
            )
        )
        or int(
            _mapping(row.get("leave_held_out_map_evaluation")).get(
                "failed_pair_count",
                0,
            )
        )
        or int(
            _mapping(row.get("mapped_on_target_model_fresh_source")).get(
                "failed_pair_count",
                0,
            )
        )
        or int(
            _mapping(row.get("mapped_on_target_model_fresh_target")).get(
                "failed_pair_count",
                0,
            )
        )
        or float(row.get("mapped_direction_cosine_to_target_bridge", 0.0)) < 0.0
        or float(row.get("leave_held_out_map_cosine_to_target_bridge", 0.0)) < 0.0
    )


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


def _direction_rows_by_family(
    rows: Sequence[Mapping[str, Any]],
    family: str,
) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("direction_family", "")) == family]


def _direction_present(rows: Sequence[Mapping[str, Any]], direction_id: str) -> float:
    return (
        1.0
        if any(str(row.get("direction_id", "")) == direction_id for row in rows)
        else 0.0
    )


def _min_eval_metric(
    rows: Sequence[Mapping[str, Any]],
    evaluation_key: str,
    metric_key: str,
) -> float:
    evaluations = [_mapping(row.get(evaluation_key)) for row in rows]
    return min(_fold_values(evaluations, metric_key), default=0.0)


def _sum_eval_metric(
    rows: Sequence[Mapping[str, Any]],
    evaluation_key: str,
    metric_key: str,
) -> int:
    return sum(int(_mapping(row.get(evaluation_key)).get(metric_key, 0)) for row in rows)


def _direction_eval_metric(
    rows: Sequence[Mapping[str, Any]],
    direction_id: str,
    evaluation_key: str,
    metric_key: str,
) -> float:
    for row in rows:
        if str(row.get("direction_id", "")) == direction_id:
            return float(_mapping(row.get(evaluation_key)).get(metric_key, 0.0))
    return 0.0


def _fresh_bridge_failed_pair_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        _sum_eval_metric([row], evaluation_key, "failed_pair_count")
        for row in rows
        for evaluation_key in _FRESH_BRIDGE_EVALUATION_KEYS
    )


def _direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _ridge_mapped_direction(
    *,
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    source_direction: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    source_centered = _center_columns(source_activations)
    target_centered = _center_columns(target_activations)
    kernel = source_centered @ source_centered.T
    scale = max(
        float(np.trace(kernel)) / max(1, source_centered.shape[1]),
        _EPSILON,
    )
    penalty = float(ridge_alpha) * scale
    system = kernel + penalty * np.eye(kernel.shape[0], dtype=np.float64)
    rhs = source_centered @ source_direction
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(system) @ rhs
    return _normalize_vector(coefficients @ target_centered)


def _cross_model_alignment_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    knn_k: int,
) -> dict[str, Any]:
    effective_k = max(0, min(int(knn_k), source.shape[0] - 1))
    return {
        "linear_cka": round(_linear_cka(source, target), 6),
        "mutual_knn_overlap": round(
            _mutual_knn_overlap(source, target, knn_k=effective_k),
            6,
        ),
        "effective_knn_k": effective_k,
    }


def _linear_cka(source: np.ndarray, target: np.ndarray) -> float:
    source_centered = _center_columns(source)
    target_centered = _center_columns(target)
    source_gram = source_centered @ source_centered.T
    target_gram = target_centered @ target_centered.T
    source_energy = float(np.sum(source_gram**2))
    target_energy = float(np.sum(target_gram**2))
    denominator = np.sqrt(source_energy * target_energy)
    if denominator <= _EPSILON:
        return 0.0
    return float(np.sum(source_gram * target_gram) / denominator)


def _mutual_knn_overlap(source: np.ndarray, target: np.ndarray, *, knn_k: int) -> float:
    if knn_k <= 0:
        return 0.0
    source_neighbors = _knn_indices(source, knn_k)
    target_neighbors = _knn_indices(target, knn_k)
    overlaps = [
        len(set(source_row.tolist()) & set(target_row.tolist())) / knn_k
        for source_row, target_row in zip(source_neighbors, target_neighbors, strict=True)
    ]
    return float(np.mean(overlaps)) if overlaps else 0.0


def _knn_indices(matrix: np.ndarray, knn_k: int) -> np.ndarray:
    normalized = _row_normalize(_center_columns(matrix))
    similarity = np.asarray(normalized @ normalized.T, dtype=np.float64)
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1)[:, :knn_k]


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _EPSILON)


def _center_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.mean(matrix, axis=0, keepdims=True)


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


def _construct_bridge_pair_set(
    dataset: _DomainDataset,
    *,
    candidate_pairs: Sequence[str],
    pair_count: int,
    stratum_key: str,
    composition_keys: Sequence[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    remaining = sorted(candidate_pairs)
    covered_paths: set[str] = set()
    covered_values: dict[str, set[str]] = {key: set() for key in composition_keys}
    while remaining and len(selected) < pair_count:
        best_pair = max(
            remaining,
            key=lambda pair_id: _bridge_pair_selection_score(
                dataset,
                pair_id=pair_id,
                selected_pairs=selected,
                covered_paths=covered_paths,
                covered_values=covered_values,
                stratum_key=stratum_key,
                composition_keys=composition_keys,
            ),
        )
        selected.append(best_pair)
        remaining.remove(best_pair)
        covered_paths.update(
            _bridge_path_values(dataset, {best_pair}, stratum_key=stratum_key)
        )
        for key in composition_keys:
            covered_values[key].update(_metadata_values(dataset, best_pair, key=key))
    return tuple(selected)


def _bridge_pair_selection_score(
    dataset: _DomainDataset,
    *,
    pair_id: str,
    selected_pairs: Sequence[str],
    covered_paths: set[str],
    covered_values: Mapping[str, set[str]],
    stratum_key: str,
    composition_keys: Sequence[str],
) -> tuple[int, int, int, int, str]:
    paths = set(_bridge_path_values(dataset, {pair_id}, stratum_key=stratum_key))
    new_paths = len(paths - covered_paths)
    new_coverage_values = 0
    redundancy = 0
    for key in composition_keys:
        values = set(_metadata_values(dataset, pair_id, key=key))
        new_coverage_values += len(values - covered_values.get(key, set()))
        for selected_pair in selected_pairs:
            selected_values = set(_metadata_values(dataset, selected_pair, key=key))
            if values & selected_values:
                redundancy += 1
    return (
        new_paths,
        new_coverage_values,
        len(paths),
        -redundancy,
        pair_id,
    )


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


def _metadata_values(
    dataset: _DomainDataset,
    pair_id: str,
    *,
    key: str,
) -> list[str]:
    raw_value = dataset.pair_metadata.get(pair_id, {}).get(key, "")
    return _split_metadata_values(raw_value)


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


def _coverage_values(
    dataset: _DomainDataset,
    pair_ids: set[str],
    *,
    composition_keys: Sequence[str],
) -> dict[str, list[str]]:
    return {
        key: sorted(
            {
                value
                for pair_id in pair_ids
                for value in _metadata_values(dataset, pair_id, key=key)
            }
        )
        for key in composition_keys
    }


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


def _bridge_set_fold_table(raw_folds: object) -> list[str]:
    folds = [_mapping(fold) for fold in _sequence(raw_folds)]
    lines = [
        "| Held-out group | Bridge pairs | Accuracy | Min margin | Path complete | Missing paths | Coverage values |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for fold in folds:
        lines.append(
            "| "
            f"`{fold.get('held_out_group', '')}` | "
            f"{int(fold.get('bridge_pair_count', 0))} | "
            f"{float(fold.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(fold.get('min_margin', 0.0)):+.3f} | "
            f"{bool(fold.get('path_complete', False))} | "
            f"`{', '.join(str(path) for path in _sequence(fold.get('missing_path_values')))}` | "
            f"`{_coverage_summary(fold.get('coverage_values'))}` |"
        )
    return lines


def _failed_bridge_set_table(report: Mapping[str, Any]) -> list[str]:
    inputs = _mapping(report.get("inputs"))
    min_pairwise_accuracy = float(inputs.get("min_pairwise_accuracy", 1.0))
    min_margin = float(inputs.get("min_margin", 0.0))
    lines = [
        "| Held-out dataset | Held-out group | Accuracy | Min margin | Failed pairs | Missing paths |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for fold in [
        *(_mapping(item) for item in _sequence(report.get("target_bridge_set_folds"))),
        *(_mapping(item) for item in _sequence(report.get("source_bridge_set_folds"))),
    ]:
        if (
            not _fold_failed(
                fold,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
            )
            and bool(fold.get("path_complete", False))
        ):
            continue
        lines.append(
            "| "
            f"`{fold.get('held_out_dataset', '')}` | "
            f"`{fold.get('held_out_group', '')}` | "
            f"{float(fold.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(fold.get('min_margin', 0.0)):+.3f} | "
            f"{int(fold.get('failed_pair_count', 0))} | "
            f"`{', '.join(str(path) for path in _sequence(fold.get('missing_path_values')))}` |"
        )
    if len(lines) == 2:
        return ["No failed bridge sets."]
    return lines


def _compact_eval_row(
    direction_name: str,
    evaluation_set: str,
    raw_evaluation: object,
) -> str:
    evaluation = _mapping(raw_evaluation)
    return (
        "| "
        f"`{direction_name}` | "
        f"`{evaluation_set}` | "
        f"{float(evaluation.get('pairwise_accuracy', 0.0)):.3f} | "
        f"{float(evaluation.get('min_margin', 0.0)):+.3f} | "
        f"{int(evaluation.get('failed_pair_count', 0))} |"
    )


def _fresh_transport_summary_lines(
    summary: Mapping[str, Any],
    prefix: str,
) -> list[str]:
    lines: list[str] = []
    for label, key_suffix in (
        ("fresh source", "fresh_source"),
        ("fresh target", "fresh_target"),
    ):
        accuracy_key = f"{prefix}_{key_suffix}_min_accuracy"
        margin_key = f"{prefix}_{key_suffix}_min_margin"
        if accuracy_key not in summary or margin_key not in summary:
            continue
        lines.append(
            f"- {_transport_direction_label(prefix)} {label} min accuracy/margin: "
            f"{float(summary.get(accuracy_key, 0.0)):.3f}/"
            f"{float(summary.get(margin_key, 0.0)):+.3f}"
        )
    return lines


def _fresh_transport_input_lines(inputs: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if inputs.get("model_a_fresh_source_activation_npz") or inputs.get(
        "model_b_fresh_source_activation_npz"
    ):
        lines.append(f"- Fresh source benchmark: `{inputs.get('fresh_source_name', '')}`")
    if inputs.get("model_a_fresh_target_activation_npz") or inputs.get(
        "model_b_fresh_target_activation_npz"
    ):
        lines.append(f"- Fresh target benchmark: `{inputs.get('fresh_target_name', '')}`")
    return lines


def _transport_direction_label(prefix: str) -> str:
    if prefix == "model_a_to_b":
        return "Model A -> B"
    if prefix == "model_b_to_a":
        return "Model B -> A"
    return prefix.replace("_", " ").title()


def _bridge_direction_fold_table(raw_folds: object) -> list[str]:
    folds = [_mapping(fold) for fold in _sequence(raw_folds)]
    lines = [
        "| Held-out group | Joint cosine | Source accuracy | Source min margin | Target accuracy | Target min margin | Path complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for fold in folds:
        source_eval = _mapping(fold.get("on_source"))
        target_eval = _mapping(fold.get("on_target"))
        lines.append(
            "| "
            f"`{fold.get('held_out_group', '')}` | "
            f"{float(fold.get('joint_direction_cosine', 0.0)):+.3f} | "
            f"{float(source_eval.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(source_eval.get('min_margin', 0.0)):+.3f} | "
            f"{float(target_eval.get('pairwise_accuracy', 0.0)):.3f} | "
            f"{float(target_eval.get('min_margin', 0.0)):+.3f} | "
            f"{bool(fold.get('path_complete', False))} |"
        )
    return lines


def _fresh_bridge_direction_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    lines = [
        "| Direction | Family | Source | Target | Fresh source | Fresh target | Cos source | Cos fresh | Cos source+fresh | Cos original joint |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('direction_id', '')}` | "
            f"`{row.get('direction_family', '')}` | "
            f"{_compact_eval_metric(row.get('on_source'))} | "
            f"{_compact_eval_metric(row.get('on_target'))} | "
            f"{_compact_eval_metric(row.get('on_fresh_source'))} | "
            f"{_compact_eval_metric(row.get('on_fresh_target'))} | "
            f"{float(row.get('cosine_to_source_only', 0.0)):+.3f} | "
            f"{float(row.get('cosine_to_fresh_source_only', 0.0)):+.3f} | "
            f"{float(row.get('cosine_to_source_fresh_joint', 0.0)):+.3f} | "
            f"{float(row.get('cosine_to_original_source_target_joint', 0.0)):+.3f} |"
        )
    return lines


def _compact_eval_metric(raw_evaluation: object) -> str:
    evaluation = _mapping(raw_evaluation)
    return (
        f"{float(evaluation.get('pairwise_accuracy', 0.0)):.3f}/"
        f"{float(evaluation.get('min_margin', 0.0)):+.3f}/"
        f"{int(evaluation.get('failed_pair_count', 0))}"
    )


def _failed_bridge_direction_table(report: Mapping[str, Any]) -> list[str]:
    inputs = _mapping(report.get("inputs"))
    min_pairwise_accuracy = float(inputs.get("min_pairwise_accuracy", 1.0))
    min_margin = float(inputs.get("min_margin", 0.0))
    min_direction_cosine = float(inputs.get("min_direction_cosine", 0.0))
    lines = [
        "| Direction fold | Reason | Source accuracy/min margin | Target accuracy/min margin | Joint cosine |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for fold in [
        *(
            _mapping(item)
            for item in _sequence(report.get("target_bridge_direction_folds"))
        ),
        *(
            _mapping(item)
            for item in _sequence(report.get("source_bridge_direction_folds"))
        ),
    ]:
        source_eval = _mapping(fold.get("on_source"))
        target_eval = _mapping(fold.get("on_target"))
        reasons: list[str] = []
        if _evaluation_failed(
            source_eval,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ):
            reasons.append("source_eval")
        if _evaluation_failed(
            target_eval,
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ):
            reasons.append("target_eval")
        if float(fold.get("joint_direction_cosine", 0.0)) < min_direction_cosine:
            reasons.append("joint_cosine")
        if not bool(fold.get("path_complete", False)):
            reasons.append("path_complete")
        if not reasons:
            continue
        lines.append(
            "| "
            f"`{fold.get('held_out_group', '')}` | "
            f"`{', '.join(reasons)}` | "
            f"{float(source_eval.get('pairwise_accuracy', 0.0)):.3f}/"
            f"{float(source_eval.get('min_margin', 0.0)):+.3f} | "
            f"{float(target_eval.get('pairwise_accuracy', 0.0)):.3f}/"
            f"{float(target_eval.get('min_margin', 0.0)):+.3f} | "
            f"{float(fold.get('joint_direction_cosine', 0.0)):+.3f} |"
        )
    if len(lines) == 2:
        return ["No failed constructed directions."]
    return lines


def _transport_bridge_direction_table(raw_rows: object) -> list[str]:
    rows = [_mapping(row) for row in _sequence(raw_rows)]
    has_fresh_source = any(
        row.get("mapped_on_target_model_fresh_source") is not None for row in rows
    )
    has_fresh_target = any(
        row.get("mapped_on_target_model_fresh_target") is not None for row in rows
    )
    headers = [
        "Fold",
        "Cosine",
        "Source acc",
        "Source min margin",
        "Target acc",
        "Target min margin",
        "Leave-held-out acc",
        "Leave-held-out min margin",
    ]
    aligns = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    if has_fresh_source:
        headers.extend(["Fresh source acc", "Fresh source min margin"])
        aligns.extend(["---:", "---:"])
    if has_fresh_target:
        headers.extend(["Fresh target acc", "Fresh target min margin"])
        aligns.extend(["---:", "---:"])
    headers.append("Failed pairs")
    aligns.append("---:")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        source_eval = _mapping(row.get("mapped_on_target_model_source"))
        target_eval = _mapping(row.get("mapped_on_target_model_target"))
        leave_eval = _mapping(row.get("leave_held_out_map_evaluation"))
        fresh_source_eval = _mapping(row.get("mapped_on_target_model_fresh_source"))
        fresh_target_eval = _mapping(row.get("mapped_on_target_model_fresh_target"))
        cells = [
            f"`{row.get('fold_id', '')}`",
            f"{float(row.get('mapped_direction_cosine_to_target_bridge', 0.0)):+.3f}",
            f"{float(source_eval.get('pairwise_accuracy', 0.0)):.3f}",
            f"{float(source_eval.get('min_margin', 0.0)):+.3f}",
            f"{float(target_eval.get('pairwise_accuracy', 0.0)):.3f}",
            f"{float(target_eval.get('min_margin', 0.0)):+.3f}",
            f"{float(leave_eval.get('pairwise_accuracy', 0.0)):.3f}",
            f"{float(leave_eval.get('min_margin', 0.0)):+.3f}",
        ]
        if has_fresh_source:
            cells.extend(
                [
                    f"{float(fresh_source_eval.get('pairwise_accuracy', 0.0)):.3f}",
                    f"{float(fresh_source_eval.get('min_margin', 0.0)):+.3f}",
                ]
            )
        if has_fresh_target:
            cells.extend(
                [
                    f"{float(fresh_target_eval.get('pairwise_accuracy', 0.0)):.3f}",
                    f"{float(fresh_target_eval.get('min_margin', 0.0)):+.3f}",
                ]
            )
        cells.append(str(_transport_failed_pair_count(row)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _transport_failed_pair_count(row: Mapping[str, Any]) -> int:
    return sum(
        int(_mapping(row.get(key)).get("failed_pair_count", 0))
        for key in (
            "mapped_on_target_model_source",
            "mapped_on_target_model_target",
            "leave_held_out_map_evaluation",
            "mapped_on_target_model_fresh_source",
            "mapped_on_target_model_fresh_target",
        )
    )


def _failed_transport_bridge_direction_table(report: Mapping[str, Any]) -> list[str]:
    inputs = _mapping(report.get("inputs"))
    min_pairwise_accuracy = float(inputs.get("min_pairwise_accuracy", 1.0))
    min_margin = float(inputs.get("min_margin", 0.0))
    min_mapped_direction_cosine = float(
        inputs.get("min_mapped_direction_cosine", 0.0)
    )
    lines = [
        "| Direction | Reason | Source accuracy/min margin | Target accuracy/min margin | Leave-held-out accuracy/min margin | Cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for direction_name, raw_rows in (
        ("model_a_to_b", report.get("model_a_to_b_transport")),
        ("model_b_to_a", report.get("model_b_to_a_transport")),
    ):
        for row in (_mapping(item) for item in _sequence(raw_rows)):
            source_eval = _mapping(row.get("mapped_on_target_model_source"))
            target_eval = _mapping(row.get("mapped_on_target_model_target"))
            leave_eval = _mapping(row.get("leave_held_out_map_evaluation"))
            reasons = _transport_failure_reasons(
                row,
                min_pairwise_accuracy=min_pairwise_accuracy,
                min_margin=min_margin,
                min_mapped_direction_cosine=min_mapped_direction_cosine,
            )
            if not reasons:
                continue
            lines.append(
                "| "
                f"`{direction_name}:{row.get('fold_id', '')}` | "
                f"`{', '.join(reasons)}` | "
                f"{float(source_eval.get('pairwise_accuracy', 0.0)):.3f}/"
                f"{float(source_eval.get('min_margin', 0.0)):+.3f} | "
                f"{float(target_eval.get('pairwise_accuracy', 0.0)):.3f}/"
                f"{float(target_eval.get('min_margin', 0.0)):+.3f} | "
                f"{float(leave_eval.get('pairwise_accuracy', 0.0)):.3f}/"
                f"{float(leave_eval.get('min_margin', 0.0)):+.3f} | "
                f"{float(row.get('mapped_direction_cosine_to_target_bridge', 0.0)):+.3f} |"
            )
    if len(lines) == 2:
        return ["No failed transported directions."]
    return lines


def _failed_fresh_bridge_pair_table(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Direction | Family | Evaluation | Pair | Margin |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in (
        _mapping(item) for item in _sequence(report.get("direction_evaluations"))
    ):
        for key in _FRESH_BRIDGE_EVALUATION_KEYS:
            evaluation = _mapping(row.get(key))
            for raw_pair in _sequence(evaluation.get("failed_pairs")):
                pair = _mapping(raw_pair)
                lines.append(
                    "| "
                    f"`{row.get('direction_id', '')}` | "
                    f"`{row.get('direction_family', '')}` | "
                    f"`{key.removeprefix('on_')}` | "
                    f"`{pair.get('pair_id', '')}` | "
                    f"{float(pair.get('margin', 0.0)):+.3f} |"
                )
    if len(lines) == 2:
        return ["No failed pairs."]
    return lines


def _failed_transport_pair_table(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Direction | Evaluation | Fold | Pair | Margin |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for direction_name, raw_rows in (
        ("model_a_to_b", report.get("model_a_to_b_transport")),
        ("model_b_to_a", report.get("model_b_to_a_transport")),
    ):
        for row in (_mapping(item) for item in _sequence(raw_rows)):
            for key, label in (
                ("mapped_on_target_model_source", "source"),
                ("mapped_on_target_model_target", "target"),
                ("leave_held_out_map_evaluation", "leave_held_out"),
                ("mapped_on_target_model_fresh_source", "fresh_source"),
                ("mapped_on_target_model_fresh_target", "fresh_target"),
            ):
                evaluation = _mapping(row.get(key))
                for raw_pair in _sequence(evaluation.get("failed_pairs")):
                    pair = _mapping(raw_pair)
                    lines.append(
                        "| "
                        f"`{direction_name}` | "
                        f"`{label}` | "
                        f"`{row.get('fold_id', '')}` | "
                        f"`{pair.get('pair_id', '')}` | "
                        f"{float(pair.get('margin', 0.0)):+.3f} |"
                    )
    if len(lines) == 2:
        return ["No failed transported pairs."]
    return lines


def _transport_failure_reasons(
    row: Mapping[str, Any],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
    min_mapped_direction_cosine: float,
) -> list[str]:
    reasons: list[str] = []
    for key, reason in (
        ("mapped_on_target_model_source", "source_eval"),
        ("mapped_on_target_model_target", "target_eval"),
        ("leave_held_out_map_evaluation", "leave_held_out_map"),
        ("mapped_on_target_model_fresh_source", "fresh_source_eval"),
        ("mapped_on_target_model_fresh_target", "fresh_target_eval"),
    ):
        if key not in row:
            continue
        if _evaluation_failed(
            _mapping(row.get(key)),
            min_pairwise_accuracy=min_pairwise_accuracy,
            min_margin=min_margin,
        ):
            reasons.append(reason)
    for key, reason in (
        ("mapped_direction_cosine_to_target_bridge", "bridge_cosine"),
        (
            "leave_held_out_map_cosine_to_target_bridge",
            "leave_held_out_bridge_cosine",
        ),
    ):
        if float(row.get(key, 0.0)) < min_mapped_direction_cosine:
            reasons.append(reason)
    return reasons


def _evaluation_failed(
    evaluation: Mapping[str, Any],
    *,
    min_pairwise_accuracy: float,
    min_margin: float,
) -> bool:
    return (
        float(evaluation.get("pairwise_accuracy", 0.0)) < min_pairwise_accuracy
        or float(evaluation.get("min_margin", 0.0)) < min_margin
    )


def _coverage_summary(raw_coverage_values: object) -> str:
    coverage_values = _mapping(raw_coverage_values)
    return "; ".join(
        f"{key}={','.join(str(value) for value in _sequence(values))}"
        for key, values in sorted(coverage_values.items())
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []
