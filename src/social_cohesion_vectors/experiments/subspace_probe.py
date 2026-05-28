"""Multi-direction activation subspace probes.

This module stress-tests the one-vector story by fitting signed SVD bases over
positive-minus-negative pair differences, then comparing signed component votes
against squared subspace energy. The squared score is intentionally separate:
it answers the reviewer concern that unsigned localization can hide axis sign.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.transfer import load_activation_payload
from social_cohesion_vectors.schemas import PairwiseExample

_EPSILON = 1e-12


@dataclass(frozen=True)
class PairActivation:
    pair_id: str
    group_values: tuple[str, ...]
    positive: np.ndarray
    negative: np.ndarray

    @property
    def difference(self) -> np.ndarray:
        return self.positive - self.negative


@dataclass(frozen=True)
class SignedBasis:
    kind: str
    components: np.ndarray
    singular_values: np.ndarray
    global_direction: np.ndarray | None


def run_activation_subspace_probe_from_files(
    *,
    activation_npz: str | Path,
    pairs_path: str | Path,
    metadata_key: str = "primary_fault_class",
    max_components: int = 8,
) -> dict[str, Any]:
    """Load activations and pair metadata, then run the subspace probe."""

    return run_activation_subspace_probe(
        activation_npz=activation_npz,
        pairs=load_pairwise_examples_jsonl(pairs_path),
        pairs_path=str(pairs_path),
        metadata_key=metadata_key,
        max_components=max_components,
    )


def run_activation_subspace_probe(
    *,
    activation_npz: str | Path,
    pairs: Sequence[PairwiseExample],
    metadata_key: str = "primary_fault_class",
    pairs_path: str | None = None,
    max_components: int = 8,
) -> dict[str, Any]:
    """Evaluate signed and unsigned multi-component activation probes."""

    payload = load_activation_payload(activation_npz)
    records = pair_activation_records(
        activations=payload.activations,
        pair_ids=[str(pair_id) for pair_id in payload.pair_ids],
        labels=[str(label) for label in payload.labels],
        pairs=pairs,
        metadata_key=metadata_key,
    )
    component_count = max(1, min(max_components, len(records), payload.activations.shape[1]))
    component_values = list(range(1, component_count + 1))
    rows = [
        *evaluate_component_sweep(
            records,
            split="leave_one_pair_out",
            folds=pair_folds(records),
            component_values=component_values,
        ),
        *evaluate_component_sweep(
            records,
            split=f"leave_one_{metadata_key}_out",
            folds=metadata_folds(records),
            component_values=component_values,
        ),
    ]
    return {
        "experiment": "activation_subspace_probe",
        "description": (
            "Fits k-component signed SVD bases over positive-minus-negative "
            "activation differences and compares signed votes with squared "
            "subspace energy."
        ),
        "inputs": {
            "activation_npz": str(activation_npz),
            "pairs_path": pairs_path,
            "metadata_key": metadata_key,
            "pairs": len(records),
            "prompts": int(payload.activations.shape[0]),
            "activation_dim": int(payload.activations.shape[1]),
            "max_components": component_count,
        },
        "summary": summarize_rows(rows),
        "pair_difference_subspace": pair_difference_subspace(records),
        "component_sweeps": rows,
    }


def pair_activation_records(
    *,
    activations: np.ndarray,
    pair_ids: Sequence[str],
    labels: Sequence[str],
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> list[PairActivation]:
    """Create complete pair-level positive/negative activation records."""

    group_values = {
        pair.pair_id: metadata_values(pair.metadata.get(metadata_key)) for pair in pairs
    }
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_id, label, activation in zip(pair_ids, labels, activations, strict=True):
        grouped[str(pair_id)][str(label)].append(np.asarray(activation, dtype=np.float64))

    records: list[PairActivation] = []
    for pair_id, label_vectors in sorted(grouped.items()):
        if "positive" not in label_vectors or "negative" not in label_vectors:
            continue
        records.append(
            PairActivation(
                pair_id=pair_id,
                group_values=group_values.get(pair_id, ("unlabeled",)),
                positive=np.vstack(label_vectors["positive"]).mean(axis=0),
                negative=np.vstack(label_vectors["negative"]).mean(axis=0),
            )
        )
    return records


def evaluate_component_sweep(
    records: Sequence[PairActivation],
    *,
    split: str,
    folds: Sequence[tuple[str, set[str]]],
    component_values: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate raw and residual signed bases across k components."""

    rows: list[dict[str, Any]] = []
    max_components = max(component_values, default=1)
    for basis_kind in ("raw_pair_difference_svd", "residual_after_global_direction"):
        reports_by_component: dict[int, list[dict[str, Any]]] = {
            components: [] for components in component_values
        }
        for held_out, test_pair_ids in folds:
            train_records = [
                record for record in records if record.pair_id not in test_pair_ids
            ]
            test_records = [
                record for record in records if record.pair_id in test_pair_ids
            ]
            if not train_records or not test_records:
                continue
            basis = fit_signed_basis(
                train_records,
                kind=basis_kind,
                components=max_components,
            )
            for components in component_values:
                prefix = basis_prefix(basis, components)
                reports_by_component[components].append(
                    {
                        "split": split,
                        "held_out": held_out,
                        "basis": prefix.kind,
                        "components": components,
                        "actual_components": int(prefix.components.shape[0]),
                        "train_pairs": len(train_records),
                        "test_pairs": len(test_records),
                        **score_records(test_records, prefix),
                    }
                )
        for components in component_values:
            rows.append(
                summarize_fold_reports(
                    split,
                    basis_kind,
                    components,
                    reports_by_component[components],
                )
            )
    return rows


def basis_prefix(basis: SignedBasis, components: int) -> SignedBasis:
    """Return the first k components from a prefit basis."""

    actual = max(0, min(components, basis.components.shape[0]))
    return SignedBasis(
        kind=basis.kind,
        components=basis.components[:actual],
        singular_values=basis.singular_values[:actual],
        global_direction=basis.global_direction,
    )


def evaluate_fold(
    records: Sequence[PairActivation],
    *,
    split: str,
    held_out: str,
    test_pair_ids: set[str],
    basis_kind: str,
    components: int,
) -> dict[str, Any] | None:
    """Train one basis on train pairs and score held-out pairs."""

    train_records = [record for record in records if record.pair_id not in test_pair_ids]
    test_records = [record for record in records if record.pair_id in test_pair_ids]
    if not train_records or not test_records:
        return None
    basis = fit_signed_basis(
        train_records,
        kind=basis_kind,
        components=components,
    )
    scored = score_records(test_records, basis)
    return {
        "split": split,
        "held_out": held_out,
        "basis": basis.kind,
        "components": components,
        "train_pairs": len(train_records),
        "test_pairs": len(test_records),
        **scored,
    }


def fit_signed_basis(
    records: Sequence[PairActivation],
    *,
    kind: str,
    components: int,
) -> SignedBasis:
    """Fit an oriented SVD basis from pair differences."""

    differences = difference_matrix(records)
    global_direction = normalized(differences.mean(axis=0)) if len(differences) else None
    training_differences = differences
    if kind == "residual_after_global_direction" and global_direction is not None:
        training_differences = project_out(differences, global_direction)
    elif kind != "raw_pair_difference_svd":
        raise ValueError(f"Unknown basis kind: {kind}")

    if training_differences.size == 0:
        return SignedBasis(
            kind=kind,
            components=np.empty((0, differences.shape[1] if differences.ndim == 2 else 0)),
            singular_values=np.empty((0,), dtype=np.float64),
            global_direction=global_direction,
        )

    _, singular_values, vt = np.linalg.svd(training_differences, full_matrices=False)
    selected = vt[:components].astype(np.float64, copy=True)
    selected_singular_values = singular_values[: selected.shape[0]].astype(np.float64)
    for index in range(selected.shape[0]):
        if float(np.mean(training_differences @ selected[index])) < 0:
            selected[index] *= -1.0
    return SignedBasis(
        kind=kind,
        components=selected,
        singular_values=selected_singular_values,
        global_direction=global_direction,
    )


def score_records(
    records: Sequence[PairActivation],
    basis: SignedBasis,
) -> dict[str, Any]:
    """Score held-out pairs with signed and squared subspace margins."""

    if basis.components.size == 0:
        return metric_block([], [], [], [])

    signed_sum_margins: list[float] = []
    signed_vote_margins: list[float] = []
    weighted_signed_margins: list[float] = []
    squared_energy_margins: list[float] = []
    weights = component_weights(basis.singular_values)
    for record in records:
        positive_projection = basis.components @ record.positive
        negative_projection = basis.components @ record.negative
        difference_projection = positive_projection - negative_projection
        signed_sum_margins.append(float(np.sum(difference_projection)))
        signed_vote_margins.append(float(np.sum(np.sign(difference_projection))))
        weighted_signed_margins.append(float(np.sum(difference_projection * weights)))
        squared_energy_margins.append(
            float(np.sum(positive_projection**2) - np.sum(negative_projection**2))
        )
    return metric_block(
        signed_sum_margins,
        signed_vote_margins,
        weighted_signed_margins,
        squared_energy_margins,
    )


def metric_block(
    signed_sum_margins: Sequence[float],
    signed_vote_margins: Sequence[float],
    weighted_signed_margins: Sequence[float],
    squared_energy_margins: Sequence[float],
) -> dict[str, Any]:
    return {
        "signed_sum_accuracy": accuracy_from_margins(signed_sum_margins),
        "signed_vote_accuracy": accuracy_from_margins(signed_vote_margins),
        "weighted_signed_accuracy": accuracy_from_margins(weighted_signed_margins),
        "squared_energy_accuracy": accuracy_from_margins(squared_energy_margins),
        "mean_signed_sum_margin": mean(signed_sum_margins),
        "mean_signed_vote_margin": mean(signed_vote_margins),
        "mean_weighted_signed_margin": mean(weighted_signed_margins),
        "mean_squared_energy_margin": mean(squared_energy_margins),
        "min_signed_sum_margin": minimum(signed_sum_margins),
        "min_squared_energy_margin": minimum(squared_energy_margins),
    }


def summarize_fold_reports(
    split: str,
    basis_kind: str,
    components: int,
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test_pairs = sum(int(fold.get("test_pairs", 0)) for fold in folds)
    return {
        "split": split,
        "basis": basis_kind,
        "components": components,
        "folds": len(folds),
        "test_pairs": test_pairs,
        "signed_sum_accuracy": weighted_mean(folds, "signed_sum_accuracy", "test_pairs"),
        "signed_vote_accuracy": weighted_mean(folds, "signed_vote_accuracy", "test_pairs"),
        "weighted_signed_accuracy": weighted_mean(
            folds,
            "weighted_signed_accuracy",
            "test_pairs",
        ),
        "squared_energy_accuracy": weighted_mean(
            folds,
            "squared_energy_accuracy",
            "test_pairs",
        ),
        "mean_signed_sum_margin": weighted_mean(
            folds,
            "mean_signed_sum_margin",
            "test_pairs",
        ),
        "mean_squared_energy_margin": weighted_mean(
            folds,
            "mean_squared_energy_margin",
            "test_pairs",
        ),
        "fold_details": list(folds),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_signed = best_row(rows, "signed_vote_accuracy")
    best_energy = best_row(rows, "squared_energy_accuracy")
    return {
        "rows": len(rows),
        "best_signed_vote": compact_best(best_signed),
        "best_squared_energy": compact_best(best_energy),
        "best_pair_loo_signed_vote_accuracy": best_accuracy(
            rows,
            split="leave_one_pair_out",
            metric="signed_vote_accuracy",
        ),
        "best_pair_loo_squared_energy_accuracy": best_accuracy(
            rows,
            split="leave_one_pair_out",
            metric="squared_energy_accuracy",
        ),
    }


def save_activation_subspace_probe_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_activation_subspace_probe_markdown(report), encoding="utf-8")


def render_activation_subspace_probe_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact reviewer-facing subspace report."""

    inputs = mapping(report.get("inputs"))
    summary = mapping(report.get("summary"))
    subspace = mapping(report.get("pair_difference_subspace"))
    rows = sequence(report.get("component_sweeps"))
    lines = [
        "# Activation Subspace Probe",
        "",
        str(report.get("description", "")),
        "",
        "## Inputs",
        "",
        f"- Activation file: `{Path(str(inputs.get('activation_npz', ''))).name}`",
        f"- Metadata key: `{inputs.get('metadata_key', '')}`",
        f"- Pairs: {int(inputs.get('pairs', 0))}",
        f"- Prompts: {int(inputs.get('prompts', 0))}",
        f"- Activation dim: {int(inputs.get('activation_dim', 0))}",
        f"- Max components: {int(inputs.get('max_components', 0))}",
        "",
        "## Summary",
        "",
        "- Best pair-LOO signed-vote accuracy: "
        f"{float(summary.get('best_pair_loo_signed_vote_accuracy', 0.0)):.3f}",
        "- Best pair-LOO squared-energy accuracy: "
        f"{float(summary.get('best_pair_loo_squared_energy_accuracy', 0.0)):.3f}",
        "- Pair-difference energy captured by first component: "
        f"{first_energy(subspace):.3f}",
        "",
        "## Component Sweeps",
        "",
        "| Split | Basis | k | Test pairs | Signed vote acc | Weighted signed acc | Squared energy acc | Signed margin | Energy margin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        item = mapping(row)
        lines.append(
            "| "
            f"{item.get('split', '')} | "
            f"{item.get('basis', '')} | "
            f"{int(item.get('components', 0))} | "
            f"{int(item.get('test_pairs', 0))} | "
            f"{float(item.get('signed_vote_accuracy', 0.0)):.3f} | "
            f"{float(item.get('weighted_signed_accuracy', 0.0)):.3f} | "
            f"{float(item.get('squared_energy_accuracy', 0.0)):.3f} | "
            f"{float(item.get('mean_signed_sum_margin', 0.0)):+.3f} | "
            f"{float(item.get('mean_squared_energy_margin', 0.0)):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            (
                "Signed-vote accuracy is the relevant discrimination check. "
                "Squared-energy accuracy is useful for localization, but it can "
                "erase whether a component points toward the positive or negative "
                "pole."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def pair_difference_subspace(records: Sequence[PairActivation]) -> dict[str, Any]:
    differences = difference_matrix(records)
    return {
        "pairs": int(differences.shape[0]) if differences.ndim == 2 else 0,
        "svd_cumulative_energy": svd_cumulative_energy(differences),
    }


def pair_folds(records: Sequence[PairActivation]) -> list[tuple[str, set[str]]]:
    return [(record.pair_id, {record.pair_id}) for record in records]


def metadata_folds(records: Sequence[PairActivation]) -> list[tuple[str, set[str]]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for value in record.group_values:
            grouped[value].add(record.pair_id)
    if len(grouped) < 2:
        return []
    return sorted(grouped.items())


def difference_matrix(records: Sequence[PairActivation]) -> np.ndarray:
    if not records:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack([record.difference for record in records]).astype(np.float64)


def project_out(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return matrix - np.outer(matrix @ direction, direction)


def normalized(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def component_weights(singular_values: np.ndarray) -> np.ndarray:
    if singular_values.size == 0:
        return singular_values
    total = float(np.sum(np.abs(singular_values)))
    if total <= _EPSILON:
        return np.ones_like(singular_values) / float(singular_values.size)
    return np.abs(singular_values) / total


def accuracy_from_margins(margins: Sequence[float]) -> float:
    if not margins:
        return 0.0
    outcomes = [1.0 if margin > 0 else 0.5 if margin == 0 else 0.0 for margin in margins]
    return round(float(sum(outcomes) / len(outcomes)), 6)


def weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    weight_key: str,
) -> float:
    total_weight = sum(int(row.get(weight_key, 0)) for row in rows)
    if total_weight == 0:
        return 0.0
    total = sum(
        float(row.get(value_key, 0.0)) * int(row.get(weight_key, 0)) for row in rows
    )
    return round(total / total_weight, 6)


def best_row(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Mapping[str, Any] | None:
    scored = [row for row in rows if isinstance(row.get(metric), int | float)]
    if not scored:
        return None
    return max(
        scored,
        key=lambda row: (
            float(row.get(metric, 0.0)),
            float(row.get("weighted_signed_accuracy", 0.0)),
            float(row.get("mean_signed_sum_margin", 0.0)),
        ),
    )


def compact_best(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "split": row.get("split"),
        "basis": row.get("basis"),
        "components": row.get("components"),
        "signed_vote_accuracy": row.get("signed_vote_accuracy"),
        "weighted_signed_accuracy": row.get("weighted_signed_accuracy"),
        "squared_energy_accuracy": row.get("squared_energy_accuracy"),
    }


def best_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    metric: str,
) -> float:
    values = [
        float(row.get(metric, 0.0))
        for row in rows
        if row.get("split") == split and isinstance(row.get(metric), int | float)
    ]
    return round(max(values), 6) if values else 0.0


def svd_cumulative_energy(
    differences: np.ndarray,
    max_components: int = 8,
) -> list[dict[str, Any]]:
    if differences.size == 0:
        return []
    singular_values = np.linalg.svd(differences, full_matrices=False, compute_uv=False)
    energies = singular_values**2
    total = float(np.sum(energies))
    if total <= _EPSILON:
        return []
    rows: list[dict[str, Any]] = []
    running = 0.0
    for index, energy in enumerate(energies[:max_components], start=1):
        running += float(energy)
        rows.append(
            {
                "components": index,
                "singular_value": round(float(singular_values[index - 1]), 6),
                "cumulative_energy_fraction": round(running / total, 6),
            }
        )
    return rows


def first_energy(subspace: Mapping[str, Any]) -> float:
    rows = sequence(subspace.get("svd_cumulative_energy"))
    if not rows:
        return 0.0
    first = mapping(rows[0])
    return float(first.get("cumulative_energy_fraction", 0.0))


def metadata_values(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ("unlabeled",)
    values = tuple(
        dict.fromkeys(
            part.strip() for part in str(raw_value).split(",") if part.strip()
        )
    )
    return values or ("unlabeled",)


def mean(values: Sequence[float]) -> float:
    return round(float(np.mean(values)), 6) if values else 0.0


def minimum(values: Sequence[float]) -> float:
    return round(float(np.min(values)), 6) if values else 0.0


def mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
