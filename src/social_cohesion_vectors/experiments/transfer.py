"""Transfer and cross-validation experiments for cohesion ranking baselines."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.activations.contrastive import train_direction_from_arrays
from social_cohesion_vectors.datasets import (
    load_pairwise_examples_jsonl,
    load_scored_runs_jsonl,
)
from social_cohesion_vectors.experiments.baselines import (
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    strategy_prior_score,
)
from social_cohesion_vectors.scenario_library import load_scenarios
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun

ScoreFn = Callable[[ScoredRun], float]
FeatureFn = Callable[[ScoredRun], np.ndarray]

_EPSILON = 1e-12
_KIND_RE = re.compile(r"\bkind=([a-zA-Z0-9_:-]+)")
_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_METRIC_FEATURES = (
    ("cooperation_rate", "cooperation"),
    ("repair_attempt_rate", "repair"),
    ("fairness_score", "fairness"),
    ("hostility_rate", "hostility"),
    ("joint_payoff",),
    ("defection_rate", "defection"),
)
_TEXT_BASELINE_DESCRIPTIONS = {
    "strategy_prior": "Fixed metadata prior: cooperative > self-protective > adversarial.",
    "metrics_only": "Train split contrastive direction over numeric simulation metrics only.",
    "lexical_only": "Train split contrastive direction over hand-written lexical counts only.",
}


@dataclass(frozen=True)
class ActivationPayload:
    path: str
    activations: np.ndarray
    pair_ids: np.ndarray
    labels: np.ndarray


def run_transfer_from_files(
    *,
    scored_runs_path: str | Path,
    pairs_path: str | Path,
    generated_scored_runs_path: str | Path | None = None,
    generated_pairs_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
    activation_npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load inputs from disk and run all available transfer experiments."""

    scored_runs = load_scored_runs_jsonl(scored_runs_path)
    generated_runs = load_optional_scored_runs(generated_scored_runs_path)
    pairs = load_pairwise_examples_jsonl(pairs_path)
    generated_pairs = load_optional_pairwise_examples(generated_pairs_path)
    scenario_kinds = load_scenario_kind_map(scenarios_path)
    return run_transfer_experiment(
        scored_runs=scored_runs,
        generated_scored_runs=generated_runs,
        pairs=pairs,
        generated_pairs=generated_pairs,
        scenario_kinds=scenario_kinds,
        activation_npz_path=activation_npz_path,
        input_paths={
            "scored_runs": str(scored_runs_path),
            "generated_scored_runs": (
                str(generated_scored_runs_path)
                if generated_scored_runs_path is not None
                else None
            ),
            "pairs": str(pairs_path),
            "generated_pairs": (
                str(generated_pairs_path) if generated_pairs_path is not None else None
            ),
            "scenarios": str(scenarios_path) if scenarios_path is not None else None,
            "activation_npz": (
                str(activation_npz_path) if activation_npz_path is not None else None
            ),
        },
    )


def run_transfer_experiment(
    *,
    scored_runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    generated_scored_runs: Sequence[ScoredRun] = (),
    generated_pairs: Sequence[PairwiseExample] = (),
    scenario_kinds: Mapping[str, str] | None = None,
    activation_npz_path: str | Path | None = None,
    input_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run text/rubric transfer folds and optional activation-vector folds."""

    all_runs = [*scored_runs, *generated_scored_runs]
    all_pairs = [*pairs, *generated_pairs]
    run_index = index_runs(all_runs)
    inferred_kinds = infer_scenario_kinds(all_runs, all_pairs, scenario_kinds or {})
    text_results = evaluate_text_transfer(
        pairs=pairs,
        run_index=run_index,
        scenario_kinds=inferred_kinds,
    )
    pair_set_results = evaluate_pair_set_transfer(
        train_pairs=pairs,
        test_pairs=generated_pairs,
        run_index=run_index,
    )
    activation_results = evaluate_activation_transfer_from_npz(
        activation_npz_path,
        pairs=all_pairs,
        pair_set_transfers=(
            _pair_set_transfer_specs(pairs, generated_pairs) if generated_pairs else ()
        ),
    )
    return {
        "inputs": {
            "paths": dict(input_paths or {}),
            "n_scored_runs": len(scored_runs),
            "n_generated_scored_runs": len(generated_scored_runs),
            "n_total_runs": len(all_runs),
            "n_indexed_runs": len(run_index),
            "n_pairs": len(pairs),
            "n_generated_pairs": len(generated_pairs),
            "n_total_pairs": len(all_pairs),
            "n_scenarios": len({pair.scenario_id for pair in all_pairs}),
            "n_scenario_kinds": len(set(inferred_kinds.values())),
        },
        "text_transfer": {
            "by_scenario_kind": text_results["scenario_kind"],
            "by_scenario_id": text_results["scenario_id"],
            "by_pair_set": pair_set_results,
            "summary": summarize_fold_results(
                [
                    *text_results["scenario_kind"],
                    *text_results["scenario_id"],
                    *pair_set_results,
                ]
            ),
        },
        "activation_transfer": activation_results,
    }


def load_optional_scored_runs(path: str | Path | None) -> list[ScoredRun]:
    if path is None:
        return []
    scored_path = Path(path)
    if not scored_path.exists():
        return []
    return load_scored_runs_jsonl(scored_path)


def load_optional_pairwise_examples(path: str | Path | None) -> list[PairwiseExample]:
    if path is None:
        return []
    pairs_path = Path(path)
    if not pairs_path.exists():
        return []
    return load_pairwise_examples_jsonl(pairs_path)


def load_scenario_kind_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    scenario_path = Path(path)
    if not scenario_path.exists():
        return {}
    return {scenario.id: str(scenario.kind) for scenario in load_scenarios(scenario_path)}


def index_runs(runs: Sequence[ScoredRun]) -> dict[str, ScoredRun]:
    indexed: dict[str, ScoredRun] = {}
    for run in runs:
        indexed[run.run_id] = run
    return indexed


def infer_scenario_kinds(
    runs: Sequence[ScoredRun],
    pairs: Sequence[PairwiseExample],
    known: Mapping[str, str],
) -> dict[str, str]:
    scenario_kinds = dict(known)
    for run in runs:
        scenario_kinds.setdefault(run.scenario_id, _kind_from_text(run.transcript))
    for pair in pairs:
        scenario_kinds.setdefault(pair.scenario_id, _kind_from_text(pair.positive_text))
        scenario_kinds.setdefault(pair.scenario_id, _kind_from_text(pair.negative_text))
    return {key: value for key, value in scenario_kinds.items() if value}


def evaluate_text_transfer(
    *,
    pairs: Sequence[PairwiseExample],
    run_index: Mapping[str, ScoredRun],
    scenario_kinds: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "scenario_kind": _evaluate_text_folds(
            folds=_make_folds(
                pairs,
                split="scenario_kind",
                group_fn=lambda pair: scenario_kinds.get(pair.scenario_id),
            ),
            run_index=run_index,
        ),
        "scenario_id": _evaluate_text_folds(
            folds=_make_folds(
                pairs,
                split="scenario_id",
                group_fn=lambda pair: pair.scenario_id,
            ),
            run_index=run_index,
        ),
    }


def evaluate_pair_set_transfer(
    *,
    train_pairs: Sequence[PairwiseExample],
    test_pairs: Sequence[PairwiseExample],
    run_index: Mapping[str, ScoredRun],
) -> list[dict[str, Any]]:
    """Train baselines on one pair set and evaluate on another pair set."""

    return _evaluate_text_folds(
        folds=_pair_set_transfer_specs(train_pairs, test_pairs),
        run_index=run_index,
    )


def evaluate_pair_scores(
    pairs: Sequence[PairwiseExample],
    *,
    run_index: Mapping[str, ScoredRun],
    score_fn: ScoreFn,
) -> dict[str, float | int]:
    outcomes: list[float] = []
    margins: list[float] = []
    skipped = 0
    ties = 0
    for pair in pairs:
        runs = _pair_runs(pair, run_index)
        if runs is None:
            skipped += 1
            continue
        positive, negative = runs
        margin = score_fn(positive) - score_fn(negative)
        margins.append(margin)
        if margin > 0:
            outcomes.append(1.0)
        elif margin < 0:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)
            ties += 1
    return _metric_summary(outcomes, margins, skipped=skipped, ties=ties)


def evaluate_activation_transfer_from_npz(
    path: str | Path | None,
    *,
    pairs: Sequence[PairwiseExample],
    pair_set_transfers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    if path is None:
        return None
    activation_path = Path(path)
    if not activation_path.exists():
        return None

    payload = load_activation_payload(activation_path)
    pair_scenarios = activation_pair_scenarios(
        [str(pair_id) for pair_id in payload.pair_ids],
        pairs,
    )
    pair_folds = _activation_pair_folds(payload)
    scenario_folds = _activation_scenario_folds(payload, pair_scenarios)
    return {
        "baseline": "activation_vector",
        "activation_npz": payload.path,
        "n_prompts": int(payload.activations.shape[0]),
        "activation_dim": int(payload.activations.shape[1]),
        "leave_one_pair_out": _evaluate_activation_folds(
            payload,
            folds=pair_folds,
            split="pair_id",
        ),
        "leave_one_scenario_out": _evaluate_activation_folds(
            payload,
            folds=scenario_folds,
            split="scenario_id",
        ),
        "pair_set_transfer": _evaluate_activation_pair_set_transfers(
            payload,
            folds=pair_set_transfers,
        ),
    }


def load_activation_payload(path: str | Path) -> ActivationPayload:
    activation_path = Path(path)
    with np.load(activation_path, allow_pickle=False) as data:
        activations = _npz_array(data, ("activations", "pooled_activations", "vectors"))
        pair_ids = _npz_array(data, ("pair_ids",))
        labels = _npz_array(data, ("labels", "label"))
    return ActivationPayload(
        path=str(activation_path),
        activations=_activation_matrix(activations),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
    )


def activation_pair_scenarios(
    pair_ids: Sequence[object],
    pairs: Sequence[PairwiseExample],
) -> dict[str, str]:
    from_pairs = {pair.pair_id: pair.scenario_id for pair in pairs}
    scenarios: dict[str, str] = {}
    for raw_pair_id in pair_ids:
        pair_id = str(raw_pair_id)
        scenarios[pair_id] = from_pairs.get(pair_id) or _scenario_prefix(pair_id)
    return {pair_id: scenario for pair_id, scenario in scenarios.items() if scenario}


def find_activation_npz(root: str | Path) -> Path | None:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix == ".npz":
        return root_path
    if not root_path.exists():
        return None
    candidates = sorted(root_path.rglob("*.npz"))
    if not candidates:
        return None
    preferred = [
        path for path in candidates if path.name.startswith("activation_prompts__")
    ]
    return max(preferred or candidates, key=lambda path: path.stat().st_mtime)


def save_transfer_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    inputs = _mapping(report.get("inputs"))
    text = _mapping(report.get("text_transfer"))
    activation = report.get("activation_transfer")
    lines = [
        "# Transfer Experiment",
        "",
        f"- Scored runs: {inputs.get('n_scored_runs', 0)}",
        f"- Generated scored runs: {inputs.get('n_generated_scored_runs', 0)}",
        f"- Pairwise examples: {inputs.get('n_pairs', 0)}",
        f"- Generated pairwise examples: {inputs.get('n_generated_pairs', 0)}",
        f"- Scenario ids: {inputs.get('n_scenarios', 0)}",
        f"- Scenario kinds: {inputs.get('n_scenario_kinds', 0)}",
        "",
        "## Text/Rubric Transfer Summary",
        "",
        "| Split | Baseline | Folds | Test pairs | Mean test accuracy | Mean train accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(_summary_rows(text.get("summary")))
    lines.extend(_fold_table("Held-out scenario kinds", text.get("by_scenario_kind")))
    lines.extend(_fold_table("Held-out scenario ids", text.get("by_scenario_id")))
    lines.extend(_fold_table("Pair-set transfer", text.get("by_pair_set")))
    lines.extend(_activation_lines(activation))
    return "\n".join(lines) + "\n"


def summarize_fold_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["split"]), str(result["baseline"]))].append(result)

    summaries: list[dict[str, Any]] = []
    for (split, baseline), fold_results in sorted(grouped.items()):
        train = _weighted_accuracy(fold_results, "train")
        test = _weighted_accuracy(fold_results, "test")
        summaries.append(
            {
                "split": split,
                "baseline": baseline,
                "folds": len(fold_results),
                "train_pairs": train["n_pairs"],
                "test_pairs": test["n_pairs"],
                "mean_train_accuracy": train["accuracy"],
                "mean_test_accuracy": test["accuracy"],
            }
        )
    return summaries


def _evaluate_text_folds(
    *,
    folds: Sequence[dict[str, Any]],
    run_index: Mapping[str, ScoredRun],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fold in folds:
        for baseline in _TEXT_BASELINE_DESCRIPTIONS:
            score_fn = _fit_text_baseline(
                baseline,
                fold["train_pairs"],
                run_index=run_index,
            )
            results.append(
                _text_fold_result(
                    baseline,
                    fold=fold,
                    run_index=run_index,
                    score_fn=score_fn,
                )
            )
    return results


def _text_fold_result(
    baseline: str,
    *,
    fold: Mapping[str, Any],
    run_index: Mapping[str, ScoredRun],
    score_fn: ScoreFn,
) -> dict[str, Any]:
    train_pairs = fold["train_pairs"]
    test_pairs = fold["test_pairs"]
    return {
        "split": fold["split"],
        "held_out": fold["held_out"],
        "baseline": baseline,
        "description": _TEXT_BASELINE_DESCRIPTIONS[baseline],
        "train": evaluate_pair_scores(
            train_pairs,
            run_index=run_index,
            score_fn=score_fn,
        ),
        "test": evaluate_pair_scores(
            test_pairs,
            run_index=run_index,
            score_fn=score_fn,
        ),
    }


def _fit_text_baseline(
    baseline: str,
    train_pairs: Sequence[PairwiseExample],
    *,
    run_index: Mapping[str, ScoredRun],
) -> ScoreFn:
    if baseline == "strategy_prior":
        return strategy_prior_score
    if baseline == "metrics_only":
        return _train_feature_score(train_pairs, run_index, metrics_feature)
    if baseline == "lexical_only":
        return _train_feature_score(train_pairs, run_index, lexical_feature)
    raise ValueError(f"Unknown text baseline: {baseline}")


def _train_feature_score(
    train_pairs: Sequence[PairwiseExample],
    run_index: Mapping[str, ScoredRun],
    feature_fn: FeatureFn,
) -> ScoreFn:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for pair in train_pairs:
        runs = _pair_runs(pair, run_index)
        if runs is None:
            continue
        positive, negative = runs
        positives.append(feature_fn(positive))
        negatives.append(feature_fn(negative))
    if not positives or not negatives:
        return lambda _run: 0.0
    direction = _feature_direction(positives, negatives)
    return lambda run: float(feature_fn(run) @ direction)


def metrics_feature(run: ScoredRun) -> np.ndarray:
    return np.asarray(
        [_metric_value(run.metrics, names) for names in _METRIC_FEATURES],
        dtype=np.float64,
    )


def lexical_feature(run: ScoredRun) -> np.ndarray:
    text = run.transcript.lower()
    positive = [float(len(re.findall(term, text))) for term in POSITIVE_WORDS]
    negative = [float(len(re.findall(term, text))) for term in NEGATIVE_WORDS]
    token_count = float(len(_TOKEN_RE.findall(text)))
    return np.asarray([*positive, *negative, np.log1p(token_count)], dtype=np.float64)


def _make_folds(
    pairs: Sequence[PairwiseExample],
    *,
    split: str,
    group_fn: Callable[[PairwiseExample], str | None],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[PairwiseExample]] = defaultdict(list)
    for pair in pairs:
        group = group_fn(pair)
        if group:
            grouped[group].append(pair)
    if len(grouped) < 2:
        return []
    grouped_pairs = [pair for group in grouped.values() for pair in group]
    return [
        _fold(split, held_out, test_pairs, grouped_pairs)
        for held_out, test_pairs in sorted(grouped.items())
    ]


def _fold(
    split: str,
    held_out: str,
    test_pairs: Sequence[PairwiseExample],
    grouped_pairs: Sequence[PairwiseExample],
) -> dict[str, Any]:
    test_ids = {pair.pair_id for pair in test_pairs}
    train_pairs = [pair for pair in grouped_pairs if pair.pair_id not in test_ids]
    return {
        "split": split,
        "held_out": held_out,
        "train_pairs": train_pairs,
        "test_pairs": list(test_pairs),
    }


def _pair_set_transfer_specs(
    train_pairs: Sequence[PairwiseExample],
    test_pairs: Sequence[PairwiseExample],
) -> list[dict[str, Any]]:
    if not train_pairs or not test_pairs:
        return []
    return [
        {
            "split": "scripted_to_generated",
            "held_out": "generated_pairs",
            "train_pairs": list(train_pairs),
            "test_pairs": list(test_pairs),
        },
        {
            "split": "generated_to_scripted",
            "held_out": "scripted_pairs",
            "train_pairs": list(test_pairs),
            "test_pairs": list(train_pairs),
        },
    ]


def _evaluate_activation_folds(
    payload: ActivationPayload,
    *,
    folds: Sequence[tuple[str, set[str]]],
    split: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    complete_pairs = set(_complete_activation_pairs(payload.pair_ids, payload.labels))
    for held_out, test_pair_ids in folds:
        test_ids = test_pair_ids & complete_pairs
        train_ids = complete_pairs - test_ids
        if not test_ids or not train_ids:
            continue
        results.append(
            _activation_fold_result(
                payload,
                split=split,
                held_out=held_out,
                train_pair_ids=train_ids,
                test_pair_ids=test_ids,
            )
        )
    return results


def _evaluate_activation_pair_set_transfers(
    payload: ActivationPayload,
    *,
    folds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    complete_pairs = set(_complete_activation_pairs(payload.pair_ids, payload.labels))
    for fold in folds:
        train_ids = {
            pair.pair_id
            for pair in fold["train_pairs"]
            if pair.pair_id in complete_pairs
        }
        test_ids = {
            pair.pair_id for pair in fold["test_pairs"] if pair.pair_id in complete_pairs
        }
        if not train_ids or not test_ids:
            continue
        results.append(
            _activation_fold_result(
                payload,
                split=str(fold["split"]),
                held_out=str(fold["held_out"]),
                train_pair_ids=train_ids,
                test_pair_ids=test_ids,
            )
        )
    return results


def _activation_fold_result(
    payload: ActivationPayload,
    *,
    split: str,
    held_out: str,
    train_pair_ids: set[str],
    test_pair_ids: set[str],
) -> dict[str, Any]:
    train_mask = _pair_mask(payload.pair_ids, train_pair_ids)
    test_mask = _pair_mask(payload.pair_ids, test_pair_ids)
    direction = train_direction_from_arrays(
        payload.activations[train_mask],
        labels=payload.labels[train_mask],
    )
    train_eval = _activation_projection_eval(payload, train_mask, direction.direction)
    test_eval = _activation_projection_eval(payload, test_mask, direction.direction)
    return {
        "split": split,
        "held_out": held_out,
        "baseline": "activation_vector",
        "train": train_eval,
        "test": test_eval,
        "direction_top_count": direction.top_count,
        "direction_bottom_count": direction.bottom_count,
    }


def _activation_projection_eval(
    payload: ActivationPayload,
    mask: np.ndarray,
    direction: np.ndarray,
) -> dict[str, float | int]:
    projections = payload.activations[mask] @ direction
    return _projection_summary(
        pair_ids=payload.pair_ids[mask],
        labels=payload.labels[mask],
        projections=projections,
    )


def _projection_summary(
    *,
    pair_ids: np.ndarray,
    labels: np.ndarray,
    projections: np.ndarray,
) -> dict[str, float | int]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, label, projection in zip(pair_ids, labels, projections, strict=True):
        grouped[str(pair_id)][str(label)].append(float(projection))

    outcomes: list[float] = []
    margins: list[float] = []
    skipped = 0
    ties = 0
    for label_scores in grouped.values():
        if "positive" not in label_scores or "negative" not in label_scores:
            skipped += 1
            continue
        margin = float(np.mean(label_scores["positive"])) - float(
            np.mean(label_scores["negative"])
        )
        margins.append(margin)
        outcomes.append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)
        ties += 1 if margin == 0 else 0
    return _metric_summary(outcomes, margins, skipped=skipped, ties=ties)


def _activation_pair_folds(payload: ActivationPayload) -> list[tuple[str, set[str]]]:
    return [
        (pair_id, {pair_id})
        for pair_id in sorted(_complete_activation_pairs(payload.pair_ids, payload.labels))
    ]


def _activation_scenario_folds(
    payload: ActivationPayload,
    pair_scenarios: Mapping[str, str],
) -> list[tuple[str, set[str]]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair_id in _complete_activation_pairs(payload.pair_ids, payload.labels):
        scenario = pair_scenarios.get(pair_id)
        if scenario:
            grouped[scenario].add(pair_id)
    if len(grouped) < 2:
        return []
    return sorted(grouped.items())


def _complete_activation_pairs(pair_ids: np.ndarray, labels: np.ndarray) -> list[str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair_id, label in zip(pair_ids, labels, strict=True):
        grouped[str(pair_id)].add(str(label))
    return [
        pair_id
        for pair_id, pair_labels in grouped.items()
        if {"positive", "negative"}.issubset(pair_labels)
    ]


def _pair_mask(pair_ids: np.ndarray, selected_pair_ids: set[str]) -> np.ndarray:
    return np.asarray(
        [str(pair_id) in selected_pair_ids for pair_id in pair_ids],
        dtype=bool,
    )


def _metric_summary(
    outcomes: Sequence[float],
    margins: Sequence[float],
    *,
    skipped: int,
    ties: int,
) -> dict[str, float | int]:
    accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
    return {
        "n_pairs": len(outcomes),
        "skipped_pairs": skipped,
        "accuracy": round(float(accuracy), 6),
        "ties": ties,
        "mean_margin": _rounded_mean(margins),
        "min_margin": round(float(min(margins)), 6) if margins else 0.0,
        "max_margin": round(float(max(margins)), 6) if margins else 0.0,
    }


def _weighted_accuracy(
    results: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, float | int]:
    pair_count = 0
    correct = 0.0
    for result in results:
        metrics = _mapping(result[key])
        n_pairs = int(metrics.get("n_pairs", 0))
        pair_count += n_pairs
        correct += float(metrics.get("accuracy", 0.0)) * n_pairs
    accuracy = correct / pair_count if pair_count else 0.0
    return {"n_pairs": pair_count, "accuracy": round(accuracy, 6)}


def _feature_direction(
    positives: Sequence[np.ndarray],
    negatives: Sequence[np.ndarray],
) -> np.ndarray:
    direction = np.vstack(positives).mean(axis=0) - np.vstack(negatives).mean(axis=0)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > _EPSILON else np.zeros_like(direction)


def _metric_value(metrics: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _pair_runs(
    pair: PairwiseExample,
    run_index: Mapping[str, ScoredRun],
) -> tuple[ScoredRun, ScoredRun] | None:
    positive = run_index.get(pair.positive_run_id)
    negative = run_index.get(pair.negative_run_id)
    if positive is None or negative is None:
        return None
    return positive, negative


def _kind_from_text(text: str) -> str:
    match = _KIND_RE.search(text)
    return match.group(1) if match else ""


def _scenario_prefix(pair_id: str) -> str:
    return pair_id.split("::", 1)[0]


def _activation_matrix(raw: np.ndarray) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Activation array must be two-dimensional.")
    return matrix


def _npz_array(data: Any, keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"Activation file is missing one of: {', '.join(keys)}")


def _rounded_mean(values: Sequence[float]) -> float:
    return round(float(np.mean(values)), 6) if values else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _summary_rows(value: object) -> list[str]:
    summaries = value if isinstance(value, list) else []
    if not summaries:
        return ["| n/a | n/a | 0 | 0 | 0.000 | 0.000 |", ""]
    rows = []
    for summary in summaries:
        item = _mapping(summary)
        rows.append(
            "| {split} | {baseline} | {folds} | {pairs} | {test:.3f} | {train:.3f} |".format(
                split=item.get("split", ""),
                baseline=item.get("baseline", ""),
                folds=item.get("folds", 0),
                pairs=item.get("test_pairs", 0),
                test=float(item.get("mean_test_accuracy", 0.0)),
                train=float(item.get("mean_train_accuracy", 0.0)),
            )
        )
    rows.append("")
    return rows


def _fold_table(title: str, value: object) -> list[str]:
    folds = value if isinstance(value, list) else []
    lines = [
        f"## {title}",
        "",
        "| Held out | Baseline | Train pairs | Train acc | Test pairs | Test acc |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not folds:
        lines.extend(["| n/a | n/a | 0 | 0.000 | 0 | 0.000 |", ""])
        return lines
    for fold in folds:
        item = _mapping(fold)
        train = _mapping(item.get("train"))
        test = _mapping(item.get("test"))
        lines.append(
            "| {held_out} | {baseline} | {train_n} | {train_acc:.3f} | {test_n} | {test_acc:.3f} |".format(
                held_out=item.get("held_out", ""),
                baseline=item.get("baseline", ""),
                train_n=train.get("n_pairs", 0),
                train_acc=float(train.get("accuracy", 0.0)),
                test_n=test.get("n_pairs", 0),
                test_acc=float(test.get("accuracy", 0.0)),
            )
        )
    lines.append("")
    return lines


def _activation_lines(value: object) -> list[str]:
    if value is None:
        return ["## Activation Transfer", "", "No activation `.npz` file was evaluated.", ""]
    activation = _mapping(value)
    lines = [
        "## Activation Transfer",
        "",
        f"- Activation file: `{Path(str(activation.get('activation_npz', ''))).name}`",
        f"- Prompts: {activation.get('n_prompts', 0)}",
        f"- Activation dim: {activation.get('activation_dim', 0)}",
        "",
    ]
    lines.extend(_activation_fold_rows("Leave-one-pair-out", activation))
    lines.extend(_activation_fold_rows("Leave-one-scenario-out", activation))
    lines.extend(_activation_fold_rows("Pair-set transfer", activation))
    return lines


def _activation_fold_rows(title: str, activation: Mapping[str, Any]) -> list[str]:
    if "Pair-set" in title:
        key = "pair_set_transfer"
    else:
        key = "leave_one_pair_out" if "pair" in title else "leave_one_scenario_out"
    rows = [
        f"### {title}",
        "",
        "| Held out | Train pairs | Train acc | Test pairs | Test acc | Mean test margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    folds = activation.get(key)
    if not isinstance(folds, list) or not folds:
        rows.extend(["| n/a | 0 | 0.000 | 0 | 0.000 | 0.000 |", ""])
        return rows
    for fold in folds:
        item = _mapping(fold)
        train = _mapping(item.get("train"))
        test = _mapping(item.get("test"))
        rows.append(
            "| {held_out} | {train_n} | {train_acc:.3f} | {test_n} | {test_acc:.3f} | {margin:+.4f} |".format(
                held_out=item.get("held_out", ""),
                train_n=train.get("n_pairs", 0),
                train_acc=float(train.get("accuracy", 0.0)),
                test_n=test.get("n_pairs", 0),
                test_acc=float(test.get("accuracy", 0.0)),
                margin=float(test.get("mean_margin", 0.0)),
            )
        )
    rows.append("")
    return rows


def scenario_counts(pairs: Sequence[PairwiseExample]) -> dict[str, int]:
    return dict(Counter(pair.scenario_id for pair in pairs))
