"""Dataset builders and JSONL export helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from social_cohesion_vectors.schemas import (
    ActivationPrompt,
    PairwiseExample,
    ScoredRun,
    SimulationRun,
)
from social_cohesion_vectors.scoring import score_run


def load_simulation_runs_jsonl(path: str | Path) -> list[SimulationRun]:
    """Load simulation runs from JSONL."""

    return [SimulationRun.model_validate(record) for record in read_jsonl(path)]


def load_scored_runs_jsonl(path: str | Path) -> list[ScoredRun]:
    """Load scored simulation runs from JSONL."""

    return [ScoredRun.model_validate(record) for record in read_jsonl(path)]


def load_pairwise_examples_jsonl(path: str | Path) -> list[PairwiseExample]:
    """Load pairwise examples from JSONL."""

    return [PairwiseExample.model_validate(record) for record in read_jsonl(path)]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                msg = f"JSONL line {line_number} in {path} is not an object"
                raise ValueError(msg)
            records.append(record)
    return records


def write_jsonl(
    records: Iterable[BaseModel | Mapping[str, Any]], path: str | Path
) -> int:
    """Write pydantic models or mappings to JSONL and return the record count."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                handle.write(record.model_dump_json())
            else:
                handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def build_pairwise_examples(
    runs: Iterable[ScoredRun | SimulationRun | Mapping[str, Any] | object],
    *,
    min_margin: float = 0.15,
    max_pairs_per_scenario: int | None = None,
) -> list[PairwiseExample]:
    """Group scored runs by scenario and pair high-vs-low examples."""

    grouped: dict[str, list[ScoredRun]] = defaultdict(list)
    for run in runs:
        scored = _coerce_scored_run(run)
        grouped[scored.scenario_id].append(scored)

    examples: list[PairwiseExample] = []
    for scenario_id in sorted(grouped):
        ordered = sorted(
            grouped[scenario_id],
            key=lambda item: (-item.cohesion_score, item.run_id),
        )
        examples.extend(
            _pair_scenario_runs(
                scenario_id,
                ordered,
                min_margin=min_margin,
                max_pairs=max_pairs_per_scenario,
            )
        )
    return examples


def activation_prompts_from_pairs(
    pairs: Iterable[PairwiseExample | Mapping[str, Any]],
) -> list[ActivationPrompt]:
    """Create positive and negative activation prompts for each pair."""

    prompts: list[ActivationPrompt] = []
    for raw_pair in pairs:
        pair = _coerce_pairwise_example(raw_pair)
        prompts.extend(
            [
                ActivationPrompt(
                    sample_id=f"{pair.pair_id}:positive",
                    pair_id=pair.pair_id,
                    label="positive",
                    target_score=pair.positive_score,
                    text=pair.positive_text,
                ),
                ActivationPrompt(
                    sample_id=f"{pair.pair_id}:negative",
                    pair_id=pair.pair_id,
                    label="negative",
                    target_score=pair.negative_score,
                    text=pair.negative_text,
                ),
            ]
        )
    return prompts


def export_pairwise_jsonl(
    pairs: Iterable[PairwiseExample | Mapping[str, Any]],
    path: str | Path,
) -> int:
    """Export pairwise examples to JSONL."""

    coerced = [_coerce_pairwise_example(pair) for pair in pairs]
    return write_jsonl(coerced, path)


def export_activation_prompts_jsonl(
    pairs_or_prompts: Iterable[PairwiseExample | ActivationPrompt | Mapping[str, Any]],
    path: str | Path,
) -> int:
    """Export activation prompt records matching the canonical schema."""

    prompts: list[ActivationPrompt] = []
    for record in pairs_or_prompts:
        if isinstance(record, ActivationPrompt):
            prompts.append(record)
            continue
        if _looks_like_activation_prompt(record):
            prompts.append(ActivationPrompt.model_validate(record))
            continue
        prompts.extend(activation_prompts_from_pairs([record]))
    return write_jsonl(prompts, path)


def _pair_scenario_runs(
    scenario_id: str,
    ordered_runs: Sequence[ScoredRun],
    *,
    min_margin: float,
    max_pairs: int | None,
) -> list[PairwiseExample]:
    examples: list[PairwiseExample] = []
    low_index = len(ordered_runs) - 1
    high_index = 0

    while high_index < low_index:
        positive = ordered_runs[high_index]
        negative = ordered_runs[low_index]
        margin = positive.cohesion_score - negative.cohesion_score
        if margin < min_margin:
            break
        examples.append(
            _build_pairwise_example(scenario_id, positive, negative, margin)
        )
        if max_pairs is not None and len(examples) >= max_pairs:
            break
        high_index += 1
        low_index -= 1
    return examples


def _build_pairwise_example(
    scenario_id: str,
    positive: ScoredRun,
    negative: ScoredRun,
    margin: float,
) -> PairwiseExample:
    pair_id = "::".join(
        [
            _slug(scenario_id),
            _slug(positive.run_id),
            "over",
            _slug(negative.run_id),
        ]
    )
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=scenario_id,
        positive_run_id=positive.run_id,
        negative_run_id=negative.run_id,
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
        metadata={
            "score_margin": round(margin, 6),
            "positive_intervention": positive.intervention,
            "negative_intervention": negative.intervention,
            "positive_strategy": positive.strategy_profile,
            "negative_strategy": negative.strategy_profile,
            "positive_seed": float(positive.seed),
            "negative_seed": float(negative.seed),
        },
    )


def _coerce_scored_run(
    run: ScoredRun | SimulationRun | Mapping[str, Any] | object,
) -> ScoredRun:
    if isinstance(run, ScoredRun):
        return run
    if (
        isinstance(run, Mapping)
        and "cohesion_score" in run
        and "score_components" in run
    ):
        return ScoredRun.model_validate(run)
    return score_run(run)


def _coerce_pairwise_example(
    pair: PairwiseExample | Mapping[str, Any],
) -> PairwiseExample:
    if isinstance(pair, PairwiseExample):
        return pair
    return PairwiseExample.model_validate(pair)


def _looks_like_activation_prompt(record: object) -> bool:
    if isinstance(record, ActivationPrompt):
        return True
    if not isinstance(record, Mapping):
        return False
    return {"sample_id", "pair_id", "label", "target_score", "text"}.issubset(record)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "item"
