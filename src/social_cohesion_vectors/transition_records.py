"""Lane-agnostic perturbation transition records for CK cocktail reports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl


def transition_records_from_report(
    report: Mapping[str, Any],
    *,
    baseline_recipe_id: str | None = None,
    source_id: str = "",
) -> list[dict[str, Any]]:
    """Convert a supported report dictionary into transition records."""

    if _is_toy_substrate_report(report):
        return transition_records_from_toy_substrate_report(
            report,
            source_id=source_id,
        )
    return transition_records_from_ck_report(
        report,
        baseline_recipe_id=baseline_recipe_id,
        source_id=source_id,
    )


def transition_records_from_ck_report(
    report: Mapping[str, Any],
    *,
    baseline_recipe_id: str | None = None,
    source_id: str = "",
) -> list[dict[str, Any]]:
    """Convert a CK cocktail report dictionary into transition records."""

    records = _sequence_of_mappings(report.get("records"))
    summary = _mapping(report.get("summary"))
    baseline_id = baseline_recipe_id or str(
        summary.get("baseline_recipe_id") or "baseline"
    )
    context = {
        "source_id": source_id,
        "source_experiment": str(report.get("experiment", "")),
        "source_description": str(report.get("description", "")),
        "source_record_count": len(records),
    }
    return transition_records_from_ck_records(
        records,
        baseline_recipe_id=baseline_id,
        replication_context=context,
    )


def transition_records_from_ck_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_recipe_id: str = "baseline",
    replication_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Pair each non-baseline CK record with its prompt-matched baseline."""

    baselines = {
        str(record.get("prompt_id", "")): record
        for record in records
        if str(record.get("recipe_id", "")) == baseline_recipe_id
    }
    transitions: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            str(item.get("prompt_id", "")),
            str(item.get("recipe_id", "")),
        ),
    ):
        recipe_id = str(record.get("recipe_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        if recipe_id == baseline_recipe_id:
            continue
        baseline = baselines.get(prompt_id)
        if baseline is None:
            continue
        transitions.append(
            _transition_record(
                baseline,
                record,
                baseline_recipe_id=baseline_recipe_id,
                replication_context=replication_context or {},
            )
        )
    return transitions


def transition_records_from_toy_substrate_report(
    report: Mapping[str, Any],
    *,
    source_id: str = "",
) -> list[dict[str, Any]]:
    """Convert a toy substrate sweep report into transition records."""

    results = _sequence_of_mappings(report.get("results"))
    summary = _mapping(report.get("summary"))
    graph = _mapping(report.get("graph"))
    context = {
        "source_id": source_id,
        "source_experiment": str(report.get("experiment", "")),
        "source_record_count": len(results),
        "graph_id": str(summary.get("graph_id") or graph.get("graph_id", "")),
        "claim_boundary": str(report.get("claim_boundary", "")),
        "substrate": "toy_graph",
    }
    transitions: list[dict[str, Any]] = []
    for group_results in _toy_result_groups(results).values():
        baseline = _toy_baseline_result(group_results)
        if baseline is None:
            continue
        for result in sorted(group_results, key=lambda item: str(item.get("run_id", ""))):
            if result is baseline or _is_toy_baseline_result(result):
                continue
            transitions.append(
                _toy_transition_record(
                    baseline,
                    result,
                    summary=summary,
                    replication_context=context,
                )
            )
    return transitions


def summarize_transition_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count transition records by effect, side-effect, and washout status."""

    summary: dict[str, dict[str, int]] = {
        "effect_class": {},
        "side_effect_status": {},
        "washout_status": {},
    }
    for record in records:
        _increment(summary["effect_class"], str(record.get("effect_class", "")))
        side_effects = _mapping(record.get("side_effects"))
        _increment(
            summary["side_effect_status"],
            "observed" if bool(side_effects.get("observed", False)) else "none",
        )
        washout = _mapping(record.get("washout"))
        _increment(
            summary["washout_status"],
            str(washout.get("status") or "unknown"),
        )
    return summary


def load_ck_records_or_report(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load CK input as either a JSON report/object/list or JSONL records."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        records = read_jsonl(input_path)
        return records, {}

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_records = payload.get("records", [])
        return _list_of_dicts(raw_records), dict(payload)
    if isinstance(payload, list):
        return _list_of_dicts(payload), {}
    raise ValueError(f"Unsupported CK transition input payload in {input_path}")


def write_transition_records(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> int:
    """Write transition records as deterministic JSONL."""

    return write_jsonl(records, path)


def _is_toy_substrate_report(report: Mapping[str, Any]) -> bool:
    experiment = str(report.get("experiment", ""))
    return "toy_substrate" in experiment or "results" in report


def _transition_record(
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    baseline_recipe_id: str,
    replication_context: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_id = str(observed.get("prompt_id", baseline.get("prompt_id", "")))
    recipe_id = str(observed.get("recipe_id", ""))
    components = _component_records(observed.get("components"))
    observed_delta = _score_delta(observed, baseline, "ck1_score")
    safe_delta = _component_delta(observed, baseline, "safe_attunement_signal")
    pseudo_delta = _component_delta(observed, baseline, "pseudo_attunement_risk")
    return {
        "transition_id": f"{prompt_id}::{baseline_recipe_id}->{recipe_id}",
        "baseline_state": _baseline_state(baseline),
        "perturbation": {
            "recipe_id": recipe_id,
            "recipe_label": str(observed.get("recipe_label", recipe_id)),
            "components": components,
        },
        "dose": _dose(components),
        "site": _site(components),
        "timing": _timing(components),
        "effect_class": _effect_class(observed_delta, pseudo_delta),
        "observed_transition": {
            "from_recipe_id": baseline_recipe_id,
            "to_recipe_id": recipe_id,
            "ck1_score_delta": observed_delta,
            "safe_attunement_signal_delta": safe_delta,
            "pseudo_attunement_risk_delta": pseudo_delta,
            "from_generated_text": str(baseline.get("generated_text", "")),
            "to_generated_text": str(observed.get("generated_text", "")),
        },
        "side_effects": {
            "pseudo_attunement_risk": _component(observed, "pseudo_attunement_risk"),
            "pseudo_attunement_risk_delta": pseudo_delta,
            "observed": pseudo_delta > 0.0,
        },
        "antagonist": _antagonist(components),
        "washout": {
            "status": "not_measured",
            "note": (
                "CK cocktail reports are single-pass generation assays and do not "
                "measure reversibility or post-perturbation washout."
            ),
        },
        "replication_context": _replication_context(
            baseline,
            observed,
            replication_context,
        ),
    }


def _toy_transition_record(
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    replication_context: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(observed.get("run_id", ""))
    graph_id = str(observed.get("graph_id") or summary.get("graph_id", ""))
    intervention_id = str(observed.get("intervention_id", ""))
    coefficient = round(float(observed.get("coefficient", 0.0)), 6)
    transmitter = str(observed.get("transmitter") or summary.get("transmitter", ""))
    target_movement = round(float(observed.get("target_movement", 0.0)), 6)
    off_target_movement = round(float(observed.get("off_target_movement", 0.0)), 6)
    washout_value = round(float(observed.get("washout", 0.0)), 6)
    instability = bool(observed.get("instability_flag", False))
    return {
        "transition_id": f"{graph_id}::{_toy_run_id(baseline)}->{run_id}",
        "baseline_state": _toy_baseline_state(baseline, summary=summary),
        "perturbation": {
            "recipe_id": intervention_id,
            "recipe_label": f"{transmitter} edge scaling x{coefficient:g}",
            "components": [
                {
                    "component_id": intervention_id,
                    "coefficient": coefficient,
                    "transmitter": transmitter,
                    "source_class": observed.get("source_class"),
                    "scaled_edges": int(observed.get("scaled_edges", 0)),
                }
            ],
        },
        "dose": {
            "component_count": 1,
            "component_strengths": [
                {
                    "component_id": intervention_id,
                    "coefficient": coefficient,
                }
            ],
            "absolute_strength_sum": round(abs(coefficient - 1.0), 6),
        },
        "site": [
            {
                "component_id": intervention_id,
                "graph_id": graph_id,
                "target_nodes": list(_sequence_or_empty(observed.get("target_nodes"))),
                "scaled_edges": int(observed.get("scaled_edges", 0)),
            }
        ],
        "timing": [
            {
                "component_id": intervention_id,
                "steering_timing": "simulation_step",
                "steering_schedule": "constant",
            }
        ],
        "effect_class": _toy_effect_class(
            target_movement=target_movement,
            off_target_movement=off_target_movement,
            washout=washout_value,
            instability=instability,
        ),
        "observed_transition": {
            "from_recipe_id": str(baseline.get("intervention_id", "baseline")),
            "to_recipe_id": intervention_id,
            "target_movement": target_movement,
            "off_target_movement": off_target_movement,
            "selectivity_ratio": round(float(observed.get("selectivity_ratio", 0.0)), 6),
            "baseline_target": round(float(observed.get("baseline_target", 0.0)), 6),
            "perturbed_target": round(float(observed.get("perturbed_target", 0.0)), 6),
            "instability_flag": instability,
        },
        "side_effects": {
            "off_target_movement": off_target_movement,
            "instability_flag": instability,
            "observed": off_target_movement > 0.0 or instability,
        },
        "antagonist": {
            "component_ids": [],
            "inferred": False,
            "note": (
                "Toy substrate transition records do not infer biological "
                "receptor antagonists."
            ),
        },
        "washout": {
            "status": _toy_washout_status(washout_value),
            "value": washout_value,
            "note": (
                "Washout is measured as toy-graph distance from baseline after "
                "removing the scaling intervention."
            ),
        },
        "replication_context": _toy_replication_context(
            observed,
            replication_context,
        ),
    }


def _baseline_state(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": str(record.get("prompt_id", "")),
        "prompt": str(record.get("prompt") or record.get("text", "")),
        "phase": str(record.get("phase", "")),
        "mechanism": str(record.get("mechanism", "")),
        "recipe_id": str(record.get("recipe_id", "")),
        "recipe_label": str(record.get("recipe_label", "")),
        "generated_text": str(record.get("generated_text", "")),
        "ck1_score": round(float(record.get("ck1_score", 0.0)), 6),
        "score_components": dict(_mapping(record.get("score_components"))),
    }


def _component_records(value: object) -> list[dict[str, Any]]:
    components = []
    for component in _sequence_of_mappings(value):
        components.append(
            {
                "component_id": str(component.get("component_id", "")),
                "layer": _optional_int(component.get("layer")),
                "strength": round(float(component.get("strength", 0.0)), 6),
                "hook_site": str(component.get("hook_site", "")),
                "steering_position": str(component.get("steering_position", "")),
                "steering_timing": str(component.get("steering_timing", "")),
                "steering_schedule": _steering_schedule(component),
            }
        )
    return components


def _dose(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strengths = [
        {
            "component_id": str(component.get("component_id", "")),
            "strength": round(float(component.get("strength", 0.0)), 6),
        }
        for component in components
    ]
    return {
        "component_count": len(components),
        "component_strengths": strengths,
        "absolute_strength_sum": round(
            sum(abs(float(component["strength"])) for component in strengths),
            6,
        ),
    }


def _site(components: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "component_id": str(component.get("component_id", "")),
            "layer": component.get("layer"),
            "hook_site": str(component.get("hook_site", "")),
            "steering_position": str(component.get("steering_position", "")),
        }
        for component in components
    ]


def _timing(components: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "component_id": str(component.get("component_id", "")),
            "steering_timing": str(component.get("steering_timing", "")),
            "steering_schedule": _steering_schedule(component),
        }
        for component in components
    ]


def _antagonist(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    antagonist_ids = [
        str(component.get("component_id", ""))
        for component in components
        if _looks_like_antagonist(str(component.get("component_id", "")))
    ]
    return {
        "component_ids": antagonist_ids,
        "inferred": bool(antagonist_ids),
        "note": (
            "Antagonist labels are computational guardrail/control components, "
            "not biological receptor antagonists."
        ),
    }


def _replication_context(
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    extra_context: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "model_id",
        "seed",
        "max_new_tokens",
        "max_length",
        "hook_site",
        "steering_position",
        "steering_timing",
        "steering_schedule",
    )
    context = dict(extra_context)
    context.update(
        {
            "prompt_id": str(observed.get("prompt_id", baseline.get("prompt_id", ""))),
            "baseline_recipe_id": str(baseline.get("recipe_id", "")),
            "perturbation_recipe_id": str(observed.get("recipe_id", "")),
        }
    )
    for key in keys:
        if key in observed:
            context[key] = observed[key]
        elif key in baseline:
            context[key] = baseline[key]
    component_schedules = [
        {
            "component_id": str(component.get("component_id", "")),
            "steering_schedule": _steering_schedule(component),
        }
        for component in _sequence_of_mappings(observed.get("components"))
    ]
    if component_schedules:
        context["component_steering_schedules"] = component_schedules
    return context


def _toy_baseline_state(
    record: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prompt_id": str(record.get("graph_id") or summary.get("graph_id", "")),
        "prompt": "toy substrate baseline edge propagation",
        "phase": "toy_substrate",
        "mechanism": str(record.get("transmitter") or summary.get("transmitter", "")),
        "recipe_id": str(record.get("intervention_id", "baseline")),
        "recipe_label": "Baseline coefficient 1.0",
        "generated_text": "",
        "ck1_score": 0.0,
        "score_components": {
            "target_movement": round(float(record.get("target_movement", 0.0)), 6),
            "off_target_movement": round(
                float(record.get("off_target_movement", 0.0)),
                6,
            ),
            "washout": round(float(record.get("washout", 0.0)), 6),
        },
    }


def _toy_replication_context(
    observed: Mapping[str, Any],
    extra_context: Mapping[str, Any],
) -> dict[str, Any]:
    context = dict(extra_context)
    context.update(
        {
            "run_id": str(observed.get("run_id", "")),
            "intervention_id": str(observed.get("intervention_id", "")),
            "coefficient": round(float(observed.get("coefficient", 0.0)), 6),
            "transmitter": str(observed.get("transmitter", "")),
            "source_class": observed.get("source_class"),
        }
    )
    return context


def _effect_class(ck1_delta: float, pseudo_delta: float) -> str:
    if ck1_delta > 0.0 and pseudo_delta <= 0.0:
        return "beneficial_transition"
    if ck1_delta > 0.0 and pseudo_delta > 0.0:
        return "mixed_transition"
    if ck1_delta < 0.0:
        return "adverse_transition"
    return "neutral_transition"


def _toy_effect_class(
    *,
    target_movement: float,
    off_target_movement: float,
    washout: float,
    instability: bool,
) -> str:
    if instability:
        return "adverse_transition"
    if abs(target_movement) > 0.0 and off_target_movement <= 0.0 and washout <= 0.0:
        return "beneficial_transition"
    if abs(target_movement) > 0.0:
        return "mixed_transition"
    if off_target_movement > 0.0 or washout > 0.0:
        return "adverse_transition"
    return "neutral_transition"


def _toy_washout_status(washout: float) -> str:
    if washout <= 0.0:
        return "recovered"
    return "residual"


def _score_delta(
    observed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    key: str,
) -> float:
    return round(float(observed.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6)


def _component_delta(
    observed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    key: str,
) -> float:
    return round(_component(observed, key) - _component(baseline, key), 6)


def _component(record: Mapping[str, Any], key: str) -> float:
    return round(float(_mapping(record.get("score_components")).get(key, 0.0)), 6)


def _looks_like_antagonist(component_id: str) -> bool:
    lowered = component_id.lower()
    return (
        lowered.startswith("anti_")
        or "guardrail" in lowered
        or "truth" in lowered
        or "privacy" in lowered
        or "exit" in lowered
    )


def _steering_schedule(component: Mapping[str, Any]) -> str:
    value = component.get("steering_schedule", "constant")
    if value is None or value == "":
        return "constant"
    return str(value)


def _toy_baseline_result(
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for result in results:
        if _is_toy_baseline_result(result):
            return result
    return None


def _toy_result_groups(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str | None], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str, str | None], list[Mapping[str, Any]]] = {}
    for result in results:
        key = (
            str(result.get("graph_id", "")),
            str(result.get("transmitter", "")),
            _optional_string(result.get("source_class")),
        )
        groups.setdefault(key, []).append(result)
    return groups


def _is_toy_baseline_result(result: Mapping[str, Any]) -> bool:
    return float(result.get("coefficient", 0.0)) == 1.0


def _toy_run_id(record: Mapping[str, Any]) -> str:
    return str(record.get("run_id") or "baseline")


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


def _optional_string(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    return [dict(record) for record in _sequence_of_mappings(value)]


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sequence_or_empty(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return list(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
