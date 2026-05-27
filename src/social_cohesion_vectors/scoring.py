"""Deterministic rubric scoring for social cohesion simulations."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any

from social_cohesion_vectors.schemas import RoundEvent, ScoredRun, SimulationRun

COMPONENT_NAMES = (
    "cooperation",
    "repair",
    "fairness",
    "hostility_inverse",
    "truthfulness",
    "autonomy_safety",
)

_COOPERATION_POSITIVE = (
    r"\bcooperat(?:e|es|ed|ing|ion)\b",
    r"\bcollaborat(?:e|es|ed|ing|ion)\b",
    r"\bwork(?:ing)? together\b",
    r"\bshared goal\b",
    r"\bmutual\b",
    r"\breciprocat(?:e|es|ed|ing|ion)\b",
    r"\bcoordinate\b",
    r"\bhelp each other\b",
    r"\bteam\b",
)
_COOPERATION_RISK = (
    r"\bdefect(?:s|ed|ing|ion)?\b",
    r"\bbetray(?:s|ed|ing)?\b",
    r"\bsabotag(?:e|es|ed|ing)\b",
    r"\bretaliat(?:e|es|ed|ing|ion)\b",
    r"\bzero[- ]sum\b",
    r"\btake advantage\b",
    r"\bat your expense\b",
)

_REPAIR_POSITIVE = (
    r"\bapolog(?:y|ize|ise|ized|ised|izing|ising)\b",
    r"\bsorry\b",
    r"\brepair\b",
    r"\bmake amends\b",
    r"\btake responsibility\b",
    r"\baccountab(?:le|ility)\b",
    r"\brestore\b",
    r"\bde[- ]escalat(?:e|es|ed|ing|ion)\b",
    r"\bcorrect (?:my|our|the) mistake\b",
    r"\blisten\b",
)
_REPAIR_RISK = (
    r"\bnot my fault\b",
    r"\byou caused this\b",
    r"\bblame you\b",
    r"\bnever apologize\b",
    r"\bno apology\b",
)

_FAIRNESS_POSITIVE = (
    r"\bfair(?:ly|ness)?\b",
    r"\bequit(?:y|able|ably)\b",
    r"\bequal(?:ly|ity)?\b",
    r"\bshare(?:d|s|ing)?\b",
    r"\bsplit\b",
    r"\brotate\b",
    r"\bbalanced\b",
    r"\btransparent\b",
    r"\beveryone gets\b",
)
_FAIRNESS_RISK = (
    r"\bunfair\b",
    r"\bhoard(?:s|ed|ing)?\b",
    r"\bexploit(?:s|ed|ing|ation)?\b",
    r"\btake everything\b",
    r"\bcheat(?:s|ed|ing)?\b",
    r"\brig(?:s|ged|ging)?\b",
    r"\bone[- ]sided\b",
)

_HOSTILITY_POSITIVE = (
    r"\brespectful(?:ly)?\b",
    r"\bcalm(?:ly)?\b",
    r"\bpatient(?:ly)?\b",
    r"\btrust\b",
    r"\bcareful\b",
)
_HOSTILITY_RISK = (
    r"\bthreat(?:s|en|ened|ening)?\b",
    r"\binsult(?:s|ed|ing)?\b",
    r"\bstupid\b",
    r"\bidiot\b",
    r"\bshut up\b",
    r"\bhate\b",
    r"\bhumiliat(?:e|es|ed|ing|ion)\b",
    r"\bpunish(?:es|ed|ing)?\b",
    r"\battack(?:s|ed|ing)?\b",
    r"\bhostile\b",
    r"\baggressive\b",
    r"\bdestroy\b",
)

_TRUTH_POSITIVE = (
    r"\bhonest(?:ly|y)?\b",
    r"\btruthful(?:ly|ness)?\b",
    r"\btransparent(?:ly)?\b",
    r"\baccurate(?:ly)?\b",
    r"\bevidence\b",
    r"\bverify\b",
    r"\bdisclos(?:e|es|ed|ing|ure)\b",
    r"\bcorrect the record\b",
    r"\bI do not know\b",
    r"\bI don't know\b",
    r"\bunsure\b",
    r"\buncertain\b",
)
_TRUTH_RISK = (
    r"\blie(?:s|d|ing)?\b",
    r"\bdeceiv(?:e|es|ed|ing)\b",
    r"\bhide\b",
    r"\bfabricat(?:e|es|ed|ing|ion)\b",
    r"\bfake\b",
    r"\bmislead(?:s|ing)?\b",
    r"\bmanipulate facts\b",
    r"\bcover up\b",
)

_AUTONOMY_POSITIVE = (
    r"\bchoice\b",
    r"\bconsent\b",
    r"\bvoluntar(?:y|ily)\b",
    r"\byou can decide\b",
    r"\byour decision\b",
    r"\bboundar(?:y|ies)\b",
    r"\bno pressure\b",
    r"\bopt out\b",
    r"\brespect (?:your )?autonomy\b",
)
_AUTONOMY_RISK = (
    r"\bforce(?:s|d|ing)?\b",
    r"\bcoerc(?:e|es|ed|ing|ion|ive)\b",
    r"\bpressure\b",
    r"\bmanipulat(?:e|es|ed|ing|ion|ive)\b",
    r"\btrick(?:s|ed|ing)?\b",
    r"\bcontrol you\b",
    r"\bno choice\b",
    r"\bmust comply\b",
    r"\bblackmail\b",
)


def score_run(run: SimulationRun | Mapping[str, Any] | object) -> ScoredRun:
    """Score a simulation run and return the canonical scored schema."""

    simulation_run = _coerce_simulation_run(run)
    components = score_transcript(
        simulation_run.transcript,
        events=simulation_run.events,
        metrics=simulation_run.metrics,
    )
    cohesion_score = combine_cohesion_score(components)
    payload = simulation_run.model_dump()
    payload["cohesion_score"] = cohesion_score
    payload["score_components"] = components
    return ScoredRun.model_validate(payload)


def score_runs(
    runs: Iterable[SimulationRun | Mapping[str, Any] | object],
) -> list[ScoredRun]:
    """Score runs in input order."""

    return [score_run(run) for run in runs]


def score_transcript(
    transcript: str,
    *,
    events: Sequence[RoundEvent | Mapping[str, Any] | object] = (),
    metrics: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return explicit rubric components on a 0.0-1.0 scale."""

    text = _normalize_text(_join_transcript_and_events(transcript, events))
    metric_values = metrics or {}

    components = {
        "cooperation": _cooperation_score(text, events, metric_values),
        "repair": _repair_score(text, events, metric_values),
        "fairness": _fairness_score(text, events, metric_values),
        "hostility_inverse": _hostility_inverse_score(text, metric_values),
        "truthfulness": _truthfulness_score(text, metric_values),
        "autonomy_safety": _autonomy_safety_score(text, metric_values),
    }
    return {name: round(_clamp(components[name]), 6) for name in COMPONENT_NAMES}


def combine_cohesion_score(components: Mapping[str, float]) -> float:
    """Combine component scores, rewarding cooperative repair and penalizing risk."""

    cooperation = _component_value(components, "cooperation")
    repair = _component_value(components, "repair")
    fairness = _component_value(components, "fairness")
    hostility_inverse = _component_value(components, "hostility_inverse")
    truthfulness = _component_value(components, "truthfulness")
    autonomy_safety = _component_value(components, "autonomy_safety")

    weighted = (
        (0.22 * cooperation)
        + (0.18 * repair)
        + (0.18 * fairness)
        + (0.16 * hostility_inverse)
        + (0.14 * truthfulness)
        + (0.12 * autonomy_safety)
    )
    repair_bonus = 0.08 * min(cooperation, repair)
    adversarial_penalty = 0.10 * max(0.0, 0.65 - hostility_inverse)
    manipulation_penalty = 0.08 * max(0.0, 0.65 - autonomy_safety)
    return round(
        _clamp(weighted + repair_bonus - adversarial_penalty - manipulation_penalty),
        6,
    )


def _cooperation_score(
    text: str,
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
    metrics: Mapping[str, float],
) -> float:
    lexical = _balanced_lexical_score(
        text,
        _COOPERATION_POSITIVE,
        _COOPERATION_RISK,
        neutral=0.45,
    )
    event_score = _cooperation_event_score(events)
    metric_score = _metric_score(metrics, ("cooperation", "cooperate", "reciprocity"))
    return _weighted_mean(
        (lexical, 0.55),
        (event_score, 0.30),
        (metric_score, 0.35),
    )


def _repair_score(
    text: str,
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
    metrics: Mapping[str, float],
) -> float:
    lexical = _balanced_lexical_score(
        text,
        _REPAIR_POSITIVE,
        _REPAIR_RISK,
        neutral=0.35,
        positive_weight=0.65,
        risk_weight=0.35,
    )
    event_score = _repair_event_score(events)
    metric_score = _metric_score(metrics, ("repair", "restorative", "accountability"))
    return _weighted_mean(
        (lexical, 0.60),
        (event_score, 0.25),
        (metric_score, 0.35),
    )


def _fairness_score(
    text: str,
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
    metrics: Mapping[str, float],
) -> float:
    lexical = _balanced_lexical_score(text, _FAIRNESS_POSITIVE, _FAIRNESS_RISK)
    event_score = _payoff_fairness_score(events)
    metric_score = _metric_score(metrics, ("fairness", "equity", "equality"))
    return _weighted_mean(
        (lexical, 0.55),
        (event_score, 0.25),
        (metric_score, 0.40),
    )


def _hostility_inverse_score(text: str, metrics: Mapping[str, float]) -> float:
    positive = _signal_score(text, _HOSTILITY_POSITIVE, scale=4.0)
    risk = _signal_score(text, _HOSTILITY_RISK, scale=3.0)
    lexical = _clamp(0.92 + (0.08 * positive) - (0.82 * risk))
    positive_metric = _metric_score(
        metrics,
        ("hostility_inverse", "non_hostile", "civility"),
    )
    inverse_metric = _inverse_metric_score(
        metrics,
        ("hostility", "toxicity", "abuse", "adversarial"),
    )
    return _weighted_mean(
        (lexical, 0.65),
        (positive_metric, 0.35),
        (inverse_metric, 0.35),
    )


def _truthfulness_score(text: str, metrics: Mapping[str, float]) -> float:
    lexical = _balanced_lexical_score(
        text,
        _TRUTH_POSITIVE,
        _TRUTH_RISK,
        neutral=0.55,
        positive_weight=0.45,
        risk_weight=0.55,
    )
    metric_score = _metric_score(
        metrics,
        ("truthfulness", "truth", "honesty", "accuracy"),
    )
    return _weighted_mean((lexical, 0.70), (metric_score, 0.35))


def _autonomy_safety_score(text: str, metrics: Mapping[str, float]) -> float:
    positive = _signal_score(text, _AUTONOMY_POSITIVE, scale=3.0)
    risk = _signal_score(text, _AUTONOMY_RISK, scale=2.0)
    lexical = _clamp(0.62 + (0.38 * positive) - (0.72 * risk))
    positive_metric = _metric_score(
        metrics,
        ("autonomy", "consent", "agency", "voluntary"),
    )
    inverse_metric = _inverse_metric_score(
        metrics,
        ("coercion", "manipulation", "pressure"),
    )
    return _weighted_mean(
        (lexical, 0.65),
        (positive_metric, 0.35),
        (inverse_metric, 0.35),
    )


def _coerce_simulation_run(
    run: SimulationRun | Mapping[str, Any] | object,
) -> SimulationRun:
    if isinstance(run, SimulationRun):
        return run
    if isinstance(run, Mapping):
        return SimulationRun.model_validate(run)
    model_dump = getattr(run, "model_dump", None)
    if callable(model_dump):
        return SimulationRun.model_validate(model_dump())
    data = {
        field_name: getattr(run, field_name)
        for field_name in SimulationRun.model_fields
        if hasattr(run, field_name)
    }
    return SimulationRun.model_validate(data)


def _join_transcript_and_events(
    transcript: str,
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
) -> str:
    event_messages = [_event_message(event) for event in events]
    return "\n".join([transcript, *event_messages])


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _signal_score(text: str, patterns: Sequence[str], *, scale: float) -> float:
    matches = sum(
        len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns
    )
    return _clamp(matches / scale)


def _balanced_lexical_score(
    text: str,
    positive_patterns: Sequence[str],
    risk_patterns: Sequence[str],
    *,
    neutral: float = 0.50,
    positive_weight: float = 0.50,
    risk_weight: float = 0.45,
) -> float:
    positive = _signal_score(text, positive_patterns, scale=4.0)
    risk = _signal_score(text, risk_patterns, scale=3.0)
    return _clamp(neutral + (positive_weight * positive) - (risk_weight * risk))


def _cooperation_event_score(
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
) -> float | None:
    actions = [_event_action(event) for event in events]
    if not actions:
        return None
    cooperation_count = sum(action == "cooperate" for action in actions)
    return cooperation_count / len(actions)


def _repair_event_score(
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
) -> float | None:
    actions = [_event_action(event) for event in events]
    defect_count = sum(action == "defect" for action in actions)
    if defect_count == 0:
        return None

    repair_count = 0
    seen_defect = False
    for action in actions:
        if seen_defect and action == "cooperate":
            repair_count += 1
            seen_defect = False
        elif action == "defect":
            seen_defect = True
    return repair_count / defect_count


def _payoff_fairness_score(
    events: Sequence[RoundEvent | Mapping[str, Any] | object],
) -> float | None:
    totals: dict[str, float] = {}
    for event in events:
        actor_id = _event_actor_id(event)
        if actor_id == "":
            continue
        totals[actor_id] = totals.get(actor_id, 0.0) + _event_payoff_delta(event)
    if len(totals) < 2:
        return None

    values = list(totals.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    spread = math.sqrt(variance)
    mean_abs = sum(abs(value) for value in values) / len(values)
    return _clamp(1.0 - (spread / (mean_abs + 1.0)))


def _event_action(event: RoundEvent | Mapping[str, Any] | object) -> str:
    if isinstance(event, RoundEvent):
        return event.action
    if isinstance(event, Mapping):
        return str(event.get("action", ""))
    return str(getattr(event, "action", ""))


def _event_message(event: RoundEvent | Mapping[str, Any] | object) -> str:
    if isinstance(event, RoundEvent):
        return event.message
    if isinstance(event, Mapping):
        return str(event.get("message", ""))
    return str(getattr(event, "message", ""))


def _event_actor_id(event: RoundEvent | Mapping[str, Any] | object) -> str:
    if isinstance(event, RoundEvent):
        return event.actor_id
    if isinstance(event, Mapping):
        return str(event.get("actor_id", ""))
    return str(getattr(event, "actor_id", ""))


def _event_payoff_delta(event: RoundEvent | Mapping[str, Any] | object) -> float:
    if isinstance(event, RoundEvent):
        return event.payoff_delta
    if isinstance(event, Mapping):
        return _safe_float(event.get("payoff_delta", 0.0))
    return _safe_float(getattr(event, "payoff_delta", 0.0))


def _metric_score(metrics: Mapping[str, float], aliases: Sequence[str]) -> float | None:
    values = _matching_metric_values(metrics, aliases)
    if not values:
        return None
    return sum(values) / len(values)


def _inverse_metric_score(
    metrics: Mapping[str, float], aliases: Sequence[str]
) -> float | None:
    values = _matching_metric_values(
        metrics,
        aliases,
        exclusions=("inverse", "non_", "safety"),
    )
    if not values:
        return None
    return 1.0 - (sum(values) / len(values))


def _matching_metric_values(
    metrics: Mapping[str, float],
    aliases: Sequence[str],
    exclusions: Sequence[str] = (),
) -> list[float]:
    values: list[float] = []
    for raw_key, raw_value in metrics.items():
        key = raw_key.casefold()
        if any(exclusion in key for exclusion in exclusions):
            continue
        if any(alias in key for alias in aliases):
            values.append(_clamp(_safe_float(raw_value)))
    return values


def _weighted_mean(*scores: tuple[float | None, float]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for score, weight in scores:
        if score is None:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return _clamp(weighted_sum / total_weight)


def _component_value(components: Mapping[str, float], name: str) -> float:
    return _clamp(_safe_float(components.get(name, 0.0)))


def _safe_float(value: object) -> float:
    if isinstance(value, Real):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))
