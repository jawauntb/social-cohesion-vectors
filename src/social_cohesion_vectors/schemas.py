"""Canonical schemas shared by the simulation and activation pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScenarioKind = Literal[
    "iterated_prisoners_dilemma",
    "public_goods",
    "negotiation",
    "dialogue_repair",
    "resource_allocation",
]

InterventionKind = Literal[
    "none",
    "shared_identity",
    "perspective_taking",
    "reciprocity",
    "restorative_accountability",
    "truth_first",
]

StrategyProfile = Literal["cooperative", "self_protective", "adversarial"]
ScoreComponentName = Literal[
    "cooperation",
    "repair",
    "fairness",
    "hostility_inverse",
    "truthfulness",
    "autonomy_safety",
    "slack_preservation",
]


class AgentProfile(BaseModel):
    id: str
    role: str
    goal: str
    risk: str


class Scenario(BaseModel):
    id: str
    kind: ScenarioKind
    title: str
    setting: str
    dilemma: str
    agents: list[AgentProfile]
    rounds: int = Field(default=5, ge=1, le=20)
    cooperate_action: str
    defect_action: str
    success_metrics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RoundEvent(BaseModel):
    round_index: int
    actor_id: str
    action: Literal["cooperate", "defect"]
    message: str
    payoff_delta: float


class SimulationRun(BaseModel):
    run_id: str
    scenario_id: str
    intervention: InterventionKind
    strategy_profile: StrategyProfile
    seed: int
    transcript: str
    events: list[RoundEvent]
    metrics: dict[str, float]


class ScoredRun(SimulationRun):
    cohesion_score: float
    score_components: dict[str, float]


class PairwiseExample(BaseModel):
    pair_id: str
    scenario_id: str
    positive_run_id: str
    negative_run_id: str
    positive_text: str
    negative_text: str
    positive_score: float
    negative_score: float
    metadata: dict[str, str | float] = Field(default_factory=dict)


class ActivationPrompt(BaseModel):
    sample_id: str
    pair_id: str
    label: Literal["positive", "negative"]
    target_score: float
    text: str
