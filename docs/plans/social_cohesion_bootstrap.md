---
title: Social Cohesion Vector Bootstrap
status: active
created: 2026-05-27
origin: SOCIAL_COHESION_CONTENT_GENERATION_PROJECT.md
---

# Social Cohesion Vector Bootstrap

## Problem Frame

We want to push the project as far as possible before human trials or new neural
recordings. The useful near-term target is a research pipeline that can generate
social-dilemma text trajectories, score them, produce paired examples, and
extract open-model activations for probes, contrastive directions, and SAE work.

## Scope Boundaries

In scope:

- Offline social-dilemma simulation and transcript generation.
- Computational cohesion scoring with explicit ethics guardrails.
- Pairwise datasets for cooperative vs adversarial trajectories.
- Modal-ready open-weight activation extraction.
- Contrastive-vector training over saved activation features.
- Prolific-ready prompt/content artifacts later.

Out of scope for this bootstrap:

- New fMRI, EEG, fNIRS, or hyperscanning data collection.
- Claims that a computed vector changes real human cooperation without Prolific
  or equivalent behavioral validation.
- Coercive persuasion, ideological conformity, deception, or compliance
  optimization.

## Architecture Decisions

1. Keep an offline lane and a Modal lane.
   The offline lane makes progress immediately and produces data artifacts. The
   Modal lane handles GPU-heavy open-model activation extraction once credentials
   and model access are ready.

2. Treat cohesion as a rubric, not a scalar essence.
   A combined score is useful for ranking, but sub-scores stay visible:
   cooperation, repair, fairness, trust, hostility, truthfulness, and autonomy
   risk.

3. Make pairwise examples the central training unit.
   Positive/negative trajectory pairs support reward modeling, contrastive
   activation directions, SAE dataset curation, and Prolific comparisons.

4. Use open-weight activations as the first mechanistic target.
   Candidate models should start small enough for cheap Modal iteration, then
   graduate to larger instruction-tuned models once the plumbing is stable.

## Implementation Units

### U1: Project Scaffold

Files:

- `pyproject.toml`
- `README.md`
- `.env.example`
- `.gitignore`
- `src/social_cohesion_vectors/config.py`
- `src/social_cohesion_vectors/schemas.py`

Verification:

- Package imports under `uv run python`.
- `.env` is ignored and `.env.example` contains no secrets.

### U2: Scenario And Simulation Core

Files:

- `data/scenarios/seed_scenarios.json`
- `src/social_cohesion_vectors/scenario_library.py`
- `src/social_cohesion_vectors/simulations/simple_agents.py`
- `scripts/run_scenario_simulations.py`
- `tests/test_scenarios.py`
- `tests/test_simulations.py`

Verification:

- At least 20 scenarios load with unique IDs.
- Simulation output contains transcripts and bounded metrics.

### U3: Scoring And Pairwise Dataset

Files:

- `src/social_cohesion_vectors/scoring.py`
- `src/social_cohesion_vectors/datasets.py`
- `scripts/build_probe_dataset.py`
- `scripts/export_activation_prompts.py`
- `tests/test_scoring.py`
- `tests/test_datasets.py`

Verification:

- Cooperative repair text scores above hostile/adversarial text.
- Pairwise dataset picks higher-scoring runs as positives.

### U4: Activation And Modal Scaffold

Files:

- `src/social_cohesion_vectors/activations/contrastive.py`
- `src/social_cohesion_vectors/modal_app/app.py`
- `src/social_cohesion_vectors/modal_app/image_factory.py`
- `src/social_cohesion_vectors/modal_app/functions/activation_extractor.py`
- `tests/test_contrastive.py`

Verification:

- Contrastive direction projects positive synthetic features above negative
  synthetic features.
- Modal modules import locally without loading heavy ML dependencies.

## Test Plan

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`

## Follow-On Experiments

- Swap scripted agents for API-backed LLM agents and judge models.
- Add SAE export format once the target layer/model is selected.
- Run best-of-N intervention generation and score candidates with the rubric.
- Prepare a small Prolific study comparing high-scoring vs low-scoring
  interventions on trust, willingness to cooperate, and manipulation concern.
