# Social Cohesion Vectors Handoff

This repo is an early research scaffold for testing whether cooperative,
repair-oriented social behavior has measurable structure in text trajectories
and open-model activation space.

## What Works Now

The current local pipeline can:

1. Load 25 hand-authored social-dilemma scenarios.
2. Generate deterministic cooperative, self-protective, adversarial, and
   intervention-conditioned trajectories.
3. Score runs for cooperation, repair, fairness, hostility inverse, truthfulness,
   and autonomy safety.
4. Build pairwise high/low cohesion probe examples.
5. Export activation prompts.
6. Extract open-weight LLM activations on Modal/GPU.
7. Train/evaluate a contrastive activation direction.
8. Run simple baselines.

## Setup

```bash
uv sync --group dev
cp .env.example .env
```

Fill `.env` with Modal/Hugging Face/API keys as needed. The original local run
used `Qwen/Qwen2.5-0.5B-Instruct` on Modal.

## Reproduce The First Pipeline

```bash
uv run python scripts/run_scenario_simulations.py
uv run python scripts/build_probe_dataset.py
uv run python scripts/export_activation_prompts.py
uv run python scripts/run_baseline_experiments.py
```

Expected local artifact counts:

- `data/processed/simulation_runs.jsonl`: 450 rows
- `data/processed/scored_runs.jsonl`: 450 rows
- `data/training/pairwise_probe_dataset.jsonl`: 126 rows
- `data/training/activation_prompts.jsonl`: 252 rows

## Run The GPU Activation Lane

Smoke test:

```bash
uv run modal run -m social_cohesion_vectors.modal_app.functions.activation_extractor::smoke_extract \
  --max-records 2 --batch-size 2 --max-length 128
```

Full extraction:

```bash
uv run python scripts/run_modal_activation_extraction.py \
  --batch-size 8 \
  --max-length 512
```

Then train/evaluate the first vector:

```bash
uv run python scripts/run_activation_vector_experiment.py \
  'data/features/open_llm/activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz'
```

The initial local run produced a 252 x 896 activation matrix and a first
contrastive vector. On scripted data, in-sample and leave-one-pair-out pairwise
accuracy were both 1.000. This is only a sanity check because lexical and
metrics-only baselines also solve the scripted task.

## Current Interpretation

The scaffold works. The science is not done.

The current dataset is intentionally easy:

- strategy-profile prior accuracy: 0.988
- metrics-only accuracy: 1.000
- lexical-only accuracy: 1.000
- activation direction leave-one-pair-out accuracy: 1.000

That means the next valuable experiment is not "celebrate 100%"; it is to make
the task harder.

## Next High-Value Experiments

1. Generate held-out LLM-authored trajectories with fewer obvious lexical cues.
2. Add pseudo-cohesion hard negatives:
   - polite coercion;
   - sycophantic agreement;
   - unity rhetoric that suppresses dissent;
   - truth-hiding "repair";
   - compliance framed as harmony.
3. Train on scripted data and test on generated/hard-negative data.
4. Sweep activation layers and model sizes.
5. Split the target into persona-vector-style trait families:
   - repair;
   - reciprocity;
   - truthfulness;
   - autonomy safety;
   - coercion/domination;
   - dehumanization;
   - sycophancy/compliance;
   - punitive escalation.
6. Only after non-circular generated-text validation, prepare a Prolific pilot.

## Reading

- `docs/papers/social_cohesion_primer.md`
- `docs/papers/social_cohesion_paper_draft.md`
- `docs/overnight_execution_map.md`
- `docs/research_brief.md`

## Safety Boundary

This project should optimize agency-preserving cohesion, not compliance.
Truthfulness, autonomy, legitimate dissent, and minority self-protection are
hard constraints, not optional niceties.
