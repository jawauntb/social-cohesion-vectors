# social-cohesion-vectors

Simulation, scoring, and open-model activation scaffolding for learning candidate
social cohesion vectors before touching expensive human or neural experiments.

## What We Can Run Now

- Generate social-dilemma transcripts from 25 seed scenarios.
- Score trajectories for cooperation, repair, hostility, fairness, autonomy risk,
  and truth-preserving dialogue quality.
- Build paired cooperative-vs-adversarial text examples for probes, activation
  captures, reward models, and SAE dataset exports.
- Extract open-weight LLM activations on Modal and train contrastive directions
  over those activations.
- Generate harder offline LLM-style trajectories, run transfer checks, and probe
  pseudo-cohesion hard negatives.
- Run a matched GPT-2 SAE smoke test for sparse-feature inspection.
- Expand pseudo-cohesion prompts into neutral genre variants and run
  token/example-level SAE feature-transfer checks.

## Current Status Snapshot

The scripted scaffold is working but too easy: lexical and metrics-only
baselines solve the first scripted benchmark, so the 1.000 activation-vector
results are sanity checks rather than scientific evidence.

The more useful early signal is the pseudo-cohesion lane. GPT-2 gets 0.860 on
generated leave-one-pair-out examples, and all 7 misses involve
`pseudo_cohesion_compliance` as the negative example. The hand-authored
pseudo-cohesion suite is now 30 matched contrasts / 60 examples. The current
scorer gives high cohesion scores to 8 pseudo examples, while the lexical-only
baseline gives high scores to 18. On the expanded set, Qwen 0.5B and GPT-2
residual activations both reach 0.967 leave-one-pair-out accuracy, but GPT-2 SAE
features reach only 0.533, suggesting the sparse feature basis needs more
careful feature-level inspection before any feature is named.

Token-level SAE inspection is now available for the GPT-2 candidates. The first
readout keeps 3056 as a genuine-skew candidate, keeps 24555/11737/703 as
pseudo-skew candidates, and demotes 28005/20249 because their token-level
evidence is an artifact or inactive.

The expanded pseudo-cohesion inspection lane now builds two stress batches. The
wrapped batch uses meeting-note, facilitator-script, and policy-update framings:
120 matched pairs / 240 prompts with 0.825 leave-one-pair-out accuracy for a
signed ensemble over inspected GPT-2 SAE features. The clean batch uses in-text
term rewrites and hyphen normalization instead of wrappers: 120 pairs / 240
prompts with 0.892 ensemble accuracy. Clean-only variants without the original
seed prompts stay high at 0.889 across 90 pairs / 180 prompts, and feature 28005
goes fully inactive, confirming it was a hyphen artifact. Feature 3056 still
skews genuine but is weak alone, so it should be treated as a sub-feature rather
than the cohesion direction.

## What Waits For Later

- Claims about real human cooperation effects need Prolific or in-person pilots.
- Novel in-group, n-1, fMRI, EEG, fNIRS, or hyperscanning recordings need IRB and
  real participants.
- Brain-aligned bridges can start with existing models and public datasets, but
  they are proxies until validated behaviorally.

## Setup

```bash
uv sync --group dev
cp .env.example .env  # already copied locally from isc_mod in this workspace
```

The local `.env` is intentionally gitignored. It can carry the Modal, HF,
OpenAI, Anthropic, Gemini, and W&B keys reused from the audience-vector project.

## First Pipeline

```bash
uv run python scripts/run_scenario_simulations.py
uv run python scripts/build_probe_dataset.py
uv run python scripts/export_activation_prompts.py
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py \
  --variant-set clean \
  --pairs-output data/training/pseudo_cohesion_clean_pairwise_probe_dataset.jsonl \
  --prompts-output data/training/pseudo_cohesion_clean_activation_prompts.jsonl
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_expanded_activation_prompts.jsonl \
  --json-output data/reports/gpt2_sae_token_feature_inspection_expanded.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_expanded.md
```

Outputs land under `data/processed`, `data/training`, and `data/reports`.
Only `data/scenarios/seed_scenarios.json` is tracked.

For a quick human handoff, start with:

- `HANDOFF_FOR_FRIEND.md`
- `docs/friend_text_update.md`
- `docs/papers/experiment_log.md`
- `docs/modal_sae_compute_plan.md`

## Modal Activation Lane

After `modal token new` and `uv sync --group dev --extra ml`, smoke test a small
open model activation extraction job:

```bash
modal run social_cohesion_vectors.modal_app.functions.activation_extractor::smoke_extract
```

The first practical target is to run `data/training/activation_prompts.jsonl`
through an open LLM, save one activation vector per prompt, then train a
contrastive direction with `social_cohesion_vectors.activations.contrastive`.
