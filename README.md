# social-cohesion-vectors

Simulation, scoring, and open-model activation scaffolding for learning candidate
social cohesion vectors before touching expensive human or neural experiments.

## What We Can Run Now

- Generate social-dilemma transcripts from 20-30 seed scenarios.
- Score trajectories for cooperation, repair, hostility, fairness, autonomy risk,
  and truth-preserving dialogue quality.
- Build paired cooperative-vs-adversarial text examples for probes, activation
  captures, reward models, and SAE dataset exports.
- Extract open-weight LLM activations on Modal and train contrastive directions
  over those activations.

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
```

Outputs land under `data/processed`, `data/training`, and `data/reports`.
Only `data/scenarios/seed_scenarios.json` is tracked.

## Modal Activation Lane

After `modal token new` and `uv sync --group dev --extra ml`, smoke test a small
open model activation extraction job:

```bash
modal run social_cohesion_vectors.modal_app.functions.activation_extractor::smoke_extract
```

The first practical target is to run `data/training/activation_prompts.jsonl`
through an open LLM, save one activation vector per prompt, then train a
contrastive direction with `social_cohesion_vectors.activations.contrastive`.
