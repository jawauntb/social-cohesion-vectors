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
- Annotate pseudo-cohesion contrasts with a symbolic fault taxonomy and group
  scorer/SAE outcomes by specific failure mode.
- Generate fault-class pseudo-cohesion stand-ins, export scored runs/pairs/
  activation prompts, and run fault-held-out transfer folds.

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

The first symbolic fault-taxonomy pass now covers all 30 seed contrasts with 0
missing annotations. It tags pseudo-cohesion failures by consent bypass, exit
rights, dissent suppression, truth suppression, privacy bypass, social-debt
coercion, false consensus, accountability laundering, and related extended
classes. Grouped SAE reports show fault-specific structure rather than one clean
feature: feature 3056 is strongly genuine-skewed for reality validation,
social-debt coercion, exit-rights, privacy, and assimilation-pressure contrasts,
but it flips pseudo-skewed for verification-blocking and scapegoating contrasts.
That makes 3056 useful evidence for a bundle, not a standalone cohesion vector.

The paper draft now folds in the 2024-2026 arXiv state of the art around sparse
autoencoders, persona vectors, refusal and sycophancy directions, altruism
features, plural value steering, persuasion risk, and LLM-agent social dynamics.
It also adds a *Magnifica Humanitas* grounding: cohesion means truthful,
dignity-preserving, agency-respecting relation, not optimized agreement,
dependence, or conformity.

The generated fault-class lane now exports 180 deterministic offline stand-ins
for LLM-authored hard negatives: 30 seed contrasts x 3 social settings x
pseudo/genuine sides. It produces 90 pairwise examples across 20 primary fault
classes. The scorer prefers the genuine side on 87/90 pairs. Fault-held-out
transfer now controls strategy metadata, so the strategy prior falls to chance
at 0.500. Lexical-only still reaches 1.000 and metrics-only reaches 0.983, which
means the next generation pass has to remove obvious lexical/rubric cues rather
than count this as robust generalization.

## Next Steps

The next phase is to make pseudo-cohesion more formal and less vibe-driven. The
repo now has a first neuro-symbolic fault taxonomy for why a pseudo-cohesion
example fails, so the next work is to harden it against generated language and
turn the labels into measurable guardrails.

Immediate build targets:

- Extend the seed fault labels to LLM-authored hard negatives and generated
  trajectories.
- Replace the deterministic offline fault-class stand-ins with API-authored
  variants, then rerun fault-held-out transfer.
- Make generated pseudo/genuine sides less lexically separable so lexical-only
  can no longer solve the benchmark.
- Turn symbolic guardrails into scorer constraints: if autonomy, truth, privacy,
  dissent, or exit rights are violated, the example cannot count as high
  cohesion just because it sounds warm or group-oriented.
- Use role/asymmetry metadata as generation controls: who pressures whom, who
  controls information, who bears the cost, who can exit, and whether refusal
  remains safe.
- Compare activation and SAE reports by fault class so candidate features are
  tested against specific failures, not just one positive-vs-negative aggregate.
- Add cultural and dialect/context stress tests so the model does not learn
  "institutional therapy voice" as a proxy for cohesion.
- Use the clean deterministic results as a baseline, then generate LLM-authored
  hard negatives to test whether the same feature ensemble survives less
  templated language.
- Add persuasion-risk and narrative-reliance checks before any human-facing
  experiment.
- Add multi-agent social-dynamics tests for direct reciprocity, indirect
  reciprocity, public goods, reputation, and long-horizon cooperation.
- Try public Gemma Scope-style SAE dictionaries as the next cross-model
  interpretability baseline before training custom SAEs.

The meaningful research question for the next sprint is:

> Do specific activation directions or SAE feature bundles track specific
> social-reasoning fault classes such as consent bypass, dissent suppression, or
> social-debt coercion?

If that survives generated examples, then the project has something stronger
than a benchmark: a representational map from social failure modes to model
features.

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
uv run python scripts/export_generated_fault_class_prompts.py
uv run python scripts/run_fault_heldout_transfer.py
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_expanded_activation_prompts.jsonl \
  --json-output data/reports/gpt2_sae_token_feature_inspection_expanded.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_expanded.md
uv run python scripts/run_fault_taxonomy_report.py \
  --sae-report data/reports/gpt2_sae_token_feature_inspection_expanded.json
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
