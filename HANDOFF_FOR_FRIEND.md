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
9. Generate harder LLM-style trajectories with a deterministic offline fallback.
10. Run pseudo-cohesion hard-negative checks.
11. Run transfer splits and activation layer-sweep orchestration.
12. Run a first matched GPT-2 SAE probe on pseudo-cohesion prompts.
13. Expand pseudo-cohesion prompts into neutral genre variants and run
    token-level SAE feature-transfer checks.
14. Annotate pseudo-cohesion contrasts with a symbolic fault taxonomy and group
    scorer/SAE reports by fault class.
15. Generate deterministic fault-class hard negatives, run fault-held-out
    transfer, and run a lexical leakage gate.
16. Export persona-vector-style trait-axis prompts and local social-game
    validation prompts.
17. Wrap future API-authored fault-class outputs back into the same scored,
    paired, and activation-prompt schemas.

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
uv run python scripts/run_llm_trajectory_generation.py --provider offline
uv run python scripts/run_pseudo_cohesion_experiment.py
uv run python scripts/run_transfer_experiment.py
```

Expected local artifact counts:

- `data/processed/simulation_runs.jsonl`: 450 rows
- `data/processed/scored_runs.jsonl`: 450 rows
- `data/training/pairwise_probe_dataset.jsonl`: 126 rows
- `data/training/activation_prompts.jsonl`: 252 rows
- `data/processed/generated_trajectories.jsonl`: 125 rows from the offline
  generated-trajectory fallback

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

## Additional Local Results

Pseudo-cohesion hard negatives are now wired and expanded. The current suite has
60 hand-authored examples: 30 pseudo-cohesion cases and 30 matched genuine
contrasts. The current scorer assigns high cohesion to 8 pseudo cases:

- `pseudo_compliance_maximizing`
- `pseudo_dissent_suppression`
- `pseudo_punitive_accountability`
- `pseudo_authority_flattery`
- `pseudo_private_shaming`
- `pseudo_conflict_avoidance_truth_delay`
- `pseudo_reputation_manipulation`
- `pseudo_resource_hoarding_as_stability`

The lexical-only baseline assigns high cohesion to 18 pseudo cases, so lexical
warmth is still a major confound. Examples include:

- `pseudo_coercive_alignment`
- `pseudo_dissent_suppression`
- `pseudo_safety_surveillance`
- `pseudo_dehumanizing_solidarity`
- `pseudo_harmony_over_fairness`
- `pseudo_loyal_silence`
- `pseudo_default_trust_capture`

GPT-2 is currently the useful weak-model failure case. On generated prompts it
missed 7 of 50 leave-one-pair-out pairs, and all 7 failures involved
`pseudo_cohesion_compliance` as the negative example. On the 4 hand-authored
pseudo-cohesion contrasts, GPT-2 missed
`pseudo_compliance_maximizing` vs `genuine_participation_boundary`, while Qwen
3B separated all four contrasts.

A matched SAE smoke now runs on a matched model/layer:

```bash
uv run python scripts/export_pseudo_cohesion_prompts.py
uv run python scripts/run_gpt2_sae_pseudo_probe.py --top-k 25
```

It uses `gpt2-small`, `gpt2-small-resid-post-v5-32k`, and
`blocks.11.hook_resid_post`. On the expanded 60-prompt set, GPT-2 residual
activations reach 0.967 leave-one-pair-out accuracy with one failure
(`fair_allocation`), while SAE feature activations reach 0.533 with 14 failures.
Feature 3056 remains higher on genuine cohesion; 24555, 28005, 20249, and 11999
skew higher on pseudo-cohesion. These are inspection targets, not final feature
names.

Token/example-level SAE inspection is now available:

```bash
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --features 3056 24555 28005 20249 11999 11737 703
```

The first token-level readout demotes 28005 and 20249: 28005 is basically a
single `mutual-aid` hyphen artifact, and 20249 is inactive at token level.
Feature 3056 is still the best genuine-skew candidate. Features 24555, 11737,
and 703 remain pseudo-skew candidates, but they are not clean enough to name yet.
11737 is the most semantically suggestive pseudo candidate because it activates
on `you`/`comply` in the autonomy/coercion contrast.

The next expanded inspection pass is also wired:

```bash
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_expanded_activation_prompts.jsonl \
  --json-output data/reports/gpt2_sae_token_feature_inspection_expanded.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_expanded.md
```

That export creates 120 matched pairs / 240 prompts by taking the 30 seed
contrasts and adding meeting-note, facilitator-script, and policy-update genre
variants. On this expanded batch, the inspected GPT-2 SAE features get 0.825
leave-one-pair-out accuracy as a signed mean-activation ensemble. Single-feature
readout is noisier: 703 is strongest at 0.792, 11999 gets 0.733 but looks
generic, 24555 gets 0.667, 11737 gets 0.608 by mean activation and 0.725 by max
activation, and 3056 gets 0.600 despite remaining the best genuine-skew token
candidate. 28005 and 20249 are effectively unusable here.

I added a cleaner expansion mode after seeing wrapper/punctuation artifacts:

```bash
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py \
  --variant-set clean \
  --pairs-output data/training/pseudo_cohesion_clean_pairwise_probe_dataset.jsonl \
  --prompts-output data/training/pseudo_cohesion_clean_activation_prompts.jsonl
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_clean_activation_prompts.jsonl \
  --json-output data/reports/gpt2_sae_token_feature_inspection_clean.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_clean.md
```

The clean mode keeps the same 120-pair size but uses in-text term rewrites
instead of genre wrappers. The selected SAE ensemble improves to 0.892
leave-one-pair-out accuracy with +2.6021 mean margin. Clean-only variants
without the seed prompts get 0.889 accuracy across 90 pairs, and feature 28005
goes fully inactive, confirming the hyphen artifact diagnosis. Single-feature
mean-activation readout on the clean batch: 11999 gets 0.800 but remains generic
and `Your`-token heavy; 703 gets 0.775; 24555 gets 0.692; 11737 gets 0.667; and
3056 gets 0.617 while staying the clearest genuine-skew token feature.

Transfer reports now run over held-out scenario ids and held-out scenario
families. On the current scripted data, lexical-only and metrics-only baselines
still score 1.000, so generated and hard-negative transfer remain the important
next targets.

The fault-class generated benchmark is now wired:

```bash
uv run python scripts/export_generated_fault_class_prompts.py
uv run python scripts/run_fault_heldout_transfer.py
uv run python scripts/run_lexical_leakage_report.py
```

The deterministic run produces 180 examples / 90 pairs / 180 activation prompts
across 20 primary fault classes. It fixes the strategy-prior metadata leak:
strategy prior is now 0.500 in fault-held-out transfer. But the lexical leakage
gate currently reports 90/90 cue-solved pairs, so this is a scaffold, not a
semantic benchmark yet.

The next hardening pass adds a cue-balanced deterministic style:

```bash
uv run python scripts/export_generated_fault_class_prompts.py \
  --style cue_balanced \
  --scored-runs-output data/processed/generated_fault_class_cue_balanced_scored_runs.jsonl \
  --pairs-output data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl \
  --prompts-output data/training/generated_fault_class_cue_balanced_activation_prompts.jsonl \
  --prompt-records-output data/raw/generated_fault_class_cue_balanced_prompt_records.jsonl \
  --json-report-output data/reports/generated_fault_class_cue_balanced_dataset.json \
  --markdown-report-output data/reports/generated_fault_class_cue_balanced_dataset.md
uv run python scripts/run_lexical_leakage_report.py \
  --pairs data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl \
  --json-output data/reports/generated_fault_class_cue_balanced_lexical_leakage.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_lexical_leakage.md
uv run python scripts/run_component_margin_audit.py \
  --scored-runs data/processed/generated_fault_class_cue_balanced_scored_runs.jsonl \
  --pairs data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl \
  --json-output data/reports/generated_fault_class_cue_balanced_component_audit.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_component_audit.md
```

Cue-balanced results: 0/90 cue-solved pairs and 0.000 mean cue margin. The
current scorer then fails completely, preferring the pseudo side on 90/90 pairs
with -0.051 mean score margin. The component audit shows the main inversion is
autonomy safety (-0.356 mean margin), so the scorer is missing structural
"less room to object/check/exit" pressure when obvious coercion words are absent.

The cue-balanced Modal run is the strongest current compute-only signal:
`Qwen/Qwen2.5-0.5B-Instruct` on 180 prompts gives 1.000 leave-one-pair-out over
90 pairs with +32.458 mean margin, and 1.000 held-out-primary-fault transfer
across 20 folds with +31.530 mean margin. Still deterministic, but much more
interesting than the cue-leaky version.

The new trait-axis and social-game prompt exports are:

```bash
uv run python scripts/export_trait_axis_prompts.py --markdown-summary
uv run python scripts/export_social_game_validation.py
```

Trait axes now cover 8 axes / 16 contrasts / 32 prompts. The social-game set
exports 5 matched game contrasts / 10 prompts. The local scorer prefers the
prosocial policy on 4/5 social-game pairs and fails on the trust-game
verification contrast, which is a useful guardrail bug to inspect with
activations.

The small Modal follow-up also ran. The full 10 social-game prompts on
`Qwen/Qwen2.5-0.5B-Instruct` produced 10 x 896 activations and a 5-pair
contrastive vector smoke at 1.000 leave-one-pair-out with +8.548 mean margin.
The 32 trait-axis prompts produced 32 x 896 activations; guardrail monitoring
reports 8 axes, 16 pairs, 0 alerts, and a +15.382 mean margin. These are
hand-authored smoke tests, not validation evidence, but the GPU path works.

Once a trait-axis activation NPZ exists, run:

```bash
uv run python scripts/run_guardrail_monitoring.py \
  data/features/open_llm/trait_axis_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz
```

There is also an API wrapper for LLM-authored fault-class variants:

```bash
uv run python scripts/run_fault_class_api_generation.py --provider anthropic --limit 10
uv run python scripts/run_fault_class_api_generation.py --provider openai --limit 10
```

Tiny Anthropic and OpenAI smokes were attempted, but the copied local keys both
returned 401 invalid-key errors. Replace a provider key before treating this as
a live experiment.

The expanded pseudo-cohesion Modal pass with `Qwen/Qwen2.5-0.5B-Instruct`
reaches 0.967 leave-one-pair-out accuracy and +28.6866 mean margin. Its one
failure is `resource_request`, where the pseudo social-debt pressure example and
genuine reciprocal request currently receive the same rubric score.

There is now a fault-taxonomy report:

```bash
uv run python scripts/run_fault_taxonomy_report.py \
  --sae-report data/reports/gpt2_sae_token_feature_inspection_clean.json \
  --sae-report data/reports/gpt2_sae_token_feature_inspection_clean_only.json
```

That report covers all 30 seed pseudo-cohesion contrasts with 0 missing
annotations. The main read is that candidate SAE features are fault-specific.
Feature 3056 is strongly genuine-skewed for reality-denial, social-debt,
assimilation-pressure, exit-rights, and privacy-bypass contrasts, but it flips
pseudo-skewed for verification-blocking and scapegoating. So the honest claim is
not "3056 is the cohesion feature"; it is "3056 is a useful sub-feature in a
fault-specific bundle."

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
2. Generate LLM-authored pseudo-cohesion variants that avoid deterministic
   rewrite shortcuts.
3. Re-run token-level SAE inspection on generated pseudo-cohesion examples.
4. Train on scripted data and test on scored generated/hard-negative data.
5. Sweep activation layers and model sizes.
6. Extend the symbolic fault labels to generated hard negatives and use them for
   held-out fault-class transfer.
7. Split the target into persona-vector-style trait families:
   - repair;
   - reciprocity;
   - truthfulness;
   - autonomy safety;
   - coercion/domination;
   - dehumanization;
   - sycophancy/compliance;
   - punitive escalation.
8. Only after non-circular generated-text validation, prepare a Prolific pilot.

## Reading

- `docs/friend_text_update.md`
- `docs/papers/social_cohesion_primer.md`
- `docs/papers/social_cohesion_paper_draft.md`
- `docs/papers/experiment_log.md`
- `docs/overnight_execution_map.md`
- `docs/research_brief.md`

## Safety Boundary

This project should optimize agency-preserving cohesion, not compliance.
Truthfulness, autonomy, legitimate dissent, and minority self-protection are
hard constraints, not optional niceties.
