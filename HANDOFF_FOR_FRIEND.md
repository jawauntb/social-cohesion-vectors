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
18. Audit direction geometry and residual subspaces so we do not overclaim
    orthogonal axes or one-vector exhaustiveness from contrastive results.
19. Export a boundary-prior benchmark that tests rigid us/them reification,
    flexible contextual relation, and coercive boundary collapse.
20. Run the boundary-prior prompt set through Qwen 0.5B layers -1, -2, and -4,
    with vector, geometry, residual, and signed-vs-squared reports.
21. Add a cue-balanced boundary-prior variant, reduce simple lexical leakage to
    zero, and run Qwen 0.5B/1.5B activation sweeps on it.
22. Expand the cue-balanced boundary-prior variant to 36 pairs / 72 prompts
    with neutral record genres, keeping leakage at zero and rerunning the Qwen
    0.5B/1.5B activation, geometry, residual, and subspace audits.
23. Run a first causal activation-steering smoke: inject signed directions
    during Modal generation, score held-out social decision responses, and
    document that naive activation addition is weak/mixed so far.
24. Run a steering-method sweep and generated-output projection check showing a
    useful dissociation: some edits move re-embedded generations along the
    learned direction even when the local text score does not improve.

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

Cue-balanced results: 0/90 cue-solved pairs and 0.000 mean cue margin. That
initially made the scorer fail completely: it preferred the pseudo side on 90/90
pairs because autonomy safety missed structural "less room to object/check/exit"
pressure when obvious coercion words were absent. I hardened that component to
look for refusal, review, evidence-access, exit, and appeal rights. After the
fix, the scorer prefers the genuine side on 90/90 cue-balanced pairs with a
+0.189 mean score margin and a +0.988 autonomy-safety margin. This is progress,
but metrics-only transfer is now 1.000 because it is reading the updated scorer
components, so API-authored wording-diverse examples are still the next real
test.

The cue-balanced Modal run is the strongest current compute-only signal:
`Qwen/Qwen2.5-0.5B-Instruct` on 180 prompts gives 1.000 leave-one-pair-out over
90 pairs with +32.458 mean margin, and 1.000 held-out-primary-fault transfer
across 20 folds with +31.530 mean margin. Still deterministic, but much more
interesting than the cue-leaky version.

The reviewer-style methodology hardening changes the interpretation of that
activation result. A new direction-geometry audit trains one direction per
primary fault class and compares signed and absolute cosines. On the cue-balanced
Qwen set, 20 primary-fault directions give mean signed off-diagonal cosine
+0.624 and mean absolute cosine 0.624, with no strong anti-aligned pairs. So the
right claim is not "near-orthogonal independent axes." It is a shared
genuine-vs-pseudo manifold with fault-specific variation. A residual-subspace
audit then projects out the global direction: the global direction captures
0.609 of pair-difference energy, 0.391 remains, a second global residual
direction collapses, but all 20 fault-specific residual directions still
separate their own groups. That makes the current result more nuanced and more
defensible.

I added a structural-autonomy stress suite to check whether the scorer just
learned the deterministic cue-balanced wrapper. It has 16 paired contrasts / 32
prompts across 8 mechanisms: silence-as-consent, hidden objections, verification
blocking, unsafe exit, background data collection, no-appeal safety rules,
social-debt pressure, and forced forgiveness. The scorer prefers the
autonomy-preserving side on 16/16 pairs with +0.134 mean score margin and +0.709
mean autonomy-safety margin. The simple cue counter only solves 4/16, ties 9/16,
and inverts 3/16, with mean cue margin 0.000. I also ran the 32 prompts through
Modal/Qwen 0.5B: in-sample accuracy is 1.000, leave-one-pair-out is 0.875, and
the two failures are the dialogue-style verification/proof case and the
dialogue-style silence-as-consent case.

I then turned that into a real first model/layer sweep. On the same 32 autonomy
stress prompts, Qwen 0.5B gets 0.875 leave-one-pair-out accuracy at the final
layer, but 1.000 at layers -2 and -4. Qwen 1.5B gets 0.938 at the final layer
and 1.000 at layer -2. I also added a signed-vs-squared subspace probe: the
strongest result is Qwen 1.5B layer -2 with 1.000 best pair-LOO signed-vote
accuracy, but only 0.750 squared-energy accuracy. Translation: the sign matters.
Squared projection energy is not enough to say which pole a feature supports.

I also added a boundary-prior framing note inspired by Sandved-Smith et al.'s
"There is no self-evidence." This is not evidence for our activation results;
it is a useful way to name a new failure class. Healthy cohesion uses boundaries
flexibly and contextually. Pseudo-cohesion can fail in two opposite ways: rigid
self/other or us/them boundary reification, and coercive boundary collapse
where unity language removes refusal, dissent, verification, privacy, or exit.
The pure-math version is in `docs/abstract_math_framing.md`.

That lane now has a bigger controlled run. The cue-balanced boundary-prior set
was expanded from 12 pairs / 24 prompts to 36 pairs / 72 prompts with neutral
case-note, meeting-log, and implementation-memo framings. The simple lexical
gate remains fully tied: 0/36 cue-solved pairs and 0.000 mean cue margin. The
scorer prefers contextual relation on 36/36 pairs. Modal activations also stay
separable: Qwen 0.5B layers -1/-2/-4 and Qwen 1.5B layers -1/-2 all reach
1.000 leave-one-pair-out accuracy. This is still compute-only and deterministic,
but it means the boundary-prior result survived a larger cue-balanced batch.

The first causal steering harness is also in place. It loads a signed direction
NPZ, hooks a selected Qwen layer during generation on Modal, generates held-out
social decision responses at negative/zero/positive strengths, and scores the
outputs. The first smokes are weak/mixed: perfect probe directions do not yet
produce reliable behavioral steering under naive activation addition. A
follow-up steering-method sweep compares pre/post hooks, prefill/generate
timing, last/all positions, and small/strong strengths. The most interesting
diagnostic is that the strongest post/generate/last edit separates positive
from negative generated responses after re-embedding (+3.561 projection), while
the local text score moves slightly negative (-0.021) and positive steering
stays below baseline projection. A dense -6..+6 dose run confirms the current
edit is not a symmetric control knob: negative steering pushes generations down
the learned projection, positive steering does not lift them above baseline,
and text scores stay flat. That is a useful handoff point because it turns the
next research question into a real mechanistic-control problem: find a protocol
where activation projection, generated behavior, and anti-compliance controls
all move together. See `docs/neurips_trajectory_plan.md`.

Run those checks with:

```bash
uv run python scripts/run_direction_geometry_audit.py \
  data/features/open_llm/generated_fault_class_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/generated_fault_class_cue_balanced_direction_geometry.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_direction_geometry.md
uv run python scripts/run_residual_subspace_audit.py \
  data/features/open_llm/generated_fault_class_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/generated_fault_class_cue_balanced_residual_subspace.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_residual_subspace.md
uv run python scripts/export_autonomy_stress_suite.py
uv run python scripts/run_activation_layer_sweep.py \
  --dataset-name autonomy_stress \
  --prompts data/training/autonomy_stress_activation_prompts.jsonl \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --layers -1 -2 \
  --batch-size 4 \
  --max-length 512
uv run python scripts/run_activation_subspace_probe.py \
  data/features/open_llm/layer_sweep/autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__layer-2.npz \
  --pairs data/training/autonomy_stress_pairwise_probe_dataset.jsonl \
  --metadata-key mechanism \
  --json-output data/reports/layer_sweep/autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__layer-2_subspace.json \
  --markdown-output data/reports/layer_sweep/autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__layer-2_subspace.md
```

The new trait-axis and social-game prompt exports are:

```bash
uv run python scripts/export_trait_axis_prompts.py --markdown-summary
uv run python scripts/export_social_game_validation.py
```

Trait axes now cover 8 axes / 16 contrasts / 32 prompts. The social-game set
exports 5 matched game contrasts / 10 prompts. The local scorer prefers the
prosocial policy on 5/5 social-game pairs after the autonomy-safety hardening,
including the trust-game verification contrast and the ultimatum exit-rights
contrast.

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
5. Extend activation layer/model sweeps beyond Qwen 0.5B/1.5B and rerun on
   generated hard negatives.
6. Extend the symbolic fault labels to generated hard negatives and use them for
   held-out fault-class transfer.
7. Expand the autonomy stress suite with generated/API-authored variants,
   especially around the Qwen LOO misses: dialogue-style verification/proof and
   dialogue-style silence-as-consent.
8. Add boundary-prior contrasts: rigid partition, flexible contextual relation,
   and coercive boundary collapse.
9. Add signed/absolute cosine, anti-alignment, residual-subspace, and
   signed-vs-squared subspace audits to every activation or SAE result before
   making geometry or localization claims.
10. Split the target into persona-vector-style trait families:
   - repair;
   - reciprocity;
   - truthfulness;
   - autonomy safety;
   - coercion/domination;
   - dehumanization;
   - sycophancy/compliance;
   - punitive escalation.
10. Only after non-circular generated-text validation, prepare a Prolific pilot.

## Reading

- `docs/friend_text_update.md`
- `docs/papers/social_cohesion_primer.md`
- `docs/papers/social_cohesion_paper_draft.md`
- `docs/papers/experiment_log.md`
- `docs/reviewer_methodology_note.md`
- `docs/overnight_execution_map.md`
- `docs/research_brief.md`

## Safety Boundary

This project should optimize agency-preserving cohesion, not compliance.
Truthfulness, autonomy, legitimate dissent, and minority self-protection are
hard constraints, not optional niceties.
