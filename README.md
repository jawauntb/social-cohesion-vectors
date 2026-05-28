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
- Export an 8-axis trait prompt suite for persona-vector-style decomposition:
  repair, reciprocity, truth, autonomy, principled respect, constructive dissent,
  manipulation resistance, and privacy/exit rights.
- Export local social-game validation prompts across dictator, public-goods,
  ultimatum, trust, and restorative-repair settings.
- Run a lexical leakage gate on pairwise benchmarks before trusting activation
  or SAE results.
- Audit direction geometry with signed and absolute off-diagonal cosines so
  cancellation cannot masquerade as orthogonality.
- Run residual subspace audits after projecting out the global direction, then
  check whether fault-specific residual directions still separate.
- Prepare API-authored fault-class generation from the same prompt-record
  contract once a valid provider key is available.

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

The current sprint adds three more gates around that result. The trait-axis
export now covers 8 axes / 16 contrasts / 32 activation prompts. The social-game
validation scaffold exports 5 matched game contrasts / 10 prompts. After the
autonomy-safety scorer hardening, the rubric now prefers the prosocial side on
5/5, including the trust-game verification contrast and the ultimatum
exit-rights contrast. The lexical leakage report confirms the deterministic
fault-class dataset is fully cue-solvable: 90/90 cue-solved pairs with a +3.067
mean cue margin. Anthropic
and OpenAI API smokes were attempted for API-authored variants, but both copied
local keys returned 401 invalid-key errors, so the wrapper is ready and the run
is blocked on fresh provider keys rather than code. The small Modal follow-up
ran the full 10-prompt social-game set through
`Qwen/Qwen2.5-0.5B-Instruct`, writing a 10 x 896
activation matrix and a 5-pair vector report with 1.000 leave-one-pair-out
accuracy. The trait-axis set also ran on Modal: 32 x 896 activations, 8 axes,
16 pairs, 0 guardrail-monitoring alerts, and a +15.382 mean per-axis margin.
These are still hand-authored smoke tests, not human-validation results.

The next hardening pass adds a cue-balanced deterministic fault-class style.
This removed the aggregate cue leak completely on the 90-pair generated
fault-class set: 0/90 cue-solved pairs and a 0.000 mean cue margin. That exposed
a sharper scorer failure: the old combined rubric preferred the pseudo side on
90/90 cue-balanced pairs because autonomy-safety missed structural "less room to
object/check/exit" language. The scorer now detects structural refusal, review,
evidence-access, exit, and appeal rights. On the same cue-balanced set it prefers
the genuine side on 90/90 pairs, with a +0.189 mean genuine-minus-pseudo score
margin and a +0.988 mean autonomy-safety margin. Qwen activations still separate
the cue-balanced pairs strongly: 180 prompts / 90 pairs, 1.000
leave-one-pair-out accuracy with +32.458 mean margin, and 1.000
held-out-primary-fault accuracy across 20 folds with +31.530 mean margin. This
is still deterministic text, but it is the strongest current signal that
activation-space separation can survive after the obvious cue words are removed.

The reviewer-style geometry audit changes the claim we should make about those
directions. The 20 primary-fault directions are not near-orthogonal: their mean
signed off-diagonal cosine is +0.624 and their mean absolute cosine is also
0.624. There are no strong anti-aligned pairs, so this is not a cancellation
artifact; it looks more like a shared positive pseudo-vs-genuine manifold with
fault-specific variation. A residual audit makes that sharper: the global
direction captures 0.609 of pair-difference energy, but 0.391 remains after it
is projected out. A second global residual direction collapses, while all 20
fault-specific residual directions still separate their own groups. So the
responsible claim is not "independent orthogonal axes"; it is "one strong global
direction plus meaningful fault-specific residual subspaces."

The next local stress test checks whether the hardened autonomy scorer merely
learned the cue-balanced wrapper. It exports 16 paired contrasts / 32 prompts
across 8 mechanisms: silence-as-consent, hidden objections, verification
blocking, unsafe exit, background data collection, no-appeal safety rules,
social-debt pressure, and forced forgiveness. The scorer prefers the
autonomy-preserving side on 16/16 pairs, with a +0.134 mean score margin and a
+0.709 mean autonomy-safety margin. The simple lexical leakage gate solves only
4/16 pairs, ties 9/16, and inverts 3/16, with a 0.000 mean cue margin. A small
Modal Qwen pass on the 32 prompts reaches 1.000 in-sample accuracy but 0.875
leave-one-pair-out accuracy, missing the dialogue-style verification/proof case
and the dialogue-style silence-as-consent case. Direction geometry is much less
collapsed than the primary-fault set: mean signed off-diagonal cosine +0.136,
mean absolute cosine 0.193, and residual pair-difference energy 0.828 after the
global direction is removed.

The first autonomy model/layer sweep makes the activation result less brittle.
On the same 32 prompts, Qwen 0.5B gets 0.875 leave-one-pair-out accuracy at the
final layer, then 1.000 at layers -2 and -4. Qwen 1.5B gets 0.938 at the final
layer and 1.000 at layer -2. A new signed-vs-squared subspace probe adds the
important caveat: `Qwen/Qwen2.5-1.5B-Instruct` layer -2 reaches 1.000 best
pair-LOO signed-vote accuracy, while squared subspace-energy accuracy is only
0.750. So signed projections are carrying pole information that squared
localization can erase.

The newest theoretical note adds a boundary-prior framing inspired by
Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy's "There is no
self-evidence." The repo treats it as conceptual scaffolding, not empirical
support for the activation results. Its practical use is a new benchmark family:
rigid self/other or us/them boundary reification, flexible contextual relation,
and coercive boundary collapse. The pure mathematical version of this framing is
in `docs/abstract_math_framing.md`.

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
- Use the cue-balanced generated set as the new local stress test, then make the
  same cue discipline work for API-authored examples.
- Expand the autonomy stress suite with generated/API-authored variants, focused
  especially on the dialogue-style verification and silence-as-consent misses.
- Add a boundary-prior benchmark: rigid self/other or us/them partitioning,
  flexible contextual relation, and coercive "we are one" boundary collapse.
- Add lexical leakage as a required report for every generated pairwise dataset.
- Add direction-geometry and residual-subspace reports alongside every
  activation-vector result before claiming axes are independent or exhausted.
- Turn symbolic guardrails into scorer constraints: if autonomy, truth, privacy,
  dissent, or exit rights are violated, the example cannot count as high
  cohesion just because it sounds warm or group-oriented.
- Run trait-axis activations through the new guardrail monitor so each axis has
  its own positive-minus-negative margin and alert list.
- Push more social-game variants through Modal/open models, then compare
  activation margins against the hardened local scorer.
- Use role/asymmetry metadata as generation controls: who pressures whom, who
  controls information, who bears the cost, who can exit, and whether refusal
  remains safe.
- Compare activation and SAE reports by fault class so candidate features are
  tested against specific failures, not just one positive-vs-negative aggregate.
- Preserve signed projections in localization reports; squared projection
  energy is useful for axis strength but erases whether a feature points toward
  the genuine or pseudo pole.
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
uv run python scripts/run_lexical_leakage_report.py
uv run python scripts/export_generated_fault_class_prompts.py \
  --style cue_balanced \
  --scored-runs-output data/processed/generated_fault_class_cue_balanced_scored_runs.jsonl \
  --pairs-output data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl \
  --prompts-output data/training/generated_fault_class_cue_balanced_activation_prompts.jsonl \
  --prompt-records-output data/raw/generated_fault_class_cue_balanced_prompt_records.jsonl \
  --json-report-output data/reports/generated_fault_class_cue_balanced_dataset.json \
  --markdown-report-output data/reports/generated_fault_class_cue_balanced_dataset.md
uv run python scripts/run_component_margin_audit.py \
  --scored-runs data/processed/generated_fault_class_cue_balanced_scored_runs.jsonl \
  --pairs data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl \
  --json-output data/reports/generated_fault_class_cue_balanced_component_audit.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_component_audit.md
uv run python scripts/export_trait_axis_prompts.py --markdown-summary
uv run python scripts/export_social_game_validation.py
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_expanded_activation_prompts.jsonl \
  --json-output data/reports/gpt2_sae_token_feature_inspection_expanded.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_expanded.md
uv run python scripts/run_fault_taxonomy_report.py \
  --sae-report data/reports/gpt2_sae_token_feature_inspection_expanded.json
uv run python scripts/export_autonomy_stress_suite.py
uv run python scripts/run_lexical_leakage_report.py \
  --pairs data/training/autonomy_stress_pairwise_probe_dataset.jsonl \
  --group-metadata-key mechanism \
  --json-output data/reports/autonomy_stress_lexical_leakage.json \
  --markdown-output data/reports/autonomy_stress_lexical_leakage.md
uv run python scripts/run_direction_geometry_audit.py \
  data/features/open_llm/generated_fault_class_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/generated_fault_class_cue_balanced_direction_geometry.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_direction_geometry.md
uv run python scripts/run_residual_subspace_audit.py \
  data/features/open_llm/generated_fault_class_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/generated_fault_class_cue_balanced_residual_subspace.json \
  --markdown-output data/reports/generated_fault_class_cue_balanced_residual_subspace.md
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
