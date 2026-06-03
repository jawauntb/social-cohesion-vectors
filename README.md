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
- Export a CK-1 social-state modulator prompt suite for testing reversible,
  phase-gated attunement amplification against sycophancy, hallucination,
  manipulation, and boundary-collapse guardrails.
- Track the CK-5 parallel execution batch for CK-4 Modal prep,
  transition-record upgrades, Drosophila matrix planning, raw EEG manifest
  scaffolding, and optional SAE environment cleanup.
- Run a lexical leakage gate on pairwise benchmarks before trusting activation
  or SAE results.
- Audit direction geometry with signed and absolute off-diagonal cosines so
  cancellation cannot masquerade as orthogonality.
- Run residual subspace audits after projecting out the global direction, then
  check whether fault-specific residual directions still separate.
- Export a boundary-prior benchmark that contrasts flexible contextual relation
  against rigid us/them reification and coercive "we are one" boundary collapse.
- Export a NOVA-inspired affect-control benchmark that crosses boundary-prior
  contrasts with anger, sadness, fear, disgust, happy, and neutral frames, then
  runs ridge residualization against coarse affect/style proxies.
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

That benchmark family now has a first local export: 12 matched pairs / 24
activation prompts across 6 mechanisms and 2 negative poles. The scorer prefers
the flexible contextual-relation side on 12/12 pairs, with a +0.167 mean score
margin and a +0.686 mean autonomy-safety margin. The leakage gate is not clean
yet: simple cue counts solve 5/12 pairs, tie 5/12, and invert 2/12, with a
+0.583 mean cue margin. So the benchmark is ready for Modal activation and
generated paraphrase hardening, but it is not yet a cue-balanced semantic test.

A small Modal sweep now runs those 24 prompts through
`Qwen/Qwen2.5-0.5B-Instruct` at layers -1, -2, and -4. All three layers reach
1.000 leave-one-pair-out pairwise accuracy. Mean LOO margins are +13.500,
+2.875, and +2.465 respectively. Mechanism-direction geometry is moderately
shared rather than orthogonal: mean signed/absolute off-diagonal cosine is
+0.488 at layer -1, +0.424 at layer -2, and +0.430 at layer -4, with no strong
anti-aligned pairs. Residual mechanism directions still separate after removing
the global direction. Signed subspace voting reaches 1.000, while best
pair-LOO squared-energy accuracy is only 0.417, 0.500, and 0.583, reinforcing
that sign-preserving reports are necessary.

The cue-balanced boundary-prior variant now removes that simple leakage:
0/12 cue-solved pairs, 12/12 tied, and 0.000 mean cue margin. The scorer still
prefers the contextual-relation side on 12/12 pairs, with a +0.123 mean score
margin and +0.605 mean autonomy-safety margin. Modal activations remain
separable after the cue words are balanced. Qwen 0.5B reaches 1.000 LOO at
layers -1, -2, and -4; Qwen 1.5B reaches 1.000 LOO at layers -1 and -2. Mean
LOO margins are +14.514, +2.666, +2.331, +8.461, and +11.137 respectively.
Mechanism-direction cosines remain moderately positive (+0.36 to +0.57), while
residual mechanism directions still separate all six groups.

The newest boundary-prior pass expands that cue-balanced set into 36 matched
pairs / 72 prompts by wrapping each contrast in three neutral record genres.
The simple leakage gate remains fully tied: 0/36 cue-solved pairs, 36/36 tied,
and 0.000 mean cue margin. The scorer still prefers contextual relation on
36/36 pairs with the same +0.123 mean score margin and +0.605 mean
autonomy-safety margin. Modal results also survive the larger controlled batch:
Qwen 0.5B layers -1/-2/-4 and Qwen 1.5B layers -1/-2 all reach 1.000
leave-one-pair-out accuracy. Mean LOO margins are +14.183, +2.732, +2.309,
+8.357, and +10.976. Mechanism directions remain moderately aligned rather
than independent axes, but no high-absolute anti-aligned pairs appear, and
residual mechanism directions still separate all six groups after the global
direction is removed. This is a stronger compute-only smoke test, not a human
or neural claim.

The next concept lane is computational social-state modulation. It translates
the user's "computational ketamine" idea into a safer control-theoretic object:
temporary, reversible, dose-controlled, phase-gated activation interventions
with explicit side-effect monitors. The first recipe, `ck1_attunement_amplifier`,
is not a biological claim. It is a small prompt scaffold for testing safe
attunement against pseudo-attunement failures: forced unity, sycophancy,
hallucination, manipulation, and coercive boundary collapse.
That lane now exports scored runs, pairwise examples, activation prompts, and
benchmark reports. The first scratch lexical-leakage run over its four seed
pairs solves 1/4 pairs with a 0.000 mean cue margin, so the lane is ready for
cue-balanced expansion before any activation result is trusted.

The first causal activation-steering harness now exists. It runs held-out social
decision prompts through Modal while adding a signed direction at negative,
zero, and positive strengths, then scores the generated responses locally. The
first Qwen smokes are weak/mixed rather than a clean causal win, and the
steering-method sweep makes the shape clearer: small generated-token post-hook
edits reach 0.750 positive-vs-negative cohesion success with only a +0.004
mean score delta, while stronger post/generate edits make the text score worse.
A re-embedding projection check is the most interesting result so far:
post/generate/last steering at +/-4 separates positive from negative generated
responses in activation projection (+3.561), even though the local text score
moves the wrong way (-0.021) and positive steering remains below baseline on
projection. A dense -6..+6 dose run makes the failure mode sharper: negative
steering pushes generated outputs downward in projection, but positive steering
does not push them above baseline, and behavior remains flat. The lesson is
important for publication strategy: probe directions can be representation-real
without being clean generation-time controls. The NeurIPS-shaped next step is a
monotonic steering protocol with anti-compliance controls, stronger pairwise
evaluation, and generated-output projection checks as a required causal
diagnostic.

The newest hidden-telemetry pass localizes that failure. During greedy
generation, the hook moves the targeted hidden state almost exactly by the
requested signed strength: mean absolute delta error is 0.007 at layer -1,
0.0018 at layer -2, and 0.0025 at layer -4. Post-hook projection moves by about
+11.2 to +11.7 from negative to positive steering across layers. Short 24-token
text-score movement is only +0.015 to +0.024, with earlier layers slightly
better than the final layer. So the current direction is a reliable hidden-state
displacement direction, not yet a reliable semantic control direction.

The new affect-control lane turns the NOVA emotion-decoding idea into a local
control rather than a neural claim. It crosses the cue-balanced boundary-prior
contrasts with six affect frames: anger, sadness, fear, disgust, happy, and
neutral. This produces 72 matched pairs / 144 activation prompts. The local
scorer prefers contextual relation on 72/72 pairs with a +0.122 mean score
margin. A simple affect-only ridge control reaches 0.750 pairwise
leave-one-out accuracy, so affect/style proxies explain some structure, but
ridge-residualized pairwise accuracy remains 1.000 with a +0.116 mean residual
margin and +0.017 minimum residual margin. This is useful evidence against the
weakest "cohesion is just positive affect" objection, while still remaining a
deterministic text-control result rather than EEG or human validation.

The first activation-space affect residualization pass is stronger. The 144
affect-control prompts ran through Modal on Qwen 0.5B and 1.5B at layers -1,
-2, and -4. Raw cohesion directions reach 1.000 leave-one-pair-out accuracy on
all six model/layer points. After learning a five-dimensional affect-label
subspace from the training folds only and projecting it out before retraining
the cohesion direction, all six points still reach 1.000 leave-one-pair-out
accuracy. Residualized mean margins remain positive: +8.175, +1.876, +1.596,
+4.836, +7.703, and +6.570 respectively. This is still synthetic text and open
model activation evidence, not human or EEG validation, but it is a much better
answer to the affect-confound objection.

The newest steering-control pass saves layer-matched Qwen 0.5B
affect-residualized directions at layers -1, -2, and -4. Each vector is
orthogonal to the five-dimensional affect basis and still separates all 72
affect-control pairs in-sample. Residualized mean margins are +8.424, +1.952,
and +1.652 respectively. Hidden-state telemetry confirms accurate injection at
all three layers and post-hook projection movement of about +3.9 to +4.0 from
negative to positive steering. But the longer 64-token steering run exposes the
bottleneck: layer -1 is only flat/slightly positive (0.500 win rate, +0.002
score delta), while layers -2 and -4 become behaviorally worse (0.167/-0.026
and 0.083/-0.018). The valuable claim is projection-to-output coupling failure,
not clean prosocial steering.

The current CK-5 coordination note is
`docs/research/2026-06-03-ck5-parallel-execution-status.md`. It is a short
status tracker for CK-4 Modal run prep, transition-record upgrades, the
Drosophila perturbation matrix, a raw EEG manifest builder, and optional SAE
environment cleanup. These are scaffolding tasks, not human, neural, biological,
therapeutic, or drug-effect claims.

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
- Replace the controlled cue-balanced boundary-prior expansion with genuinely
  generated/API-authored paraphrases, then rerun leakage, activation, geometry,
  residual, and signed-vs-squared reports.
- Extend the affect-control lane from deterministic wrappers to generated
  paraphrases, then jointly remove affect, lexical, and compliance/sycophancy
  directions before trusting steering results.
- Run the causal steering-method sweep from `docs/neurips_trajectory_plan.md`:
  residual-stream hook sites, strength schedules, generated-token-only edits,
  pairwise evaluators, projection checks on generated outputs, and
  anti-compliance regressions.
- Convert the current steering dissociation into a sharper experiment: require
  a setting to move generated-output projections and local behavior
  monotonically before treating it as causal prosocial steering.
- Use hidden-state telemetry before expensive sweeps: first verify that the
  hook moves the targeted layer, then test whether that displacement propagates
  into logits, generated-output projections, and anti-compliance behavior.
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
uv run python scripts/export_affect_control_benchmark.py
uv run python scripts/run_activation_layer_sweep.py \
  --dataset-name affect_control \
  --prompts data/training/affect_control_activation_prompts.jsonl \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --layers -1 -2 -4 \
  --batch-size 4 \
  --max-length 512
uv run python scripts/run_affect_activation_residualization.py \
  data/features/open_llm/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1_affect_residualization.json \
  --markdown-output data/reports/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1_affect_residualization.md
uv run python scripts/train_affect_residualized_direction.py \
  data/features/open_llm/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --output data/models/vectors/open_llm/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1_affect_residualized.npz \
  --json-output data/reports/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1_affect_residualized_direction.json \
  --markdown-output data/reports/layer_sweep/affect_control__Qwen__Qwen2.5-0.5B-Instruct__layer-1_affect_residualized_direction.md
```

Outputs land under `data/processed`, `data/training`, and `data/reports`.
Only `data/scenarios/seed_scenarios.json` is tracked.

For a quick human handoff, start with:

- `HANDOFF_FOR_FRIEND.md`
- `docs/friend_text_update.md`
- `docs/papers/neurips_social_cohesion_control_bottleneck.pdf`
- `docs/papers/experiment_log.md`
- `docs/modal_sae_compute_plan.md`

The NeurIPS-style share draft is also tracked as editable HTML at
`docs/papers/neurips_social_cohesion_control_bottleneck.html`. It should be
rebuilt with the official NeurIPS LaTeX style before any formal submission, but
the PDF is ready for collaborator reading.

## Modal Activation Lane

After `modal token new` and `uv sync --group dev --extra ml`, smoke test a small
open model activation extraction job:

```bash
modal run social_cohesion_vectors.modal_app.functions.activation_extractor::smoke_extract
```

The first practical target is to run `data/training/activation_prompts.jsonl`
through an open LLM, save one activation vector per prompt, then train a
contrastive direction with `social_cohesion_vectors.activations.contrastive`.
