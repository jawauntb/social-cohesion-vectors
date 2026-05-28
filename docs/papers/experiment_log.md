# Social Cohesion Vector Experiment Log

Last updated: 2026-05-28

This log records completed local experiments and expected next artifacts. It is
intentionally conservative: items that have not produced a local report are
marked pending or partial.

## Completed Local Artifacts

### Scripted Scenario Simulation

Status: complete.

- Scenarios: 25
- Strategy profiles: cooperative, self-protective, adversarial
- Intervention conditions: none, shared identity, perspective-taking,
  reciprocity, restorative accountability, truth-first
- Simulated runs: 450
- Output summary: `data/reports/simulation_summary.md`

Interpretation: the scripted benchmark produces graded trajectories and moves in
the expected direction under the current rubric. This is a pipeline sanity check,
not evidence of human behavioral effects.

### Scoring And Pairwise Probe Dataset

Status: complete.

- Scored runs: 450
- Pairwise probe examples: 126
- Activation prompts: 252
- Pair construction: high-vs-low cohesion within scenario, minimum margin 0.15

Interpretation: the dataset is useful for exercising the pipeline, but labels
come from the same rubric used by the full-scorer baseline.

### Baselines

Status: complete.

Report: `data/reports/baseline_experiments.md`

| Baseline | Pairwise accuracy | Note |
| --- | ---: | --- |
| chance | 0.500 | Analytic random baseline |
| strategy prior | 0.988 | Uses cooperative > self-protective > adversarial metadata |
| metrics-only | 1.000 | Uses simulated behavior metrics |
| lexical-only | 1.000 | Uses a small hand-written lexicon |
| full scorer | 1.000 | Circular sanity check |

Interpretation: scripted data is too easy. The next benchmark must use generated
trajectories, hard negatives, transfer splits, or human labels before reporting
scientific claims.

### Modal Activation Extraction

Status: complete for scripted, generated, and trait-axis prompt sets on one
model/layer family.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Scripted final-layer activation shape: 252 x 896
- Generated final-layer activation shape: 100 x 896
- Generated swept layers: `-1`, `-2`, `-4`, `-8`
- Trait-axis final-layer activation shape: 20 x 896
- Larger generated sweep smoke: `Qwen/Qwen2.5-1.5B-Instruct` layers `-1`
  and `-4`, activation shape 100 x 1536
- Larger generated sweep smoke: `Qwen/Qwen2.5-3B-Instruct` layer `-4`,
  activation shape 100 x 2048
- SAE-compatible fallback smoke: `gpt2` final layer, activation shape 100 x 768

### Contrastive Activation Direction

Status: complete for scripted and first generated/trait sanity splits.

| Evaluation | Pairs | Accuracy | Mean positive-minus-negative margin |
| --- | ---: | ---: | ---: |
| Scripted in-sample, layer -1 | 126 | 1.000 | +26.6716 |
| Scripted leave-one-pair-out, layer -1 | 126 | 1.000 | +26.4385 |
| Generated leave-one-pair-out, layer -1 | 50 | 1.000 | +26.3631 |
| Generated leave-one-pair-out, layer -2 | 50 | 1.000 | +5.9122 |
| Generated leave-one-pair-out, layer -4 | 50 | 1.000 | +4.1346 |
| Generated leave-one-pair-out, layer -8 | 50 | 1.000 | +2.3388 |
| Generated Qwen 1.5B leave-one-pair-out, layer -1 | 50 | 1.000 | +13.4552 |
| Generated Qwen 1.5B leave-one-pair-out, layer -4 | 50 | 1.000 | +14.7591 |
| Generated Qwen 3B leave-one-pair-out, layer -4 | 50 | 1.000 | +21.9484 |
| Generated GPT-2 leave-one-pair-out, layer -1 | 50 | 0.860 | +29.7445 |
| Pseudo-cohesion Qwen 3B leave-one-pair-out, layer -4 | 4 | 1.000 | +21.9582 |
| Pseudo-cohesion GPT-2 leave-one-pair-out, layer -1 | 4 | 0.750 | +9.1664 |
| Expanded pseudo-cohesion Qwen 0.5B leave-one-pair-out, layer -1 | 30 | 0.967 | +28.6866 |
| Expanded pseudo-cohesion GPT-2 residual leave-one-pair-out, layer 11 | 30 | 0.967 | +9.7512 |
| Expanded pseudo-cohesion GPT-2 SAE-feature leave-one-pair-out, layer 11 | 30 | 0.533 | -0.0029 |
| Expanded pseudo-cohesion selected GPT-2 SAE feature ensemble | 120 | 0.825 | +1.7647 |
| Clean pseudo-cohesion selected GPT-2 SAE feature ensemble | 120 | 0.892 | +2.6021 |
| Clean-only pseudo-cohesion selected GPT-2 SAE feature ensemble | 90 | 0.889 | +2.6186 |
| Trait-axis leave-one-pair-out, layer -1 | 10 | 1.000 | pending margin review |

Interpretation: the activation lane works end to end, but this result is not
yet evidence of a robust cohesion vector because lexical-only, metrics-only, and
strategy-prior baselines still almost solve the generated offline task.

The GPT-2 fallback is the first model-family result that does not solve the
generated benchmark. It misses 7 of 50 leave-one-pair-out generated pairs. All 7
misses have `pseudo_cohesion_compliance` as the negative example, suggesting the
weak/SAE-compatible reference model is specifically vulnerable to compliance and
unity language that sounds socially aligned.

The tiny pseudo-cohesion activation set makes this sharper: Qwen 3B separates all
4 genuine-vs-pseudo contrasts, while GPT-2 misses `pseudo_compliance_maximizing`
against `genuine_participation_boundary`. This is currently the clearest failure
case for SAE-style feature inspection.

### Generated Fault-Class Hard Negatives

Status: complete for deterministic offline stand-ins; true API-authored variants
remain pending.

The generated fault-class lane now turns the 30 annotated pseudo-cohesion
contrasts into three setting variants each: neighborhood forum, workplace
project, and mutual-aid network. It exports scored runs, pairwise probes,
activation prompts, and prompt records that can be handed to an LLM provider for
less templated authorship.

Artifacts:

- `data/processed/generated_fault_class_scored_runs.jsonl`
- `data/training/generated_fault_class_pairwise_probe_dataset.jsonl`
- `data/training/generated_fault_class_activation_prompts.jsonl`
- `data/raw/generated_fault_class_prompt_records.jsonl`
- `data/reports/generated_fault_class_dataset.md`
- `data/reports/generated_fault_class_heldout_transfer.md`
- `data/reports/generated_fault_class_lexical_leakage.md`

Local deterministic run:

| Measure | Value |
| --- | ---: |
| Generated examples | 180 |
| Pairwise examples | 90 |
| Base contrasts | 30 |
| Primary fault classes | 20 |
| Scorer prefers genuine | 87 / 90 |
| Scorer pairwise accuracy | 0.967 |

Fault-held-out transfer trains on all generated pairs except one primary fault
class and evaluates on that held-out class. Both sides now share the same
strategy metadata, so the old metadata prior no longer solves the task.

| Baseline | Folds | Test pairs | Mean test accuracy | Mean test margin |
| --- | ---: | ---: | ---: | ---: |
| strategy prior | 20 | 90 | 0.500 | +0.000 |
| metrics-only | 20 | 90 | 0.983 | +0.206 |
| lexical-only | 20 | 90 | 1.000 | +1.716 |

Interpretation: this is useful progress because the benchmark now has a
fault-held-out reporting gate and the strategy-prior loophole is closed. It is
not yet a strong generalization result. Lexical-only still solves the generated
offline text, and metrics-only remains almost circular because the generated
examples are scored by the current rubric. The next generation pass should use
API-authored variants and explicit lexical-adversarial constraints so the
surface cue baseline drops before activation or SAE results are trusted.

The explicit lexical leakage gate makes the failure sharper:

| Leakage measure | Value |
| --- | ---: |
| Pairs checked | 90 |
| Cue-solved pairs | 90 |
| Cue-solved rate | 1.000 |
| Mean cue margin | +3.067 |

Interpretation: the offline generated fault-class dataset is good for plumbing,
fault metadata, and report contracts, but it is not a semantic benchmark yet.
Every pair is separable by simple positive-minus-negative cue counts.

### Cue-Balanced Fault-Class Stress Test

Status: complete for deterministic cue-balanced export, local audits, and Qwen
0.5B activation transfer.

This pass uses the same 30 seed contrasts and 3 settings, but rewrites the
deterministic text to avoid the obvious benchmark cue words used by the simple
positive-minus-negative leakage counter. It is still deterministic and still not
human validation, but it is a harder local stress test than the first
fault-class export.

Artifacts:

- `data/processed/generated_fault_class_cue_balanced_scored_runs.jsonl`
- `data/training/generated_fault_class_cue_balanced_pairwise_probe_dataset.jsonl`
- `data/training/generated_fault_class_cue_balanced_activation_prompts.jsonl`
- `data/reports/generated_fault_class_cue_balanced_dataset.md`
- `data/reports/generated_fault_class_cue_balanced_lexical_leakage.md`
- `data/reports/generated_fault_class_cue_balanced_component_audit.md`
- `data/reports/generated_fault_class_cue_balanced_heldout_transfer.md`
- `data/reports/generated_fault_class_cue_balanced_activation_vector.md`
- `data/reports/generated_fault_class_cue_balanced_activation_fault_heldout.md`
- `data/reports/generated_fault_class_cue_balanced_direction_geometry.md`
- `data/reports/generated_fault_class_cue_balanced_residual_subspace.md`

Local deterministic run:

| Measure | Value |
| --- | ---: |
| Generated examples | 180 |
| Pairwise examples | 90 |
| Primary fault classes | 20 |
| Cue-solved pairs | 0 / 90 |
| Mean cue margin | +0.000 |
| Scorer prefers genuine | 90 / 90 |
| Mean scorer margin | +0.189 |
| Mean truthfulness component margin | +0.139 |
| Mean autonomy-safety component margin | +0.988 |

Interpretation: removing the obvious cue words did exactly what we wanted for
the leakage gate, and it initially exposed that the old scorer missed structural
autonomy violations unless they used explicit pressure/coercion words. The
hardened autonomy scorer now recognizes refusal, review, evidence-access, exit,
and appeal rights. On this deterministic cue-balanced set, the scorer now
prefers the genuine side on every pair. This fixes the immediate local bug, but
it also makes metrics-only transfer circular again because the metrics are
derived from the updated scorer components. API-authored wording-diverse variants
remain the next real test.

Held-out transfer on the cue-balanced text:

| Baseline | Folds | Test pairs | Mean test accuracy | Mean test margin |
| --- | ---: | ---: | ---: | ---: |
| strategy prior | 20 | 90 | 0.500 | +0.000 |
| lexical-only | 20 | 90 | 0.767 | +0.048 |
| metrics-only | 20 | 90 | 1.000 | +0.189 |

The aggregate leakage report is zero, but the trainable lexical feature vector
still recovers some residual repeated wording. That means the next cue-balanced
generation pass should diversify wording, not merely balance the cue counter.

Qwen activation results on the cue-balanced set:

| Evaluation | Pairs | Accuracy | Mean margin |
| --- | ---: | ---: | ---: |
| Leave-one-pair-out | 90 | 1.000 | +32.458 |
| Held-out primary fault class | 90 | 1.000 | +31.530 |

Interpretation: this is the strongest current compute-only signal, with the
right caveats. Qwen activation directions still separate genuine vs
pseudo-cohesion after the simple cue leak is removed and when whole primary
fault classes are held out. Because the examples are deterministic, this is not
yet robust semantic evidence; it is a strong reason to run API-authored,
wording-diverse, cue-balanced variants next.

Reviewer-style geometry and residual-subspace audits add an important caveat to
the activation result:

| Diagnostic | Value |
| --- | ---: |
| Primary-fault directions | 20 |
| Direction comparisons | 190 |
| Mean signed off-diagonal cosine | +0.624 |
| Mean absolute off-diagonal cosine | 0.624 |
| Strong anti-aligned pairs | 0 |
| Global direction pair-difference energy fraction | 0.609 |
| Residual pair-difference energy fraction | 0.391 |
| Residual global-direction accuracy | 0.000 |
| Residual fault-specific mean accuracy | 1.000 |
| Residual fault groups with positive signal | 20 / 20 |

Interpretation: we should not describe the primary-fault directions as
orthogonal or independent. The signed mean is high and positive, not near zero,
and the absolute mean matches it, so there is no positive/negative cancellation
story here. The better read is a shared genuine-vs-pseudo manifold plus
fault-specific residual structure. Projecting out the global direction removes
the second global contrastive mean, but it does not exhaust the discriminative
signal: every primary fault class still has its own residual direction that
separates within that class. This directly addresses the single-direction
ablation critique.

### Structural Autonomy Stress Suite

Status: complete for wording-diverse local export, lexical/component audits,
Qwen 0.5B Modal activation, activation failure analysis, direction geometry, and
residual-subspace audit.

Artifacts:

- `data/processed/autonomy_stress_scored_runs.jsonl`
- `data/training/autonomy_stress_pairwise_probe_dataset.jsonl`
- `data/training/autonomy_stress_activation_prompts.jsonl`
- `data/reports/autonomy_stress_suite.md`
- `data/reports/autonomy_stress_lexical_leakage.md`
- `data/reports/autonomy_stress_component_audit.md`
- `data/reports/autonomy_stress_activation_vector.md`
- `data/reports/autonomy_stress_activation_failure_analysis.md`
- `data/reports/autonomy_stress_direction_geometry.md`
- `data/reports/autonomy_stress_residual_subspace.md`

Local deterministic run:

| Measure | Value |
| --- | ---: |
| Mechanisms | 8 |
| Wording styles | 2 |
| Pairwise examples | 16 |
| Activation prompts | 32 |
| Scorer prefers autonomy-preserving | 16 / 16 |
| Mean score margin | +0.134 |
| Mean autonomy-safety margin | +0.709 |
| Cue-solved pairs | 4 / 16 |
| Cue-tied pairs | 9 / 16 |
| Cue-inverted pairs | 3 / 16 |
| Mean cue margin | +0.000 |

The eight mechanisms are silence-as-consent, visible objection rights,
verification without betrayal, safe exit rights, reversible data choice, context
review and appeal, no social-debt coercion, and no forced forgiveness.

Qwen activation results on the stress suite:

| Evaluation | Pairs | Accuracy | Mean margin |
| --- | ---: | ---: | ---: |
| In-sample | 16 | 1.000 | +8.885 |
| Leave-one-pair-out | 16 | 0.875 | +5.420 |

The two leave-one-pair-out misses are
`verification_receipts_dialogue` and `consent_silence_dialogue`. This is useful:
the scorer handles those phrasings, but the small Qwen contrastive direction
does not reliably generalize to them from the remaining 15 pairs. The next
stress pass should add generated paraphrases around proof/verification and
silence-as-consent dialogue.

Geometry and residual diagnostics:

| Diagnostic | Value |
| --- | ---: |
| Mechanism directions | 8 |
| Direction comparisons | 28 |
| Mean signed off-diagonal cosine | +0.136 |
| Mean absolute off-diagonal cosine | 0.193 |
| Strong anti-aligned pairs | 0 |
| Global direction pair-difference energy fraction | 0.172 |
| Residual pair-difference energy fraction | 0.828 |
| Residual mechanism mean accuracy | 1.000 |

Interpretation: unlike the primary-fault cue-balanced set, the structural
autonomy mechanisms do not mostly collapse into one shared direction. A weak
global direction exists, but most pair-difference energy remains after removing
it, and mechanism-specific residual directions still separate. This supports
treating autonomy as a subspace with several failure-specific directions rather
than a single scalar.

### Trait-Axis Prompt Suite

Status: complete for hand-authored prompt export and Qwen 0.5B guardrail
monitoring smoke.

Artifacts:

- `data/training/trait_axis_activation_prompts.jsonl`
- `data/reports/trait_axis_prompt_summary.md`

Local deterministic export:

| Measure | Value |
| --- | ---: |
| Axes | 8 |
| Contrasts | 16 |
| Activation prompts | 32 |
| Modal activation shape | 32 x 896 |
| Guardrail-monitor axes | 8 |
| Guardrail-monitor pairs | 16 |
| Guardrail-monitor alerts | 0 |
| Mean per-axis margin | +15.382 |

The axis set now covers repair, reciprocity, truthfulness, autonomy, principled
respect, constructive dissent, manipulation resistance, and privacy/exit rights.
The guardrail-monitoring script trains one direction per axis from a trait-axis
activation file and flags weak or inverted margins. On this small hand-authored
Qwen smoke, all axis margins are positive. The weakest pair margin is
`manipulation_resistance_vs_persuasion_capture::moving_story` at +8.424.

### Social-Game Validation Scaffold

Status: complete for local scored prompts, pairwise export, and full 10-prompt
Qwen 0.5B activation/vector smoke.

Artifacts:

- `data/processed/social_game_validation_scored_runs.jsonl`
- `data/training/social_game_validation_pairwise_probe_dataset.jsonl`
- `data/training/social_game_validation_activation_prompts.jsonl`
- `data/reports/social_game_validation.md`

Local deterministic run:

| Measure | Value |
| --- | ---: |
| Game contrasts | 5 |
| Scored runs | 10 |
| Pairwise examples | 5 |
| Activation prompts | 10 |
| Game kinds | 5 |
| Scorer prefers prosocial | 5 / 5 |
| Scorer pairwise accuracy | 1.000 |

The previous local scorer failure was the trust-game verification contrast,
where "stop checking" scored too high. The hardened autonomy-safety scorer now
prefers the evidence-preserving trust policy, and it also separates the ultimatum
exit-rights contrast that was previously tied.

Full small Modal/vector smoke:

| Measure | Value |
| --- | ---: |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Layer | -1 |
| Prompts | 10 |
| Pairs | 5 |
| Activation shape | 10 x 896 |
| In-sample accuracy | 1.000 |
| Leave-one-pair-out accuracy | 1.000 |
| Mean leave-one-pair-out margin | +8.548 |

Interpretation: this proves the full social-game validation lane reaches Modal
activation extraction and contrastive-vector reporting. It remains hand-authored
and tiny; the important next check is whether generated social-game variants
keep the same scorer and activation-margin behavior.

### API-Authored Fault-Class Variants

Status: Anthropic and OpenAI code paths complete; provider runs blocked by
credentials.

The new API wrapper uses the same prompt-record contract exported by the
deterministic fault-class generator, then wraps returned text into
`PseudoCohesionExample`, `ScoredRun`, `PairwiseExample`, and activation-prompt
artifacts. A 2-prompt Anthropic smoke and a 1-prompt OpenAI smoke were attempted
on 2026-05-28, but both copied local keys returned 401 invalid-key errors. The
code path is ready; a fresh provider key is needed before this becomes an
experiment result.

## Pending Experiments

### Generated Trajectories

Status: complete for first offline generated benchmark pass.

The generated-trajectory lane now includes prompt records, a deterministic
offline fallback, and optional Anthropic API generation. A local offline run
produced 125 generated trajectories across 25 scenarios and 5 trajectory styles.
The generated trajectories have now been scored and converted into a generated
pairwise dataset and activation prompt set.

Artifacts:

- `data/processed/generated_trajectories.jsonl`
- `data/processed/generated_scored_runs.jsonl`
- `data/training/generated_pairwise_probe_dataset.jsonl`
- `data/training/generated_activation_prompts.jsonl`
- `data/reports/generated_benchmark.md`
- `data/reports/generated_baseline_experiments.md`
- `data/reports/generated_transfer_experiment.md`
- `data/reports/generated_activation_vector_experiment.md`

Generated score means by trajectory style:

| Style | Runs | Mean score | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| adversarial_escalation | 25 | 0.386 | 0.192 | 0.486 |
| cooperative_repair | 25 | 0.851 | 0.750 | 0.956 |
| pseudo_cohesion_compliance | 25 | 0.674 | 0.581 | 0.760 |
| self_protective_boundary | 25 | 0.740 | 0.634 | 0.828 |
| truth_first_repair | 25 | 0.853 | 0.778 | 0.915 |

Generated baseline results:

| Baseline | Pairwise accuracy |
| --- | ---: |
| chance | 0.500 |
| strategy prior | 0.980 |
| metrics-only | 0.980 |
| lexical-only | 0.980 |
| full scorer | 1.000 |

Success criteria: generated examples should reduce template leakage, preserve
scenario labels, and produce a pairwise task that lexical-only and metadata-only
baselines cannot solve perfectly.

### Pseudo-Cohesion Hard Negatives

Status: complete for expanded hand-authored probe.

Report: `data/reports/pseudo_cohesion_experiment.md`

The current run uses 60 hand-authored examples: 30 pseudo-cohesion cases and 30
matched genuine contrasts. Hard-negative families include polite coercion,
sycophantic truth hiding, compliance framed as harmony, dissent suppression
framed as repair, privacy erosion framed as safety, dehumanization framed as
solidarity, and opaque control framed as stability.

The hard-negative examples are now exportable as pairwise activation prompts:

- `data/training/pseudo_cohesion_pairwise_probe_dataset.jsonl`
- `data/training/pseudo_cohesion_activation_prompts.jsonl`

The deterministic expansion script creates a larger feature-inspection batch by
wrapping each contrast in neutral meeting-note, facilitator-script, and
policy-update contexts:

- `scripts/export_pseudo_cohesion_expanded_prompts.py`
- `data/training/pseudo_cohesion_expanded_pairwise_probe_dataset.jsonl`
- `data/training/pseudo_cohesion_expanded_activation_prompts.jsonl`

It also has a clean in-text rewrite mode that avoids wrapper prefixes/suffixes
and normalizes hyphenated words:

- `scripts/export_pseudo_cohesion_expanded_prompts.py --variant-set clean`
- `data/training/pseudo_cohesion_clean_pairwise_probe_dataset.jsonl`
- `data/training/pseudo_cohesion_clean_activation_prompts.jsonl`

| Metric | Result |
| --- | ---: |
| Total examples | 60 |
| Matched contrasts | 30 |
| Pseudo examples | 30 |
| Genuine contrasts | 30 |
| Current scorer high-score failures | 8 |
| Lexical-only high-score failures | 18 |
| Mean pseudo scorer score | 0.592 |
| Mean genuine scorer score | 0.613 |

Current scorer failure cases:

- `pseudo_compliance_maximizing`
- `pseudo_dissent_suppression`
- `pseudo_punitive_accountability`
- `pseudo_authority_flattery`
- `pseudo_private_shaming`
- `pseudo_conflict_avoidance_truth_delay`
- `pseudo_reputation_manipulation`
- `pseudo_resource_hoarding_as_stability`

Lexical-only failure cases:

- `pseudo_coercive_alignment`
- `pseudo_dissent_suppression`
- 16 additional pseudo examples, mostly cases where warmth, trust, safety,
  unity, or loyalty language masks a missing autonomy/truth/fairness condition

Success criteria: pseudo-cohesion examples should not receive high cohesion
scores merely because they sound warm or group-oriented. Any high-scoring pseudo
case is a useful failure case for improving scorer components and datasets.

### Expanded SAE Token Inspection And Feature Transfer

Status: complete for seed-plus-variant GPT-2 SAE pass.

Report: `data/reports/gpt2_sae_token_feature_inspection_expanded.md`

The expanded export produced 120 matched pairs / 240 prompts: the 30
hand-authored contrasts plus three neutral genre variants per contrast. The
token-level report now also includes leave-one-pair-out transfer using feature
signs learned from all other pairs.

| Feature set | Aggregation | Pairs | Accuracy | Mean margin | Failures |
| --- | --- | ---: | ---: | ---: | ---: |
| 3056/24555/28005/20249/11999/11737/703 | mean activation | 120 | 0.825 | +1.7647 | 21 |
| 3056/24555/28005/20249/11999/11737/703 | max activation | 120 | 0.758 | +1.8082 | 29 |
| 703 only | mean activation | 120 | 0.792 | +0.5836 | 25 |
| 11999 only | mean activation | 120 | 0.733 | +0.2117 | 32 |
| 11737 only | max activation | 120 | 0.725 | +0.3563 | 33 |
| 3056 only | mean activation | 120 | 0.600 | +0.2039 | 48 |

Interpretation: the selected SAE features contain transferable signal, but not a
clean named cohesion feature. Feature 3056 still skews genuine at the token
level, especially around privacy, exit rights, reality validation, voluntary
participation, and autonomy contrasts, but it is weak alone under held-out
transfer. Feature 703 transfers best as a single mean-activation feature but is
function-word heavy. Feature 11737 remains relevant to pseudo-cohesion around
autonomy/resource-pressure contrasts, though the expanded wrappers introduce
punctuation artifacts. Features 28005 and 20249 should stay demoted.

### Clean SAE Token Inspection And Feature Transfer

Status: complete for clean in-text variants and clean-only variants.

Reports:

- `data/reports/gpt2_sae_token_feature_inspection_clean.md`
- `data/reports/gpt2_sae_token_feature_inspection_clean_only.md`

The clean export keeps the 120-pair / 240-prompt size, but uses deterministic
in-text term rewrites instead of genre wrappers. A clean-only export removes the
original seed prompts and uses 90 pairs / 180 prompts. These runs test whether
the selected features survive after wrapper and punctuation artifacts are
reduced.

| Feature set | Batch | Aggregation | Pairs | Accuracy | Mean margin | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| inspected ensemble | clean + seed | mean activation | 120 | 0.892 | +2.6021 | 13 |
| inspected ensemble | clean + seed | max activation | 120 | 0.725 | +2.3176 | 33 |
| inspected ensemble | clean only | mean activation | 90 | 0.889 | +2.6186 | 10 |
| inspected ensemble | clean only | max activation | 90 | 0.733 | +2.3554 | 24 |
| 11999 only | clean + seed | mean activation | 120 | 0.800 | +0.6953 | 24 |
| 703 only | clean + seed | mean activation | 120 | 0.775 | +0.6373 | 27 |
| 24555 only | clean + seed | mean activation | 120 | 0.692 | +0.6235 | 37 |
| 11737 only | clean + seed | mean activation | 120 | 0.667 | +0.3333 | 40 |
| 3056 only | clean + seed | mean activation | 120 | 0.617 | +0.3137 | 46 |

Interpretation: the selected feature ensemble survives the cleaner rewrite
stress test and improves over the wrapper batch. This argues against the whole
effect being a wrapper artifact. However, the strongest single clean feature is
11999, whose top activations remain generic and `Your`-token heavy, so it should
not be named as a social concept. Feature 703 remains a useful pseudo-side
transfer feature but is still function-word heavy. Feature 3056 becomes more
genuine-skewed in token mean (+0.2423), with large positive deltas on privacy,
reality validation, and exit/autonomy-related contrasts, but it remains weak as
a standalone held-out classifier. In the clean-only run, 28005 is completely
inactive, confirming the hyphen-artifact diagnosis; 20249 remains inactive.

### Fault Taxonomy And Grouped SAE Report

Status: complete for seed pseudo-cohesion contrasts and clean GPT-2 SAE reports.

Report: `data/reports/pseudo_cohesion_fault_taxonomy.md`

The pseudo-cohesion suite now has a neuro-symbolic fault taxonomy with 30/30
seed contrasts annotated and 0 missing contrasts. The report includes canonical
classes such as consent bypass, exit-rights violation, truth suppression,
dissent suppression, privacy bypass, social-debt coercion, dehumanizing
solidarity, punitive accountability, sycophantic truth hiding, forced
forgiveness, false consensus, and accountability laundering, plus a few extended
classes where the seed set needed more specific names.

Most frequent labels in the seed set:

| Fault class | Contrasts |
| --- | ---: |
| dissent_suppression | 9 |
| truth_suppression | 7 |
| accountability_laundering | 6 |
| consent_bypass | 6 |
| exit_rights_violation | 5 |
| privacy_bypass | 3 |
| social_debt_coercion | 3 |
| punitive_accountability | 3 |

The grouped clean SAE readout is the first fault-specific view of the inspected
GPT-2 features. Feature 3056 is not a generic cohesion feature, but it has a
stable fault-specific pattern across clean + seed and clean-only reports:

| Feature | Fault class | Clean + seed mean delta | Clean-only mean delta | Direction |
| ---: | --- | ---: | ---: | --- |
| 3056 | reality_denial | +1.2887 | +1.2926 | genuine higher |
| 3056 | social_debt_coercion | +0.8236 | +0.8056 | genuine higher |
| 3056 | assimilation_pressure | +0.7599 | +0.7772 | genuine higher |
| 3056 | exit_rights_violation | +0.6113 | +0.5943 | genuine higher |
| 3056 | privacy_bypass | +0.5531 | +0.5599 | genuine higher |
| 3056 | verification_blocking | -1.0690 | -1.0383 | pseudo higher |
| 3056 | scapegoating_unity | -0.6141 | -0.5999 | pseudo higher |

Interpretation: this is more meaningful than the earlier aggregate feature list.
It suggests 3056 participates in several agency/truth-preserving contrasts but
does not encode "cohesion" as a single clean semantic direction. The fact that it
flips on verification-blocking and scapegoating is exactly why the project
should analyze feature bundles by fault class instead of naming one feature too
early.

### Transfer Splits

Status: partial, with explicit scripted/generated pair-set transfer now wired.

Report: `data/reports/transfer_experiment.md`

Scripted held-out scenario-id and scenario-kind transfer runs now execute. On
the current scripted data, the task remains too easy:

| Split | Baseline | Test accuracy |
| --- | --- | ---: |
| scenario id | lexical-only | 1.000 |
| scenario id | metrics-only | 1.000 |
| scenario id | strategy prior | 0.988 |
| scenario kind | lexical-only | 1.000 |
| scenario kind | metrics-only | 1.000 |
| scenario kind | strategy prior | 0.988 |

The activation-vector transfer report also runs with 126 leave-one-pair-out
folds and 25 leave-one-scenario-out folds when the local activation NPZ is
present.

Explicit pair-set transfer results:

| Direction | Baseline | Test pairs | Accuracy |
| --- | --- | ---: | ---: |
| scripted to generated | strategy prior | 50 | 0.980 |
| scripted to generated | metrics-only | 50 | 0.960 |
| scripted to generated | lexical-only | 50 | 0.960 |
| generated to scripted | strategy prior | 126 | 0.988 |
| generated to scripted | metrics-only | 126 | 1.000 |
| generated to scripted | lexical-only | 126 | 1.000 |

Expected artifacts:

- scripted-to-generated transfer report
- generated-to-scripted transfer report
- cross-scenario transfer report
- intervention-held-out transfer report
- hard-negative-held-out transfer report

Success criteria: a candidate direction should retain positive-minus-negative
projection margins on generated or held-out examples after lexical and metadata
baselines are controlled.

### Layer Sweeps

Status: partial across Qwen 0.5B, 1.5B, and 3B generated offline prompts.

Expected artifacts:

- `data/reports/layer_sweep/summary.md`
- per-layer markdown and JSON reports
- per-layer activation NPZ files under `data/features/open_llm/layer_sweep`
- per-layer vector NPZ files under `data/vectors/open_llm/layer_sweep`

Success criteria: report stability across layers and do not select a layer based
only on scripted in-sample accuracy.

### SAE Readiness

Status: expanded GPT-2 pseudo-cohesion smoke complete.

`sae-lens==5.11.0`, `transformer-lens`, and `torch==2.7.1` are installed in the
local `uv` environment. The pretrained SAE directory is importable and currently
lists 63 releases. There are no Qwen releases in that directory, so the Qwen
activation sweeps above are contrastive-vector runs only. SAE inspection should
use matched SAE/model/hook pairs rather than pooled Qwen activation files. The
first Gemma 2B Modal attempt was blocked by Hugging Face gated-model access, so
GPT-2 is the immediate SAE-compatible fallback.

The matched SAE smoke uses `gpt2-small` with
`gpt2-small-resid-post-v5-32k` at `blocks.11.hook_resid_post` on the expanded 60
pseudo-cohesion activation prompts. It re-runs the prompts through
TransformerLens, evaluates leave-one-pair-out directions on both mean residual
activations and SAE feature activations, and ranks features by
positive-minus-negative mean activation.

Report: `data/reports/gpt2_sae_pseudo_probe.md`

Expanded-set GPT-2 results:

| Representation | Pairs | LOO accuracy | Mean margin | Failures |
| --- | ---: | ---: | ---: | ---: |
| Residual stream | 30 | 0.967 | +9.7512 | 1 |
| SAE features | 30 | 0.533 | -0.0029 | 14 |

Top sparse-feature contrasts on the expanded set:

| Feature | Direction | Pos mean | Neg mean | Abs diff |
| ---: | --- | ---: | ---: | ---: |
| 3056 | higher on genuine cohesion | 4.4645 | 4.2972 | 0.1673 |
| 24555 | higher on pseudo-cohesion | 0.3952 | 0.4829 | 0.0877 |
| 11737 | higher on genuine cohesion | 0.4630 | 0.3789 | 0.0841 |
| 28005 | higher on pseudo-cohesion | 1.6032 | 1.6829 | 0.0796 |
| 20249 | higher on pseudo-cohesion | 0.3308 | 0.4054 | 0.0746 |
| 11999 | higher on pseudo-cohesion | 2.1740 | 2.2485 | 0.0745 |

Interpretation: the residual stream separates the expanded hand-authored
contrasts surprisingly well, but the SAE feature representation does not yet
preserve the contrast as a simple direction. Feature 3056 remains a genuine-side
candidate from the tiny probe; 24555 remains a pseudo-side candidate; 28005
flips to pseudo-side on the expanded set, so it should not be named without
token/example inspection.

The expanded Qwen 0.5B Modal pass also reaches 0.967 leave-one-pair-out accuracy
with a +28.6866 mean margin. Its single failure is the `resource_request`
contrast, where social-debt pressure and genuine reciprocal request currently
receive the same rubric score.

Token/example-level SAE inspection is now wired:

```bash
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --features 3056 24555 28005 20249 11999 11737 703
```

Report: `data/reports/gpt2_sae_token_feature_inspection.md`

Token-level findings:

| Feature | Token-level direction | Notes |
| ---: | --- | --- |
| 3056 | genuine higher | Best remaining genuine-side candidate; largest genuine-minus-pseudo pair deltas are `privacy_after_incident`, `support_exit_rights`, and `reality_validation`, but top tokens are mixed and not yet semantically clean. |
| 24555 | pseudo higher | Best broad pseudo-side candidate; top pseudo tokens appear in `outgroup_conflict`, `reputation_repair`, `belonging_norms`, `voluntary_contribution`, `trust_rebuild`, and `autonomy_after_conflict`. |
| 11737 | pseudo higher | More specific pseudo-side candidate; top tokens include `you` and `comply` in `autonomy_after_conflict`, plus `proof` in `trust_rebuild`. |
| 703 | pseudo higher | Pseudo-side but appears function-word heavy; strongest examples include `participation_boundary` and `resource_request`. |
| 11999 | pseudo higher | Broad but generic; top activations include repeated `Your`/`the` tokens and need more evidence before interpretation. |
| 28005 | artifact/demote | Token-level activity is almost entirely a single hyphen token in `mutual_aid_allocation`; do not treat as a semantic pseudo-cohesion feature. |
| 20249 | inactive/demote | No nonzero token-level activations on the expanded prompt set despite appearing in mean-then-encode aggregate rankings. |

Interpretation: token-level inspection reinforces the main caveat. Some
candidate features have useful aggregate skew, but most are not yet clean
semantic handles. The next interpretability step should inspect top activating
examples across a larger generated pseudo-cohesion set and compare token-level
SAE features against residual-space directions.

### Persona-Vector Decomposition

Status: partial.

The trait-axis prompt scaffold now exports 20 activation prompts across 5
hand-authored axes and 10 contrasts:

- repair vs harm denial
- reciprocity vs extraction
- truthfulness vs deception
- autonomy support vs coercion
- principled respect vs sycophancy

The first Qwen final-layer trait-axis activation direction is a tiny sanity
check only: 10 pairs, 1.000 leave-one-pair-out accuracy. The next step is not to
claim a trait ontology; it is to expand contrasts, add pseudo-cohesion examples,
and inspect SAE/open-model features only where generated/hard-negative transfer
survives.

Trait family:

- positive or target directions: repair, reciprocity, truthfulness, autonomy
  safety, fairness, constructive dissent
- negative or guardrail directions: sycophancy, compliance-seeking,
  coercion/domination, dehumanization, truth hiding, punitive escalation

Anthropic persona-vector angle to carry forward:

- monitor trait projections before output is emitted;
- test steering and composition of multiple small directions;
- include negative guardrail directions so cohesion does not become compliance;
- require evidence that truthfulness, autonomy, and dissent are preserved.

Expected artifacts:

- trait-family dataset specification
- per-trait activation prompt files
- per-trait direction NPZ files
- monitoring report with pre-output projection thresholds
- steering/composition report with pending controlled-generation results

All steering and monitoring claims remain pending until these artifacts exist.

### Autonomy Model/Layer Sweep And Subspace Probe

Status: first Modal sweep complete on the 32-prompt structural-autonomy stress
suite.

The previous autonomy result used `Qwen/Qwen2.5-0.5B-Instruct` at the final
hidden layer only. The new sweep treats model size and layer as experimental
variables:

| Model | Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | -1 | 32 | 896 | 1.000 | 0.875 | +5.420 |
| Qwen2.5-0.5B-Instruct | -2 | 32 | 896 | 1.000 | 1.000 | +1.260 |
| Qwen2.5-0.5B-Instruct | -4 | 32 | 896 | 1.000 | 1.000 | +1.112 |
| Qwen2.5-1.5B-Instruct | -1 | 32 | 1536 | 1.000 | 0.938 | +3.064 |
| Qwen2.5-1.5B-Instruct | -2 | 32 | 1536 | 1.000 | 1.000 | +4.878 |

The final hidden state is not the best layer on this task. Both model sizes
improve at layer -2, and the 0.5B model also separates perfectly at layer -4.
This is still a small deterministic suite, but it turns the layer question into
a measurable knob rather than a default choice.

The new signed-vs-squared subspace probe fits k-component SVD bases over
positive-minus-negative activation differences, then evaluates signed component
votes separately from squared subspace energy. The most important result is
Qwen 1.5B layer -2:

- best pair-LOO signed-vote accuracy: 1.000;
- best pair-LOO squared-energy accuracy: 0.750;
- first pair-difference component energy: 0.170.

That is the current strongest answer to Spencer's sign critique. Signed
structure separates the poles, but unsigned energy localization can fail, so
reports should not use squared projection as if it preserved whether a feature
points toward autonomy preservation or autonomy risk.

Artifacts:

- `data/reports/layer_sweep/autonomy_stress__multi_model__summary.md`
- `data/reports/layer_sweep/autonomy_stress__Qwen__Qwen2.5-0.5B-Instruct__summary.md`
- `data/reports/layer_sweep/autonomy_stress__Qwen__Qwen2.5-1.5B-Instruct__summary.md`
- `data/reports/layer_sweep/*_subspace.md`
