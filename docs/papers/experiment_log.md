# Social Cohesion Vector Experiment Log

Last updated: 2026-05-27

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
