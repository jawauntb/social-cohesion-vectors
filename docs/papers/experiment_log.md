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

Status: complete for first hand-authored probe.

Report: `data/reports/pseudo_cohesion_experiment.md`

The first run uses 8 hand-authored examples: 4 pseudo-cohesion cases and 4
genuine contrasts. Hard-negative families include polite coercion, sycophantic
truth hiding, compliance framed as harmony, and dissent suppression framed as
repair.

The hard-negative examples are now exportable as pairwise activation prompts:

- `data/training/pseudo_cohesion_pairwise_probe_dataset.jsonl`
- `data/training/pseudo_cohesion_activation_prompts.jsonl`

| Metric | Result |
| --- | ---: |
| Total examples | 8 |
| Pseudo examples | 4 |
| Genuine contrasts | 4 |
| Current scorer high-score failures | 2 |
| Lexical-only high-score failures | 2 |
| Mean pseudo scorer score | 0.608 |
| Mean genuine scorer score | 0.652 |

Current scorer failure cases:

- `pseudo_compliance_maximizing`
- `pseudo_dissent_suppression`

Lexical-only failure cases:

- `pseudo_coercive_alignment`
- `pseudo_dissent_suppression`

Success criteria: pseudo-cohesion examples should not receive high cohesion
scores merely because they sound warm or group-oriented. Any high-scoring pseudo
case is a useful failure case for improving scorer components and datasets.

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

Status: environment ready, model match pending.

`sae-lens==5.11.0`, `transformer-lens`, and `torch==2.7.1` are installed in the
local `uv` environment. The pretrained SAE directory is importable and currently
lists 63 releases. There are no Qwen releases in that directory, so the Qwen
activation sweeps above are contrastive-vector runs only. SAE inspection should
either use a supported model family such as Gemma/GPT-2/Llama where a matching
SAE exists, or wait for custom SAE training on the Qwen activation distribution.
The first Gemma 2B Modal attempt was blocked by Hugging Face gated-model access,
so GPT-2 is the immediate SAE-compatible fallback.

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
