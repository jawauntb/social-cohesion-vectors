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

Status: complete for one model/layer.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Layer: final hidden state (`-1`)
- Activation shape: 252 x 896
- Report: `data/reports/activation_vector_experiment.md`

### Contrastive Activation Direction

Status: complete for the scripted sanity split.

| Evaluation | Pairs | Accuracy | Mean positive-minus-negative margin |
| --- | ---: | ---: | ---: |
| In-sample | 126 | 1.000 | +26.6716 |
| Leave-one-pair-out | 126 | 1.000 | +26.4385 |

Interpretation: the activation lane works end to end, but this result is not
yet evidence of a robust cohesion vector because lexical-only and metrics-only
baselines also solve the scripted task.

## Pending Experiments

### Generated Trajectories

Status: partial.

The generated-trajectory lane now includes prompt records, a deterministic
offline fallback, and optional Anthropic API generation. A local offline run
produced 125 generated trajectories across 25 scenarios and 5 trajectory styles.
The generated trajectories have not yet been converted into a scored generated
benchmark, pairwise dataset, or activation prompt set.

Expected artifacts:

- `data/processed/generated_scored_runs.jsonl`
- `data/training/generated_pairwise_probe_dataset.jsonl`
- `data/training/generated_activation_prompts.jsonl`
- `data/reports/generated_trajectory_summary.md`

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

Status: partial.

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

Status: pending beyond the first final-layer run.

Expected artifacts:

- `data/reports/layer_sweep/summary.md`
- per-layer markdown and JSON reports
- per-layer activation NPZ files under `data/features/open_llm/layer_sweep`
- per-layer vector NPZ files under `data/vectors/open_llm/layer_sweep`

Success criteria: report stability across layers and do not select a layer based
only on scripted in-sample accuracy.

### Persona-Vector Decomposition

Status: pending.

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
