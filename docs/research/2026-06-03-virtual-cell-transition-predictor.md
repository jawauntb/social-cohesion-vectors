---
title: 2026-06-03 Virtual-Cell Transition Predictor Baselines
status: draft
date: 2026-06-03
origin: CK-6 parallel batch virtual-cell/transition-predictor lane
---

# 2026-06-03 Virtual-Cell Transition Predictor Baselines

This lane defines a tiny transition-prediction benchmark over existing
transition records. It is intentionally a benchmark grammar only: records are
compute artifacts from CK-style generation assays or toy graph sweeps, and the
predictor does not claim real cell, neural, behavioral, pharmacological,
therapeutic, or biological effects.

The immediate target is modest: given a baseline context plus perturbation
metadata, predict either the observed CK-1 score delta or the transition
`effect_class`. A future learned model should beat these cheap explanations
before its outputs receive interpretive attention.

## Input

The predictor consumes transition-record dictionaries with these fields:

- `baseline_state`: at minimum phase, mechanism, and baseline recipe id.
- `perturbation`: recipe id and component metadata.
- `dose`: absolute perturbation strength.
- `timing`: steering schedule metadata.
- `observed_transition`: includes `ck1_score_delta` for CK-style records.
- `effect_class`: one of the transition-record effect labels.

Toy substrate records can still be represented in the same table, but CK-1
delta prediction is meaningful only for records that carry `ck1_score_delta`.
Missing deltas are treated as zero by the tiny baseline module so lightweight
benchmarks remain total over mixed record sets.

## Baselines

`src/social_cohesion_vectors/experiments/transition_prediction.py` implements
three deterministic baselines:

- `global_mean`: predicts the mean observed CK-1 delta and majority effect class
  from all training transition records.
- `recipe_mean`: uses records with the same perturbation recipe id; if the
  recipe is unseen, it falls back to the global mean baseline.
- `nearest_context`: copies the CK-1 delta and effect class from the nearest
  training record by benchmark-only context features: baseline phase,
  mechanism, baseline recipe id, perturbation recipe id, perturbation component
  ids, absolute strength, and timing schedule.

All three are intentionally low-capacity. They test whether a proposed
transition model is doing more than memorizing recipe labels, obvious context
labels, or nearest examples.

## Evaluation

The current evaluator uses deterministic leave-one-transition-out splits:

1. Hold out one transition record.
2. Train the selected baseline on all remaining records.
3. Predict held-out `ck1_score_delta` and `effect_class`.
4. Report mean absolute CK-1 delta error and effect-class accuracy.

This is not a claim about deployment or human outcomes. It is only a small
benchmark sanity check for transition-record grammar and held-out metadata
prediction.

## Claim Boundary

Use this lane to compare computational transition records and model outputs,
not to make claims about cells, Drosophila, people, therapy, pharmacology, or
neural mechanisms. Any real human, neural, cellular, or biological claim would
require separate validation outside this benchmark.
