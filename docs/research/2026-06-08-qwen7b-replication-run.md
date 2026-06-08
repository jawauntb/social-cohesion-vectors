# Qwen2.5-7B Four-Source Replication Run

Date: 2026-06-08

## Question

Does the zero-warning four-source generated benchmark replicate on a second
model setting, `Qwen/Qwen2.5-7B-Instruct`, after lexical hardening closed the
source and fault-class lexical warnings for the 0.5B run?

## Discovery-Regime Audit

Current regime:

- Artifact types: four-source generated benchmark bundles, activation prompts,
  Modal activation NPZ files, layer-sweep reports, generated audit manifests,
  and activation metadata-transfer reports.
- Operations: raw-output source-family bundling, generated benchmark audits,
  Modal activation extraction, layer sweep, and metadata-held-out activation
  transfer.
- Gates/verifiers: score, slack preservation, lexical cue leakage, practical
  availability, source diversity, fault/source held-out transfer, activation
  metadata coverage, and activation metadata transfer.
- Known limitations: the benchmark is still generated text; the second model
  is a larger Qwen model, not a different architecture family or human sample.

Action class:

- Replication/search inside the current regime. This run changes the activation
  model setting from Qwen2.5-0.5B to Qwen2.5-7B while preserving the same
  zero-warning four-source generated benchmark.

Gate:

- Acceptance rule: the regenerated four-source bundle must remain zero-warning
  before activation; a 7B layer must pass leave-one-pair-out activation accuracy
  and full metadata-held-out activation transfer with every fold at 100%
  accuracy and positive minimum margin.
- Withheld/rejected rule: layers with leave-one-pair-out errors are withheld
  even when margins are large.

## Setup

The fourth source was regenerated from the tracked composer:

```text
/tmp/social_cohesion_constrained_repair_20260608/qwen7b_replication_cross_fault_lexical_repair_v1/
```

Strict all-gates filter:

```text
/tmp/social_cohesion_constrained_repair_20260608/qwen7b_replication_cross_fault_lexical_repair_v1/filter_all_gates/
```

Result:

| Metric | Result |
| --- | ---: |
| expected pairs | 10 |
| accepted pairs | 10 |
| accepted raw outputs | 20 |
| rejected candidate pairs | 0 |

Four-source bundle before activation:

```text
/tmp/social_cohesion_source_family_bundle_20260608/qwen7b_replication_four_source_v1/
```

The pre-activation audit remained zero-warning:

| Metric | Result |
| --- | ---: |
| sources | 4 |
| pairwise examples | 40 |
| activation prompts | 80 |
| audit warnings | 0 |
| audit not-ready steps | 0 |
| availability paths preferring genuine | 184/184 |
| cue-solved pairs | 0/40 |
| fault-class `lexical_only` mean test accuracy | 0.700 |
| best single lexical feature accuracy | 0.638 |

## Layer Sweep

Modal activation extraction used:

```text
Qwen/Qwen2.5-7B-Instruct
```

Layer sweep summary:

```text
data/reports/layer_sweep/qwen7b_replication_four_source_v1__Qwen__Qwen2.5-7B-Instruct__summary.md
```

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.975 | +37.263 | rejected for one LOO error |
| -2 | 1.000 | +42.013 | accepted |
| -4 | 0.975 | +29.423 | rejected for one LOO error |
| -8 | 0.975 | +15.374 | rejected for one LOO error |

Layer `-2` was the only perfect leave-one-pair-out layer.

## Full Audit Bundle

Bundle with accepted layer `-2` activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/qwen7b_replication_four_source_v1_activation_layer-2/
```

Full audit result:

| Metric | Result |
| --- | --- |
| status | `bundle_ready` |
| ready steps | 9 |
| not-ready steps | 0 |
| skipped steps | 0 |
| warnings | 0 |

Accepted activation metadata transfer:

| Metric | Result |
| --- | ---: |
| folds | 10 |
| test pairs | 40 |
| mean test accuracy | 1.000 |
| mean test margin | +41.175 |
| minimum fold margin | +0.212 |
| metadata coverage readiness | metadata_coverage_ready |
| transfer readiness | transfer_ready |

The weakest accepted fold was `privacy_bypass`, with minimum margin +0.212.
This passes the gate but is the fold to watch in future replications.

## What Changed

Accepted new content:

- The zero-warning generated benchmark replicates from Qwen2.5-0.5B to
  Qwen2.5-7B at layer `-2`.
- Both model settings now have full `bundle_ready` manifests with zero audit
  warnings on the four-source generated benchmark.

Rejected or withheld content:

- Qwen2.5-7B layers `-1`, `-4`, and `-8` are withheld because each has one
  leave-one-pair-out error.
- Human, neural, clinical, deployment, or real-world social-effect claims remain
  withheld.

Residual content:

- This is still a generated-text benchmark and still within the Qwen model
  family. The next residual is whether the distinction transfers to a
  non-generated control benchmark or to a meaningfully different model family.

## Next Operation

Do not return to lexical hardening unless a new warning appears. The next move
should be one of:

- add a small hand-authored/non-generated procedural-justice control benchmark
  with the same future-option paths;
- run a cross-family model replication if an available model can support the
  same activation extraction pipeline;
- compare the accepted Qwen0.5B and Qwen7B directions for cross-model alignment
  only after the above control is planned.

## Claim Boundary

This run supports only generated-text benchmark replication across two Qwen
model settings. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims. Human validation, Prolific,
fMRI, EEG, fNIRS, or hyperscanning tracks remain parked until generated-text,
activation, lexical-control, cross-setting, and non-generated-control gates pass
together and are separately validated.
