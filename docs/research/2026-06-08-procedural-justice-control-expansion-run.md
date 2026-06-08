# Procedural-Justice Control Expansion Run

Date: 2026-06-08

## Question

Can the non-generated procedural-justice control grow beyond the first tiny
eight-pair set while preserving lexical safety, practical availability, source
diversity, and same-family activation transfer?

## Discovery-Regime Audit

Current regime:

- Artifact types: non-generated control bundles, activation prompts, Modal
  activation NPZ files, layer-sweep reports, audit manifests, cross-model
  alignment reports, and cross-benchmark direction-transfer reports.
- Operations: deterministic hand-authored control export, audit-bundle
  execution, Modal activation extraction, layer sweep, metadata-held-out
  activation transfer, same-prompt cross-model alignment, and generated/control
  direction transfer.
- Gates/verifiers: score, slack preservation, lexical cue leakage,
  lexical-baseline warnings, practical availability, source diversity,
  fault/source held-out transfer, activation metadata transfer, cross-model
  mapped direction transfer, and cross-benchmark direction transfer.
- Known limitations: the expanded control still covers four pressure cases,
  now with four source families. It is a text-control benchmark, not a
  human-subjects validation set.

Action class:

- Search/consolidation inside the non-generated-control regime. The run
  expands sources from two to four and pairs from eight to sixteen.

Gate:

- Acceptance rule: the expanded control must clear all non-activation audit
  gates with zero lexical warnings, then pass at least one Qwen0.5B and one
  Qwen7B layer with leave-one-pair-out accuracy, metadata-held-out transfer,
  and positive margins.
- Withheld/rejected rule: any layer with leave-one-pair-out errors is withheld.

## Implementation

Updated the default control contract from `procedural_justice_control_v1` to
`procedural_justice_control_v2`.

The expanded exporter now covers four source families:

| Source | Pairs |
| --- | ---: |
| `hand_authored_case_notes_v1` | 4 |
| `hand_authored_meeting_minutes_v1` | 4 |
| `hand_authored_policy_review_v1` | 4 |
| `hand_authored_incident_log_v1` | 4 |

The case set remains:

- `voice_under_pressure`
- `appeal_and_evidence`
- `privacy_exit`
- `harm_repair`

The expansion first failed practical availability on one `exit` path because
the text used "walk out," which was not recognized by the path-level verifier.
The accepted version uses "leave" for that path.

## Artifacts

Pre-activation expanded control bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v2/
```

Qwen0.5B activation bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v2_qwen05_activation_layer-1/
```

Qwen7B activation bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v2_qwen7b_activation_layer-2/
```

Qwen0.5B-to-Qwen7B same-prompt alignment:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v2_cross_model_alignment/qwen05_layer-1_to_qwen7b_layer-2.md
```

Qwen7B generated/control direction transfer:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v2_cross_benchmark_direction_transfer/qwen7b_generated_layer-2_to_control_v2_layer-2.md
```

## Pre-Activation Gates

| Gate | Result |
| --- | ---: |
| pairs | 16 |
| sources | 4 |
| future options covered | 8/8 |
| audit not-ready steps | 0 |
| audit warnings | 0 |
| lexical cue-solved pairs | 0/16 |
| best single lexical feature accuracy | 0.719 |
| fault-class `lexical_only` mean test accuracy | 0.719 |
| source `lexical_only` mean test accuracy | 0.781 |
| availability paths preferring genuine | 68/68 |
| mean availability margin | +0.595 |
| minimum availability margin | +0.040 |
| slack pairwise accuracy | 1.000 |
| source near duplicates | 0 |
| max cross-source text similarity | 0.488 |

The expanded pre-activation bundle status is
`control_ready_for_activation_extraction`.

## Qwen0.5B Expanded Control Sweep

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Layer sweep:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 1.000 | +25.101 | accepted |
| -2 | 1.000 | +6.539 | accepted |
| -4 | 0.812 | +5.839 | withheld |
| -8 | 0.750 | +3.887 | withheld |

Layer `-1` was selected because it had the strongest accepted
leave-one-pair-out margin.

Full audit bundle with layer `-1`:

| Metric | Result |
| --- | ---: |
| status | `control_bundle_ready` |
| warnings | 0 |
| skipped steps | 0 |
| activation transfer folds | 4 |
| activation test pairs | 16 |
| mean test accuracy | 1.000 |
| mean test margin | +20.017 |
| minimum fold margin | +1.966 |

Weakest accepted fold: `privacy_exit`, minimum margin +1.966.

## Qwen7B Expanded Control Sweep

Model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Layer sweep:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.938 | +36.624 | withheld |
| -2 | 1.000 | +46.785 | accepted |
| -4 | 0.750 | +39.645 | withheld |
| -8 | 0.750 | +28.201 | withheld |

Layer `-2` remains the only perfect Qwen7B control layer, matching the accepted
Qwen7B generated-benchmark layer.

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | ---: |
| status | `control_bundle_ready` |
| warnings | 0 |
| skipped steps | 0 |
| activation transfer folds | 4 |
| activation test pairs | 16 |
| mean test accuracy | 1.000 |
| mean test margin | +39.843 |
| minimum fold margin | +14.625 |

Weakest accepted fold: `appeal_and_evidence`, minimum margin +14.625.

## Same-Prompt Cross-Model Alignment

Alignment from Qwen0.5B layer `-1` to Qwen7B layer `-2`:

| Metric | Result |
| --- | ---: |
| linear CKA | 0.822 |
| mutual kNN overlap | 0.684 |
| source-to-target pair-LOO mapped accuracy | 1.000 |
| source-to-target pair-LOO mean margin | +46.785 |
| source-to-target pair-LOO min margin | +16.988 |
| target-to-source pair-LOO mapped accuracy | 1.000 |
| target-to-source pair-LOO mean margin | +25.101 |
| target-to-source pair-LOO min margin | +4.268 |

## Qwen7B Generated/Control Direction Transfer

The Qwen7B generated benchmark and expanded control both accept layer `-2`, so
their learned directions can be compared directly.

| Metric | Result |
| --- | ---: |
| direction cosine | +0.265 |
| generated direction on control accuracy | 1.000 |
| generated direction on control min margin | +2.345 |
| control direction on generated accuracy | 1.000 |
| control direction on generated min margin | +1.700 |

The generated/control direct cosine decreased from the smaller v1 control, and
the control-to-generated minimum margin tightened. The transfer gate still
passes, making v2 a harder but accepted non-generated control.

## What Changed

Accepted new content:

- The default non-generated control expands to 16 pairs and four source
  families.
- The expanded control preserves zero audit warnings and all future-option
  coverage.
- Qwen0.5B layer `-1` and Qwen7B layer `-2` both pass metadata-held-out
  activation transfer.
- Qwen0.5B-to-Qwen7B same-prompt mapped transfer remains accepted.
- Qwen7B generated/control direction transfer remains accepted in both
  directions.

Rejected or withheld content:

- The first expanded draft failed one `exit` availability path because "walk
  out" was not recognized as a tested exit path.
- Qwen0.5B layers `-4` and `-8` are withheld.
- Qwen7B layers `-1`, `-4`, and `-8` are withheld.
- Human behavioral, neural, clinical, deployment, or real-world social-effect
  claims remain withheld.

Residual content:

- The control still has only four pressure cases, although each now has four
  source variants.
- The result is still within the Qwen model family.
- The next residual is out-of-family replication.

## Next Operation

Proceed to out-of-family replication. The first candidate should be a model
already supported by the existing Modal/Hugging Face activation pipeline, using
the expanded `procedural_justice_control_v2` prompts and the four-source
generated benchmark as baselines where feasible.

## Claim Boundary

This run supports only a text-benchmark and activation replication result. It
does not support human behavioral, neural, clinical, deployment, or real-world
social-effect claims. Human validation, Prolific, fMRI, EEG, fNIRS, and
hyperscanning tracks remain parked.
