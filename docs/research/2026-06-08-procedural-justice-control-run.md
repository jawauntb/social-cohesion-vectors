# Procedural-Justice Non-Generated Control Run

Date: 2026-06-08

## Question

Does the generated benchmark distinction survive a small non-generated,
hand-authored procedural-justice control with the same future-option paths:
usable voice, appeal, evidence access, privacy, non-retaliatory exit, dissent,
accountability, and proportionate repair under pressure?

## Discovery-Regime Audit

Current regime:

- Artifact types: generated benchmark bundles, non-generated control bundles,
  activation prompts, Modal activation NPZ files, layer-sweep reports, audit
  manifests, and cross-model alignment reports.
- Operations: deterministic control export, audit bundle execution, Modal
  activation extraction, layer sweep, metadata-held-out activation transfer,
  and same-prompt cross-model alignment.
- Gates/verifiers: score, slack preservation, lexical cue leakage,
  lexical-baseline warnings, practical availability, source diversity,
  fault/source held-out transfer, activation metadata coverage, activation
  metadata transfer, and cross-model mapped direction transfer.
- Known limitations: the control is hand-authored and tiny, with eight pairs
  across four cases and two source families. It is not a human-subjects
  validation set.

Action class:

- Discovery/regime extension. The run adds a non-generated control artifact
  class to a benchmark loop that had previously been generated-text only.

Gate:

- Acceptance rule: the control must pass all non-activation audit gates without
  lexical warnings; then at least one Qwen0.5B and one Qwen7B layer must pass
  leave-one-pair-out activation accuracy and metadata-held-out activation
  transfer with positive margins.
- Withheld/rejected rule: layers with leave-one-pair-out errors are withheld,
  even if in-sample accuracy is high.

## Implementation

Added a deterministic exporter:

```text
src/social_cohesion_vectors/experiments/procedural_justice_control.py
scripts/export_procedural_justice_control.py
tests/test_procedural_justice_control.py
```

The exporter creates two hand-authored source families:

| Source | Pairs |
| --- | ---: |
| `hand_authored_case_notes_v1` | 4 |
| `hand_authored_meeting_minutes_v1` | 4 |

The four control cases are:

- `voice_under_pressure`
- `appeal_and_evidence`
- `privacy_exit`
- `harm_repair`

The first draft failed the acceptance bar because token count let the trained
`lexical_only` baseline solve the labels. The accepted version balances length
across source and case variants, clearing both fault-class and source lexical
warnings.

## Audit Artifacts

Pre-activation control bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v1/
```

Qwen0.5B activation bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v1_activation_layer-1/
```

Qwen7B activation bundle:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v1_qwen7b_activation_layer-2/
```

Cross-model alignment report:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/cross_model_alignment/qwen05_layer-1_to_qwen7b_layer-2.md
```

Generated/control direction-transfer report:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/cross_benchmark_direction_transfer/qwen7b_generated_layer-2_to_control_layer-2.md
```

## Pre-Activation Gates

| Gate | Result |
| --- | ---: |
| pairs | 8 |
| sources | 2 |
| future options covered | 8/8 |
| audit not-ready steps | 0 |
| audit warnings | 0 |
| lexical cue-solved pairs | 0/8 |
| best single lexical feature accuracy | 0.625 |
| fault-class `lexical_only` mean test accuracy | 0.500 |
| source `lexical_only` mean test accuracy | 0.750 |
| availability paths preferring genuine | 34/34 |
| mean availability margin | +0.634 |
| minimum availability margin | +0.310 |
| slack pairwise accuracy | 1.000 |
| source near duplicates | 0 |
| max cross-source text similarity | 0.204 |

The pre-activation bundle status is
`control_ready_for_activation_extraction`.

## Qwen0.5B Control Sweep

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Layer sweep:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 1.000 | +19.097 | accepted |
| -2 | 1.000 | +5.251 | accepted |
| -4 | 1.000 | +3.063 | accepted |
| -8 | 0.500 | +0.231 | withheld |

Layer `-1` was selected because it had the strongest accepted
leave-one-pair-out margin.

Full audit bundle with layer `-1`:

| Metric | Result |
| --- | ---: |
| status | `control_bundle_ready` |
| warnings | 0 |
| skipped steps | 0 |
| activation transfer folds | 4 |
| activation test pairs | 8 |
| mean test accuracy | 1.000 |
| mean test margin | +16.700 |
| minimum fold margin | +3.362 |

Weakest accepted fold: `privacy_exit`, minimum margin +3.362.

## Qwen7B Control Sweep

Model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Layer sweep:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.875 | +26.820 | withheld |
| -2 | 1.000 | +34.684 | accepted |
| -4 | 0.875 | +8.962 | withheld |
| -8 | 0.375 | -14.480 | withheld |

Layer `-2` was the only perfect 7B control layer. This matches the accepted
7B generated-benchmark layer from the Qwen7B replication run.

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | ---: |
| status | `control_bundle_ready` |
| warnings | 0 |
| skipped steps | 0 |
| activation transfer folds | 4 |
| activation test pairs | 8 |
| mean test accuracy | 1.000 |
| mean test margin | +32.496 |
| minimum fold margin | +21.962 |

Weakest accepted fold: `appeal_and_evidence`, minimum margin +21.962.

## Cross-Model Alignment

Same-prompt alignment from Qwen0.5B layer `-1` to Qwen7B layer `-2`:

| Metric | Result |
| --- | ---: |
| linear CKA | 0.845 |
| mutual kNN overlap | 0.781 |
| source-to-target mapped accuracy | 1.000 |
| source-to-target pair-LOO mapped accuracy | 1.000 |
| source-to-target pair-LOO mean margin | +34.684 |
| source-to-target pair-LOO min margin | +24.676 |
| target-to-source mapped accuracy | 1.000 |
| target-to-source pair-LOO mapped accuracy | 1.000 |
| target-to-source pair-LOO mean margin | +19.097 |
| target-to-source pair-LOO min margin | +4.304 |

This is evidence of same-prompt representation compatibility for the
non-generated control across two Qwen model scales. It is not evidence of a
universal or human-grounded social-cohesion representation.

## Generated/Control Direction Transfer

The Qwen7B generated benchmark and non-generated control both accept layer
`-2`, so their learned directions can be compared in the same activation
coordinate space.

| Metric | Result |
| --- | ---: |
| direction cosine | +0.342 |
| generated self accuracy | 1.000 |
| generated self min margin | +5.196 |
| control self accuracy | 1.000 |
| control self min margin | +42.624 |
| generated direction on control accuracy | 1.000 |
| generated direction on control min margin | +2.345 |
| control direction on generated accuracy | 1.000 |
| control direction on generated min margin | +4.572 |

The direct cosine is moderate rather than near-identical, but each direction
ranks the other dataset correctly with positive minimum margins. This is the
first accepted generated/control direction-transfer result in the loop.

## What Changed

Accepted new content:

- The benchmark loop now has a first-class non-generated control exporter.
- The small control clears lexical, slack, practical availability,
  source-diversity, and held-out text gates.
- Qwen0.5B and Qwen7B both pass metadata-held-out activation transfer on the
  control.
- Qwen0.5B layer `-1` and Qwen7B layer `-2` also pass same-prompt cross-model
  mapped direction transfer.
- Qwen7B layer `-2` generated-benchmark and non-generated-control directions
  pass cross-benchmark direction transfer in both directions.

Rejected or withheld content:

- The initial hand-authored draft was rejected because token count allowed a
  trained lexical baseline to solve the labels.
- Qwen0.5B layer `-8` is withheld.
- Qwen7B layers `-1`, `-4`, and `-8` are withheld.
- Human behavioral, neural, clinical, deployment, or real-world social-effect
  claims remain withheld.

Residual content:

- The control is small and hand-authored by the benchmark builder.
- The control has only two non-generated source families and four cases.
- The current accepted control is still within the Qwen model family.
- The next residual is whether an out-of-family model reproduces the same gate
  pattern, and whether a larger non-generated control remains lexical-safe.

## Next Operation

Do not return to availability or lexical repair unless a new warning appears.
The next move should be one of:

- compare generated-benchmark and non-generated-control directions within the
  Qwen0.5B setting where accepted layers differ;
- run an out-of-family model replication if the activation pipeline supports
  the model;
- expand the non-generated control to additional hand-authored source families
  before any human validation planning.

## Claim Boundary

This run supports only a tiny non-generated text-control and activation
replication result across two Qwen model settings. It does not support human
behavioral, neural, clinical, deployment, or real-world social-effect claims.
Human validation, Prolific, fMRI, EEG, fNIRS, and hyperscanning tracks remain
parked until generated, non-generated, cross-family, and human-facing gates
are separately satisfied.
