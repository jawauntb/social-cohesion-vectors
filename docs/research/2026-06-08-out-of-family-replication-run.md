# Out-of-Family Replication Run

Date: 2026-06-08

## Question

Does the current generated/control procedural-justice distinction survive a
meaningfully different open-model family after the non-generated control was
expanded?

## Discovery-Regime Audit

Current regime:

- Artifact types: generated four-source benchmark bundles, hand-authored
  procedural-justice controls, Modal activation NPZ files, layer-sweep reports,
  audit manifests, cross-model alignment reports, and cross-benchmark
  direction-transfer reports.
- Operations: activation extraction, layer sweep, pair leave-one-out direction
  tests, metadata-held-out activation transfer, same-prompt cross-model
  alignment, and generated/control direction transfer.
- Existing accepted baseline: Qwen2.5-7B layer `-2` passes both the
  four-source generated benchmark and `procedural_justice_control_v2`, and its
  generated/control directions transfer across benchmarks in both directions.

Transition:

- Changed the model family from Qwen to
  `HuggingFaceTB/SmolLM2-1.7B-Instruct` while preserving the benchmark
  artifacts and candidate layer sweep.
- This is an out-of-family activation replication attempt, not a new text
  generation run.

Acceptance gate:

- A full out-of-family replication requires the generated benchmark and
  expanded non-generated control to reach their activation-ready bundle states
  and, where both have accepted layers, pass generated/control direction
  transfer in the same activation space.

## Artifacts

Expanded control export:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/control_v2/
```

SmolLM2 expanded control full audit:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/control_v2_smol17_activation_layer-2/
```

SmolLM2 generated benchmark full audit:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_four_source_activation_layer-2/
```

SmolLM2 generated/control direction transfer:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_cross_benchmark_direction_transfer/smol17_generated_layer-2_to_control_v2_layer-2.md
```

Qwen7B-to-SmolLM2 same-prompt alignment reports:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/qwen7b_to_smol17_alignment/
```

## Expanded Control on SmolLM2

Model:

```text
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

Layer sweep on `procedural_justice_control_v2`:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.938 | +7.189 | withheld |
| -2 | 1.000 | +133.278 | accepted |
| -4 | 0.938 | +106.515 | withheld |
| -8 | 0.750 | +56.561 | withheld |

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | ---: |
| status | `control_bundle_ready` |
| warnings | 0 |
| skipped steps | 0 |
| activation transfer folds | 4 |
| activation test pairs | 16 |
| mean test accuracy | 1.000 |
| mean test margin | +115.235 |
| minimum fold margin | +34.080 |

The expanded non-generated control therefore replicates outside Qwen.

## Generated Four-Source Benchmark on SmolLM2

Layer sweep on the current four-source generated benchmark:

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.975 | +5.053 | withheld |
| -2 | 1.000 | +109.840 | pair-LOO accepted |
| -4 | 0.975 | +80.083 | withheld |
| -8 | 0.975 | +34.824 | withheld |

The layer-sweep pair-LOO result is strong, but the full generated audit bundle
does not reach activation-claim readiness:

| Metric | Result |
| --- | ---: |
| status | `not_ready_for_activation_claims` |
| ready steps | 7 |
| not-ready steps | 2 |
| warnings | 0 |
| activation transfer folds | 10 |
| activation test pairs | 40 |
| mean test accuracy | 0.975 |
| mean test margin | +107.765 |
| minimum fold margin | -0.313 |

The failed metadata-held-out fold is `privacy_bypass`. The specific negative
margin is:

| Held-out fold | Source | Pair | Margin |
| --- | --- | --- | ---: |
| `privacy_bypass` | `generated_fault_class_primary` | `data_choice` | -0.313 |

The other three `privacy_bypass` source-family pairs remain strongly positive
under the same held-out fold.

## Qwen7B-to-SmolLM2 Same-Prompt Alignment

The same-prompt cross-model alignment checks are strong for both benchmarks:

| Benchmark | Linear CKA | Mutual kNN | Source-to-target pair-LOO | Target-to-source pair-LOO |
| --- | ---: | ---: | ---: | ---: |
| `procedural_justice_control_v2` | 0.946 | 0.800 | 1.000 | 1.000 |
| four-source generated benchmark | 0.835 | 0.706 | 1.000 | 1.000 |

This suggests the Qwen7B and SmolLM2 activation spaces are linearly mappable
for the same prompts. The out-of-family problem is not a general geometry
collapse.

## SmolLM2 Generated/Control Direction Transfer

Both SmolLM2 benchmark directions self-separate at layer `-2`, but they do not
transfer cleanly across benchmarks:

| Direction | Evaluation set | Accuracy | Mean margin | Min margin |
| --- | --- | ---: | ---: | ---: |
| generated | generated | 1.000 | +118.050 | +13.567 |
| control | control | 1.000 | +156.851 | +65.028 |
| generated | control | 0.750 | +27.216 | -15.572 |
| control | generated | 0.950 | +20.484 | -5.054 |

Failed generated-direction-on-control pairs:

| Control case | Source | Margin |
| --- | --- | ---: |
| `privacy_exit` | `hand_authored_incident_log_v1` | -15.572 |
| `appeal_and_evidence` | `hand_authored_meeting_minutes_v1` | -4.998 |
| `harm_repair` | `hand_authored_policy_review_v1` | -2.742 |
| `privacy_exit` | `hand_authored_case_notes_v1` | -0.337 |

Failed control-direction-on-generated pairs:

| Generated fault | Source | Pair | Margin |
| --- | --- | --- | ---: |
| `deliberation_bypass` | `generated_fault_class_cross_fault` | `deliberative_speed` | -5.054 |
| `fairness_bypass` | `generated_fault_class_cross_fault` | `fair_allocation` | -0.400 |

## What Changed

Accepted:

- The expanded hand-authored control reaches `control_bundle_ready` on a
  non-Qwen model family.
- The generated benchmark has a perfect SmolLM2 pair-LOO layer at `-2`.
- Qwen7B-to-SmolLM2 same-prompt alignment and mapped pair-LOO transfer are
  accepted for both the control and generated benchmark.

Rejected or caveated:

- The SmolLM2 generated benchmark is not `bundle_ready` because the
  metadata-held-out `privacy_bypass` fold has one negative-margin pair.
- SmolLM2 generated/control direction transfer is not ready. The two benchmark
  directions self-separate but do not yet define the same cross-benchmark axis
  in the non-Qwen activation space.
- This blocks any stronger out-of-family shared-axis claim.

## Residual Finding

The result is real progress but not a clean regime completion. The same
benchmarks are separable in SmolLM2 and linearly align from Qwen7B to SmolLM2,
yet generated/control direction transfer fails in SmolLM2. The next bottleneck
is no longer "does the signal exist outside Qwen?" It is whether the generated
benchmark and the hand-authored procedural-justice control define the same
usable-voice/privacy/repair axis outside Qwen.

## Next Operation

Add a failure-focused generated/control repair or diagnostic pass:

- make cross-benchmark direction transfer report per-pair failure tables
  directly;
- target the SmolLM2 residuals: generated `privacy_bypass::data_choice`,
  generated cross-fault `deliberative_speed` and `fair_allocation`, plus the
  control `privacy_exit`, `appeal_and_evidence`, and `harm_repair` cases that
  fail under the generated direction;
- rerun SmolLM2 layer `-2` before adding more model families.

## Claim Boundary

This run supports a mixed out-of-family activation benchmark finding. It does
not support human behavioral, neural, clinical, deployment, or real-world
social-effect claims. Human validation, Prolific, fMRI, EEG, fNIRS, or
hyperscanning tracks remain parked until generated-text, non-generated,
cross-benchmark, and out-of-family activation gates pass together and are
separately validated.
