# Cross-Fault Lexical Hardening Run

Date: 2026-06-08

## Question

Can a fourth source family reduce the remaining fault-class lexical-only
baseline while preserving practical availability, source diversity, and
activation metadata transfer?

## Discovery-Regime Audit

Current regime:

- Artifact types: constrained repair composers, raw-output rows,
  verifier-filtered candidate rows, source-family bundles, activation prompts,
  Modal activation NPZ files, layer-sweep reports, generated audit manifests,
  and activation metadata-transfer reports.
- Operations: deterministic repair composition, local repair filtering,
  source-family bundling, generated benchmark audit bundles, Modal activation
  extraction, layer sweep, and metadata-held-out activation transfer.
- Gates/verifiers: score, slack preservation, lexical cue leakage, practical
  availability, source diversity, fault/source held-out transfer, activation
  metadata coverage, and activation metadata transfer.
- Known limitations: this remains generated-text benchmark construction plus
  open-model activation analysis. It does not support human behavioral,
  neural, clinical, deployment, or real-world social-effect claims.

Action class:

- Discovery inside the current generated-text/activation regime. The run adds
  `cross_fault_lexical_repair_v1`, a fourth source family that targets learned
  fault-class lexical separability rather than source-level wording diversity.

Gate:

- Acceptance rule: the fourth source must pass strict local repair filtering
  for all ten pairs; the four-source bundle must pass every generated-text gate
  with zero lexical warnings; activation metadata transfer must pass every
  held-out primary fault class at 100% accuracy and positive margins.
- Withheld/rejected rule: any activation layer with leave-one-pair-out errors
  or failed metadata-transfer margins remains rejected, even if mean margins are
  large.

## Implementation

Added a fifth constrained repair composer version:

- `cross_fault_lexical_repair_v1`

The source family deliberately counterbalances the learned fault-class lexical
direction observed after the three-source run:

- pseudo rows include non-decisive positive cue words such as choice, consent,
  share, cooperate, honest truth, listen, and together while still making paths
  costly through proof, approval, alignment, disloyalty, punishment, and
  retaliation conditions;
- genuine rows keep procedural-justice paths usable now while using fair
  repair, trust, respect, evidence access, clear criteria, public records, and
  immediate updates;
- every pair keeps simple lexical cue margin non-positive, so the old lexical
  leakage gate remains meaningful.

Relevant files:

- `src/social_cohesion_vectors/experiments/fault_constrained_repair.py`
- `tests/test_fault_constrained_repair.py`

## Local Repair Filter

Generated composer artifacts:

```text
/tmp/social_cohesion_constrained_repair_20260608/cross_fault_lexical_repair_v1/
```

Strict all-gates filter:

```text
/tmp/social_cohesion_constrained_repair_20260608/cross_fault_lexical_repair_v1/filter_all_gates/
```

Result:

| Metric | Result |
| --- | ---: |
| expected pairs | 10 |
| accepted pairs | 10 |
| accepted raw outputs | 20 |
| rejected candidate pairs | 0 |

## Four-Source Bundle

Combined sources:

- `primary`: accepted tournament-selected generated rows from the constrained
  lexical repair run.
- `source_diverse`: `source_diverse_repair_v1`.
- `lexical_adversarial`: `lexical_adversarial_repair_v1`.
- `cross_fault`: `cross_fault_lexical_repair_v1`.

Bundle without activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/cross_fault_lexical_four_source_v1/
```

Bundle with selected layer `-2` activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/cross_fault_lexical_four_source_v1_activation_layer-2/
```

## Text Audit Results

The four-source bundle passed every generated-text gate with zero warnings:

| Gate | Result |
| --- | --- |
| lexical leakage | ready |
| lexical baseline diagnostic | ready |
| component margin | ready |
| slack preservation | ready |
| practical availability | ready |
| source diversity | ready |
| fault/source held-out transfer | ready |

Key metrics:

| Metric | Result |
| --- | ---: |
| sources | 4 |
| pairwise examples | 40 |
| activation prompts | 80 |
| shared fault groups | 10 |
| cross-source exact duplicates | 0 |
| cross-source near duplicates | 0 |
| max cross-source text similarity | 0.545 |
| availability paths preferring genuine | 184/184 |
| availability path accuracy | 1.000 |
| min availability margin | +0.060 |
| slack pairwise accuracy | 1.000 |
| min slack margin | +0.041 |
| cue-solved pairs | 0/40 |
| mean cue margin | -1.600 |
| best single lexical feature accuracy | 0.638 |

Held-out lexical baseline comparison:

| Split | Three-source lexical-adversarial | Four-source cross-fault |
| --- | ---: | ---: |
| source `lexical_only` | 0.467 | below warning threshold |
| fault class `lexical_only` | 0.933 | 0.700 |

The remaining text/rubric caveat is not lexical: `metrics_only` still reaches
0.925 mean held-out fault-class accuracy, which is expected for examples
filtered by the deterministic score/slack verifier and should not be confused
with a lexical shortcut.

## Activation Results

Modal activation extraction used:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Layer sweep summary:

```text
data/reports/layer_sweep/cross_fault_lexical_four_source_v1__Qwen__Qwen2.5-0.5B-Instruct__summary.md
```

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.975 | +25.858 | rejected for one LOO error |
| -2 | 1.000 | +6.187 | accepted |
| -4 | 1.000 | +5.171 | accepted |
| -8 | 0.975 | +2.911 | rejected for one LOO error |

Layer `-2` was selected because it tied for best leave-one-pair-out accuracy
and had the stronger accepted metadata-transfer mean margin.

Accepted layer `-2` activation metadata transfer:

| Metric | Result |
| --- | ---: |
| folds | 10 |
| test pairs | 40 |
| mean test accuracy | 1.000 |
| mean test margin | +6.090 |
| minimum fold margin | +0.683 |
| metadata coverage readiness | metadata_coverage_ready |
| transfer readiness | transfer_ready |

Layer `-4` also reached `bundle_ready`, with mean metadata-transfer margin
+5.104 and minimum fold margin +0.735.

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | --- |
| status | `bundle_ready` |
| ready steps | 9 |
| not-ready steps | 0 |
| skipped steps | 0 |
| warnings | 0 |

## What Changed

Accepted new content:

- Source-level and fault-class lexical warnings are both cleared for the
  generated benchmark bundle.
- The generated benchmark now has a zero-warning `bundle_ready` manifest with
  activation metadata transfer accepted at Qwen2.5-0.5B layer `-2`.
- The bottleneck has moved from lexical hardening to replication across a
  second model setting and, later, a non-generated benchmark.

Rejected or withheld content:

- Layers `-1` and `-8` are withheld because each has one leave-one-pair-out
  error on the 40-pair bundle.
- Human or neural claims remain withheld.

## Next Operation

The next move should test whether this lexical-controlled activation result is
stable outside the single Qwen2.5-0.5B setting:

- run the same four-source activation bundle on a second model setting, with
  Qwen2.5-7B as the natural next target;
- or add a small hand-authored/control benchmark that uses the same
  procedural-justice availability paths but is not generated by the current
  source-family composers;
- keep the human/neural validation tracks parked until generated-text and
  activation results replicate beyond one model setting.

## Claim Boundary

This run supports only a generated-text benchmark and open-model activation
metadata-transfer claim. It does not support human behavioral, neural,
clinical, deployment, or real-world social-effect claims. Human validation,
Prolific, fMRI, EEG, fNIRS, or hyperscanning tracks remain parked until
generated-text, activation, lexical-control, and cross-setting replication
gates pass together and are separately validated.
