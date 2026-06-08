# Lexical-Adversarial Source-Family Run

Date: 2026-06-08

## Question

Can a third wording-adversarial source family reduce source-held-out lexical
separability while preserving the generated benchmark's score, slack,
practical availability, source diversity, and activation metadata-transfer
gates?

## Discovery-Regime Audit

Current regime:

- Artifact types: constrained repair composers, raw-output rows, repair-filter
  reports, source-family bundles, activation prompts, Modal activation NPZs,
  layer-sweep reports, audit bundle manifests, and metadata-transfer reports.
- Operations: deterministic repair composition, verifier-aware repair
  filtering, source-family bundling, generated benchmark audits, Modal
  activation extraction, layer sweep, and activation metadata transfer.
- Gates/verifiers: score, slack preservation, lexical cue leakage, practical
  availability, source diversity, fault/source held-out transfer, activation
  metadata coverage, and activation metadata transfer.
- Known limitations: this is generated-text benchmark construction plus
  open-model activation analysis. It does not support human behavioral,
  neural, clinical, deployment, or real-world social-effect claims.

Action class:

- Discovery/regime transition inside the generated-text and activation regime.
  The run adds a third source family, `lexical_adversarial_repair_v1`, designed
  to break source-level wording shortcuts after `source_diverse_repair_v1`
  exposed high source-held-out lexical transfer.

Gate:

- Acceptance rule: the new family must pass strict repair filtering for all ten
  hard source-family pairs; the combined three-source bundle must pass all
  generated-text gates; activation metadata transfer must pass every held-out
  primary fault class with 100% test accuracy and positive margins.
- Caveat rule: if fault-class lexical-only transfer remains high, activation
  results are still lexical-caveated even when activation transfer passes.

## Implementation

Added a fourth constrained repair composer version:

- `lexical_adversarial_repair_v1`

The new source family keeps the same procedural-justice paths as the accepted
source-diverse family, but changes the prose geometry and deliberately
counterbalances surface lexical cues:

- pseudo rows tax future paths through delayed approval, private objections,
  checking costs, disloyalty framing, and impact tests;
- genuine rows keep the same paths usable now through public records,
  evidence access, immediate revision or repair, non-retaliation, and
  proportional review;
- wording is intentionally less similar to `source_diverse_repair_v1`.

Relevant files:

- `src/social_cohesion_vectors/experiments/fault_constrained_repair.py`
- `tests/test_fault_constrained_repair.py`

## Local Repair Filter

Generated composer artifacts:

```text
/tmp/social_cohesion_constrained_repair_20260608/lexical_adversarial_repair_v1/
```

Strict all-gates filter:

```text
/tmp/social_cohesion_constrained_repair_20260608/lexical_adversarial_repair_v1/filter_all_gates/
```

Result:

| Metric | Result |
| --- | ---: |
| expected pairs | 10 |
| accepted pairs | 10 |
| accepted raw outputs | 20 |
| rejected candidate pairs | 0 |

## Three-Source Bundle

Combined sources:

- `primary`: accepted tournament-selected generated rows from the constrained
  lexical repair run.
- `source_diverse`: `source_diverse_repair_v1`.
- `lexical_adversarial`: `lexical_adversarial_repair_v1`.

Bundle without activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/lexical_adversarial_three_source_v1/
```

Bundle with selected layer `-2` activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/lexical_adversarial_three_source_v1_activation_layer-2/
```

## Text Audit Results

The three-source bundle passed all non-activation generated benchmark gates:

| Gate | Result |
| --- | --- |
| lexical leakage | ready |
| component margin | ready |
| slack preservation | ready |
| practical availability | ready |
| source diversity | ready |
| fault/source held-out transfer | ready |

Key metrics:

| Metric | Result |
| --- | ---: |
| sources | 3 |
| pairwise examples | 30 |
| activation prompts | 60 |
| shared fault groups | 10 |
| cross-source exact duplicates | 0 |
| cross-source near duplicates | 0 |
| max cross-source text similarity | 0.545 |
| availability paths preferring genuine | 138/138 |
| availability path accuracy | 1.000 |
| min availability margin | +0.060 |
| slack pairwise accuracy | 1.000 |
| min slack margin | +0.070 |
| cue-solved pairs | 0/30 |
| mean cue margin | -1.800 |
| best single lexical feature accuracy | 0.750 |

Held-out lexical baseline comparison:

| Split | Previous two-source | Three-source lexical-adversarial |
| --- | ---: | ---: |
| source `lexical_only` | 0.950 | 0.467 |
| fault class `lexical_only` | 0.950 | 0.933 |

This clears the source-held-out lexical warning and materially reduces the
single-feature lexical diagnostic, but the fault-class lexical warning remains.

## Activation Results

Modal activation extraction used:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Layer sweep summary:

```text
data/reports/layer_sweep/lexical_adversarial_three_source_v1__Qwen__Qwen2.5-0.5B-Instruct__summary.md
```

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.967 | +27.413 | rejected for one LOO error |
| -2 | 1.000 | +6.573 | accepted |
| -4 | 1.000 | +5.677 | accepted |
| -8 | 0.967 | +3.299 | rejected for one LOO error |

Layer `-2` was selected because it tied for best leave-one-pair-out accuracy
and had the stronger accepted metadata-transfer mean margin.

Accepted layer `-2` activation metadata transfer:

| Metric | Result |
| --- | ---: |
| folds | 10 |
| test pairs | 30 |
| mean test accuracy | 1.000 |
| mean test margin | +6.424 |
| minimum fold margin | +0.807 |
| metadata coverage readiness | metadata_coverage_ready |
| transfer readiness | transfer_ready |

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | --- |
| status | `bundle_ready` |
| ready steps | 9 |
| not-ready steps | 0 |
| skipped steps | 0 |
| warnings | 1 |

Layer `-4` also reached `bundle_ready`, with mean metadata-transfer margin
+5.564.

## What Changed

Accepted new content:

- A third source family can preserve all local repair gates while reducing
  source-held-out lexical-only accuracy from 0.950 to 0.467.
- Source diversity remains strong with zero near duplicates and max
  cross-source similarity 0.545.
- The three-source bundle remains activation-ready at Qwen2.5-0.5B layer `-2`.

Remaining caveat:

- Fault-class lexical-only transfer remains high at 0.933. This means the
  representation result is still lexical-caveated for fault-class claims. The
  next hardening pass should target cross-fault wording patterns, not source
  wording diversity alone.

## Next Operation

The next move should narrow from source-level lexical control to fault-class
lexical control:

- add a fourth source family that varies the path vocabulary within each fault
  class more aggressively;
- or add a lexical-heldout adversarial selection gate that rejects candidates
  when the learned fault-class lexical baseline remains above threshold;
- then rerun the source-family bundle and activation metadata transfer.

## Claim Boundary

This run supports only a generated-text benchmark and open-model activation
metadata-transfer claim under explicit lexical caveats. It does not support
human behavioral, neural, clinical, deployment, or real-world social-effect
claims. Human validation, Prolific, fMRI, EEG, fNIRS, or hyperscanning tracks
remain parked until generated-text, activation, and lexical-control gates pass
together and are separately validated.
