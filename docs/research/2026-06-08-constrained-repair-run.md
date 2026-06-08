# Constrained Availability Repair Run

Date: 2026-06-08

## Question

Can verifier-aware constrained composition repair the remaining practical
availability failures after direct Qwen repair-v2 sampling failed to jointly
satisfy availability, length, score, and slack gates?

## Discovery-Regime Audit

Current regime:

- Artifact types: repair-target prompt records, raw-output rows, local
  repair-filter reports, authorship tournament selections, generated benchmark
  audit bundles.
- Operations: prompt-side generation, verifier-aware local filtering, and
  full-ledger tournament selection.
- Gates/verifiers: score margin, slack preservation, lexical leakage, practical
  availability, length, formatting, source diversity, and held-out transfer.
- Known limitations: generated artifacts are single-source and generated-text
  only; activation extraction remains blocked until all benchmark readiness
  gates pass together.

Action class:

- Discovery/regime transition: this adds a deterministic constrained candidate
  composer instead of another model-sampling sweep. The operation emits
  raw-output-like rows that still must pass the same local verifier filter before
  tournament entry.

Gate:

- Acceptance rule: the hard residual repair pairs must pass local score, slack,
  practical availability, length, and formatting gates; the combined tournament
  must improve availability without regressing score and slack.
- Withheld/rejected rule: candidates remain out of the tournament unless the
  local repair filter accepts complete pseudo/genuine pairs.

## Implementation

Added constrained repair composition:

- `src/social_cohesion_vectors/experiments/fault_constrained_repair.py`
- `scripts/compose_constrained_fault_repair_candidates.py`
- `tests/test_fault_constrained_repair.py`

The composer currently targets the three remaining hard contrasts:

- `autonomy_after_conflict`: focus `dissent`;
- `belonging_norms`: focus `refusal,dissent`;
- `fair_allocation`: focus `refusal,appeal,repair`.

It preserves prompt metadata and emits `constrained_repair_composer_version` and
`text_word_count` in raw-output rows so tournament-selected repairs remain
auditable.

## Experiment

Composition artifacts:

```text
/tmp/social_cohesion_constrained_repair_20260608/constrained_repair_v1/
```

Default repair-filter artifacts:

```text
/tmp/social_cohesion_constrained_repair_20260608/constrained_repair_v1_filter_default/
```

Combined tournament artifacts:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_constrained_v1_limit20/
```

The tournament compared:

- the five availability-targeted chunks;
- accepted repair chunks `repair_000` and `repair_001`;
- locally accepted `constrained_repair_v1` rows.

The same prioritized first-20 prompt-record scope was used as the accepted
repair baseline.

## Local Filter Results

The constrained composer produced six rows, three complete pairs, and all six
outputs were within the 55-75-word target range.

The default local repair filter accepted all three hard residual pairs:

| Pair | Score | Slack | Availability | Gates |
| --- | ---: | ---: | ---: | ---: |
| `autonomy_after_conflict__neighborhood_forum` | +0.077 | +0.070 | +0.480 | 5/6 |
| `belonging_norms__neighborhood_forum` | +0.076 | +0.111 | +0.580 | 6/6 |
| `fair_allocation__neighborhood_forum` | +0.132 | +0.223 | +0.940 | 5/6 |

This is the first run where all three residual hard pairs cleared score, slack,
availability, length, and formatting before tournament entry.

## Tournament Results

Combined tournament after adding constrained repair:

| Metric | Before constrained repair | After constrained repair |
| --- | ---: | ---: |
| selected pairs | 10/10 | 10/10 |
| score gate | 10/10 | 10/10 |
| slack gate | 10/10 | 10/10 |
| lexical gate | 7/10 | 5/10 |
| availability gate | 7/10 | 10/10 |
| core gates | 4/10 | 5/10 |
| all gates | 2/10 | 3/10 |
| selected availability path accuracy | 40/46 | 46/46 |
| mean selected availability margin | +0.418 | +0.432 |
| min selected availability margin | -0.150 | +0.150 |

Selected constrained winners:

- `autonomy_after_conflict`: `constrained_repair_v1`;
- `belonging_norms`: `constrained_repair_v1`;
- `fair_allocation`: `constrained_repair_v1`.

The practical availability audit passed:

| Availability metric | Result |
| --- | ---: |
| pairs | 10 |
| tested paths | 46 |
| required options covered | 8/8 |
| paths preferring genuine | 46/46 |
| path pairwise accuracy | 1.000 |
| mean availability margin | +0.562 |
| min availability margin | +0.150 |
| readiness | availability_ready |

## Remaining Blockers

The full generated benchmark audit bundle is still not ready for activation
claims:

| Gate | Status | Reason |
| --- | --- | --- |
| lexical leakage | not ready | cue-solved rate `0.500`; failed groups include `consent_bypass`, `deliberation_bypass`, `fairness_bypass`, `forced_forgiveness`, and `punitive_accountability` |
| source diversity | not ready | one generated source only |
| source-held-out transfer | not ready | held-out source folds are incomplete with one source |

The improvement is therefore a practical-availability breakthrough inside the
generated benchmark construction loop, not activation readiness.

## Residual Content

Explained by the old regime:

- Broad availability-targeted generation and repair-v1 sampling can discover
  some useful candidates but cannot reliably satisfy all path, length, score,
  and slack gates.
- Repair-v2 prompt tightening alone did not fix the hard residuals.

New content outside the old regime:

- A constrained composer plus verifier-aware local filter can construct
  complete candidate pairs for the residual hard contrasts that pass local
  availability, length, score, and slack gates.
- The combined tournament now has zero remaining failed practical availability
  paths across all tested future-option paths.

Retractions or supersessions:

- The previous bottleneck was not intrinsic impossibility of the residual
  contrasts. It was an operation mismatch: direct sampling could not reliably
  preserve the path-level tax/availability structure under length and slack
  constraints.
- Activation extraction should remain blocked because lexical leakage and
  source-diversity gates still fail.

## Next Operation

Consolidate the constrained-repair gain by targeting the blockers it exposed:

- add lexical-balanced constrained variants for the cue-leaky selected pairs;
- create a second independent source family before any source-transfer claim;
- rerun the same audit bundle after lexical and source-diverse candidates are
  available;
- only restart activation extraction after availability, lexical, source
  diversity, and transfer gates all pass together.

## Claim Boundary

This is generated-text benchmark construction only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
