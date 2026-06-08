# Lexical-Balanced Constrained Repair Run

Date: 2026-06-08

## Question

After constrained repair closed practical availability, can a second constrained
candidate operation close lexical leakage and length gates without reopening
score, slack, or path-level availability failures?

## Discovery-Regime Audit

Current regime:

- Artifact types: deterministic constrained raw-output rows, verifier-filtered
  candidate rows, full authorship tournament selections, generated benchmark
  audit bundles.
- Operations: constrained candidate composition, local all-gates repair
  filtering, and full-ledger tournament selection.
- Gates/verifiers: score, slack preservation, lexical leakage, practical
  availability, target length, formatting, source diversity, held-out transfer,
  and activation metadata transfer.
- Known limitations: the selected examples are still generated text from one
  source family; no activation extraction or human/neural claim is supported.

Action class:

- Discovery/regime transition: adds `lexical_balanced_repair_v1`, a constrained
  composer version that targets the surface-cue and length residuals exposed by
  the previous constrained availability repair. It emits candidate rows that are
  rejected unless they pass all local gates, including the lexical gate.

Gate:

- Acceptance rule: locally filtered lexical-balanced candidates must pass score,
  slack, lexical, availability, length, and formatting gates before tournament
  entry; the full tournament must select 10/10 pairs with all gates passing.
- Withheld/rejected rule: no lexical-balanced row enters the comparison
  tournament unless the local repair filter accepts its complete pseudo/genuine
  pair.

## Implementation

Extended constrained repair composition with a second composer version:

- `lexical_balanced_repair_v1`

The composer supports seven targeted pairs:

- `accountability_after_harm`;
- `autonomy_after_conflict`;
- `deliberative_speed`;
- `dissent_after_mistake`;
- `expert_review`;
- `fair_allocation`;
- `forgiveness_after_harm`.

It keeps the earlier `constrained_repair_v1` availability repair intact, while
using scorer-positive but leakage-neutral wording such as `revision`,
`transparent`, `balanced`, `visible`, and `timely` to preserve substantive gates
without letting the genuine side win through the simple prosocial cue list.

## Experiment

Expanded lexical-balanced composition artifacts:

```text
/tmp/social_cohesion_constrained_repair_20260608/lexical_balanced_repair_v1_expanded/
```

Strict all-gates local filter:

```text
/tmp/social_cohesion_constrained_repair_20260608/lexical_balanced_repair_v1_expanded_filter_all_gates/
```

Combined tournament:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_constrained_lexical_v1_expanded_limit20/
```

The tournament compared:

- the five broad availability-targeted chunks;
- accepted repair chunks `repair_000` and `repair_001`;
- `constrained_repair_v1`;
- expanded `lexical_balanced_repair_v1`.

The same prioritized first-20-record scope was used as the previous constrained
repair tournament.

## Local Filter Results

The expanded lexical-balanced composer produced 14 rows, seven complete pairs,
and all 14 outputs were within the 55-75-word range.

The strict local repair filter accepted all seven pairs:

| Metric | Result |
| --- | ---: |
| expected pairs | 7 |
| evaluated candidate pairs | 7 |
| accepted pairs | 7 |
| accepted raw outputs | 14 |
| rejected candidate pairs | 0 |

The required gates were:

- score prefers genuine;
- slack prefers genuine;
- lexical cues do not solve genuine;
- practical availability prefers genuine;
- length is in target range;
- formatting is clean.

## Tournament Results

Combined tournament after adding expanded lexical-balanced repair:

| Metric | Before lexical-balanced repair | After expanded lexical-balanced repair |
| --- | ---: | ---: |
| selected pairs | 10/10 | 10/10 |
| score gate | 10/10 | 10/10 |
| slack gate | 10/10 | 10/10 |
| lexical gate | 5/10 | 10/10 |
| availability gate | 10/10 | 10/10 |
| core gates | 5/10 | 10/10 |
| all gates | 3/10 | 10/10 |
| selected availability path accuracy | 46/46 | 46/46 |
| mean selected score margin | +0.080 | +0.068 |
| mean selected slack margin | +0.117 | +0.174 |
| mean selected cue margin | +0.500 | -2.300 |
| mean selected availability margin | +0.432 | +0.496 |
| min availability audit margin | +0.150 | +0.060 |

Selected lexical-balanced winners:

- `accountability_after_harm`: score `+0.061`, slack `+0.251`,
  cue `-5.000`, availability `+0.500`;
- `autonomy_after_conflict`: score `+0.047`, slack `+0.070`,
  cue `+0.000`, availability `+0.270`;
- `deliberative_speed`: score `+0.120`, slack `+0.334`,
  cue `-5.000`, availability `+0.600`;
- `dissent_after_mistake`: score `+0.094`, slack `+0.181`,
  cue `-3.000`, availability `+0.480`;
- `expert_review`: score `+0.093`, slack `+0.293`,
  cue `-2.000`, availability `+0.940`;
- `fair_allocation`: score `+0.012`, slack `+0.111`,
  cue `-3.000`, availability `+0.670`;
- `forgiveness_after_harm`: score `+0.106`, slack `+0.251`,
  cue `-3.000`, availability `+0.060`.

Tournament status changed to:

```text
selected_dataset_ready_for_audits
```

## Audit Bundle Results

The generated benchmark audit bundle still blocks activation claims, but the
reason changed.

Ready steps:

- lexical leakage: ready;
- lexical baseline diagnostic: ready;
- component margin audit: ready;
- slack preservation audit: ready;
- availability audit: ready.

Not-ready steps:

- source diversity audit;
- source-held-out transfer.

Skipped:

- activation metadata transfer, because no activation NPZ was provided.

Lexical leakage now passes:

| Metric | Result |
| --- | ---: |
| cue-solved pairs | 0/10 |
| cue-solved rate | 0.000 |
| cue-tied pairs | 2 |
| cue-inverted pairs | 8 |
| mean cue margin | -2.300 |

Availability remains fully green:

| Metric | Result |
| --- | ---: |
| tested paths | 46 |
| required options covered | 8/8 |
| paths preferring genuine | 46/46 |
| path pairwise accuracy | 1.000 |
| mean availability margin | +0.532 |
| min availability margin | +0.060 |

The bundle emits one warning:

```text
fault_class_lexical_baseline_high
```

`lexical_only` reaches `0.900` mean held-out fault-class accuracy despite the
simple cue-leakage gate. This means the simple cue gate is fixed, but broader
wording-level lexical separability remains too strong for activation claims.

## Remaining Blockers

The next blocker is no longer practical availability, lexical cue leakage,
score, slack, length, or formatting. The remaining blockers are:

- only one generated source family;
- source-held-out folds are incomplete;
- wording-level lexical baselines remain high;
- activation metadata has not been extracted.

The next regime should therefore add a genuinely independent source/wording
family, not keep polishing the same selected first-source texts.

## Residual Content

Explained by the old regime:

- Direct Qwen sampling could not jointly satisfy availability, length, score,
  slack, and lexical gates.
- Availability-only constrained repair solved path availability but exposed
  lexical and length residuals.

New content outside the old regime:

- A lexical-balanced constrained composer plus strict local filtering can
  produce candidates that pass all six tournament gates.
- The first-20 generated benchmark can now select 10/10 pairs with score,
  slack, lexical, availability, length, and formatting gates all passing.

Retractions or supersessions:

- The previous bottleneck was not intrinsic incompatibility among availability,
  lexical balance, and length. It was lack of a constrained operation that could
  explicitly satisfy all three while preserving score and slack.
- Activation extraction should still remain blocked because source diversity,
  source-held-out transfer, lexical-baseline warning, and activation metadata
  are not resolved.

## Next Operation

Build an independent source/wording family with the same fault contrasts and
the same future-option paths, then rerun the audit bundle with both source
families included. Acceptance should require:

- at least two source groups;
- shared fault groups across sources;
- no cross-source duplicates or near duplicates;
- source-held-out transfer readiness;
- lower wording-level lexical baseline dependence;
- all prior score, slack, lexical cue, availability, length, and formatting
  gates preserved.

Only after those pass together should activation extraction resume.

## Claim Boundary

This is generated-text benchmark construction only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
