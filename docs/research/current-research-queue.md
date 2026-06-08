# Current Research Queue

Last updated: 2026-06-08

This is the short control surface for staying focused. It should answer three
questions at any point:

1. What is the active research bottleneck?
2. What exact gate decides whether we move on?
3. What is parked so it does not steal attention yet?

Generated data, model outputs, and audit bundles stay out of git. Durable
findings and decisions get summarized here or in dated notes under
`docs/research/`.

## Current State

The current bottleneck is not GPU scale or activation extraction. It is benchmark
validity for generated pseudo-cohesion hard negatives.

The latest accepted finding is in
`docs/research/2026-06-08-lexical-negative-regime-audit.md`:

- The `lexical_negative_v1` generation regime reduced simple lexical cue
  margin, but it weakened behavioral and slack separation.
- The four-candidate tournament selected 10/10 pairs but selected no `v4`
  lexical-negative pairs.
- Activation extraction remains blocked because lexical and slack gates do not
  pass together.

Interpretation: balancing vocabulary alone is too blunt. The next regime should
verify practical availability of future paths directly.

## Active Objective

Build a paired counterfactual availability verifier for generated
pseudo/genuine pairs.

The verifier should judge whether a future path is actually usable, not merely
mentioned. For each matched pair, it should check paths such as appeal, dissent,
exit, evidence access, privacy choice, refusal, repair, and proportional review.

Availability dimensions:

- public enough to be accountable;
- timely enough to matter;
- non-retaliatory;
- evidence-accessible when evidence is relevant;
- usable without loyalty tests, tone tests, unanimity requirements, private-only
  channels, or waiting-until-after-alignment rules.

## Definition Of Done

The active objective is complete when the repo can produce an availability audit
for pairwise generated examples with:

- per-pair path-level availability records;
- positive-minus-negative availability margins;
- group summaries by primary fault class and future-option path;
- a readiness gate that blocks activation when any tested path has nonpositive
  availability margin;
- tests covering pseudo paths that are mentioned but practically taxed;
- a dated research note interpreting the first run.

## Likely Files

Implementation should probably follow existing audit patterns:

- `src/social_cohesion_vectors/experiments/slack_preservation_audit.py`
- `src/social_cohesion_vectors/experiments/generated_audit_bundle.py`
- `scripts/run_slack_preservation_audit.py`
- `tests/test_slack_preservation_audit.py`
- `tests/test_generated_audit_bundle.py`

Create new files only if the availability audit becomes meaningfully different
from the current slack-preservation audit. A likely split is:

- `src/social_cohesion_vectors/experiments/availability_audit.py`
- `scripts/run_availability_audit.py`
- `tests/test_availability_audit.py`

## Next Sequence

1. Implement the availability audit as a typed verifier.
2. Run it on the existing `/tmp` v1/v2/v3/v4 first-20 tournament artifacts.
3. If it catches the `v4` failure mode, add it to the generated audit bundle.
4. Rerun candidate selection with availability as a core gate.
5. Only after lexical, slack, source-diversity, component, and availability
   gates pass together, send a generated shard into activation extraction.

## Decision Gates

Move to activation extraction only when all are true:

- lexical leakage readiness passes;
- slack preservation readiness passes;
- availability readiness passes;
- source diversity readiness passes for the intended claim;
- component margins still prefer genuine examples;
- generated data remains documented as generated-text evidence only.

Move to broader Modal generation only when:

- the verifier catches known first-20 failures;
- the selection policy improves the core-gate pass rate over the previous
  tournament;
- missing future-option paths such as `evidence_access` and `privacy_choice` are
  covered.

Move to human validation only when:

- generated and hand-authored benchmarks agree on the target distinction;
- lexical and availability shortcuts are controlled;
- activation directions transfer across held-out fault classes and at least two
  model settings.

## Parked Tracks

These are valuable, but not the active bottleneck:

- Platonic/cross-model representation analysis beyond the current alignment
  audit.
- Larger Modal generation sweeps.
- SAE feature naming.
- TRIBE or other brain-aligned proxy work.
- Prolific, fMRI, EEG, fNIRS, or hyperscanning validation.
- New broad theory writing.

## Update Protocol

After each serious run or design change:

1. Add or update one dated note under `docs/research/`.
2. Update `Current State` and `Active Objective` here if the bottleneck changed.
3. Keep only one active objective unless there is a true independent parallel
   lane.
4. Put rejected alternatives in the dated note, not only in chat.
5. Keep claim boundaries explicit: generated-text and activation results do not
   imply human behavioral or neural effects.

