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

The current bottleneck is not GPU scale or activation extraction. It is
candidate generation under the new practical-availability verifier.

Recent accepted findings:

- `docs/research/2026-06-08-lexical-negative-regime-audit.md`: the
  `lexical_negative_v1` generation regime reduced simple lexical cue margin,
  but weakened behavioral and slack separation.
- `docs/research/2026-06-08-availability-audit-first-run.md`: the new
  availability audit shows that both standalone `v4` and the selected
  tournament still fail practical future-path availability. After availability
  was added to candidate selection, the first-20 candidate pool still produced
  only 1/10 selected pairs passing availability and 0/10 core gates.
- `docs/research/2026-06-08-availability-targeted-generation-contract.md`:
  the prompt ledger now supports `availability_targeted_v1` and
  `--availability-priority`. A local smoke check confirms the prioritized
  first-20 prompt records cover all eight future-option paths.

Activation extraction remains blocked because lexical, slack, availability,
source-diversity, and transfer gates do not pass together.

## Active Objective

Run an availability-targeted generation batch and rerun the availability-aware
first-20 tournament with the matching prompt-ledger flags.

The availability audit, availability-aware tournament, and
`availability_targeted_v1` prompt contract now exist. The next move is to test
the candidate pool itself: generated candidates must cover all eight future
paths, mention each declared path in both labels, preserve each path on the
genuine side, and make those same paths practically weaker on the pseudo side.

Selection should preserve the availability dimensions already exposed by the
audit:

- public enough to be accountable;
- timely enough to matter;
- non-retaliatory;
- evidence-accessible when evidence is relevant;
- usable without loyalty tests, tone tests, unanimity requirements, private-only
  channels, or waiting-until-after-alignment rules.

## Definition Of Done

The active objective is complete when the repo can generate and select a
first-20 shard with:

- availability pass rate above the current `1/10` selected baseline;
- core-gate pass rate above the current `0/10` selected baseline;
- explicit coverage for `evidence_access` and `privacy_choice`;
- no regression in score/slack/lexical readiness relative to the current
  selected tournament;
- a dated research note interpreting the first run.

## Likely Files

Implementation should probably follow existing audit patterns:

- `src/social_cohesion_vectors/experiments/fault_authorship_tournament.py`
- `tests/test_fault_authorship_tournament.py`
- `scripts/run_fault_authorship_tournament.py`
- `src/social_cohesion_vectors/experiments/availability_audit.py`
- `src/social_cohesion_vectors/experiments/fault_generation.py`
- `scripts/run_fault_class_modal_generation.py`
- `tests/test_fault_generation.py`
- `tests/test_fault_class_api_generation.py`
- `src/social_cohesion_vectors/experiments/generated_audit_bundle.py`

## Next Sequence

1. Generate a new first-20 candidate batch through Modal HF with
   `--prompt-contract-version availability_targeted_v1` and
   `--availability-priority`.
2. Generate at least one additional candidate batch with the same flags if the
   first batch does not beat the `1/10` availability baseline.
3. Rerun candidate selection with availability as a core gate and the same
   prompt-ledger flags.
4. Compare selected winners and gate counts against the current tournament.
5. Add a dated run note with accepted/rejected candidates and residual failure
   modes.
6. Only after lexical, slack, source-diversity, component, and availability
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
