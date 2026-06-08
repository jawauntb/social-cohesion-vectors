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
joint repair of practical availability, lexical balance, and length for the last
few hard generated-text contrasts.

Recent accepted findings:

- `docs/research/2026-06-08-literature-foundation-audit.md`: the
  literature map leaves the active bottleneck unchanged, but sharpens why it
  matters. Availability is now treated as a procedural-justice and
  effective-voice gate, not just a wording heuristic. The next repair pass
  should preserve timely voice, evidence access, public-enough accountability,
  non-retaliatory exit, and proportionate repair while still passing lexical,
  length, slack, and source-diversity checks.
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
- `docs/research/2026-06-08-availability-targeted-modal-run.md`: five live
  Modal/Qwen candidate chunks restored selected score and slack gates to
  `10/10`, improved availability to `5/10`, core gates to `4/10`, and all gates
  to `2/10`, but activation remains blocked by residual `refusal`, `appeal`,
  `dissent`, and `repair` failures.
- `docs/research/2026-06-08-availability-repair-run.md`: targeted
  `availability_repair_v1` generation improved the selected availability gate
  from `5/10` to `7/10` while preserving score and slack at `10/10`, but core
  stayed at `4/10` and all gates stayed at `2/10`. The remaining blockers are
  `autonomy_after_conflict`, `belonging_norms`, and `fair_allocation`.

Activation extraction remains blocked because lexical, slack, availability,
source-diversity, and transfer gates do not pass together.

## Active Objective

Revise the repair loop so it jointly optimizes practical availability, lexical
balance, and length on the remaining hard contrasts.

The availability audit, availability-aware tournament, `availability_targeted_v1`,
`availability_targeted_v2`, `availability_repair_v1`, and live Modal repair
sweep now exist. Broad generation is no longer the best next move. Simple
repair sampling improved availability but did not improve core/all gates.
The literature audit adds one constraint to the same objective: availability
should be interpreted as procedural justice under pressure. A candidate should
not pass merely because it mentions an abstract future option; it should make
voice, review, evidence access, appeal, and exit usable without loyalty tests,
tone tests, unanimity requirements, retaliation risk, private-only channels, or
waiting-until-after-alignment rules.

- `autonomy_after_conflict`: `dissent`;
- `belonging_norms`: `refusal`, `dissent`;
- `fair_allocation`: `refusal`, `appeal`, `repair`;

The next operation should either add `availability_repair_v2` with stricter
one-paragraph length and lexical-balance constraints, or revise the tournament
selection tuple so availability margin is not dominated by length among
availability-failing candidates. The repair loop should still preserve the
availability dimensions exposed by the audit:

- public enough to be accountable;
- timely enough to matter;
- non-retaliatory;
- evidence-accessible when evidence is relevant;
- usable without loyalty tests, tone tests, unanimity requirements, private-only
  channels, or waiting-until-after-alignment rules.
- public enough for accountability and appeal;
- proportionate enough that repair is not forced confession, coerced apology, or
  institutional compliance theater.

## Definition Of Done

The active objective is complete when the repo can generate and select a
repaired first-20 shard with:

- availability pass rate above the current `7/10` selected baseline;
- core-gate pass rate above the current `4/10` selected baseline;
- all-gate pass rate above the current `2/10` selected baseline;
- selected score and slack gates remain at `10/10`;
- lexical gate recovers above the current `7/10` repaired baseline;
- no loss of all-eight-path coverage;
- a dated research note interpreting accepted and rejected repair candidates.

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

1. Decide whether the next regime move is prompt-side (`availability_repair_v2`)
   or selection-side (availability margin priority among failed-availability
   rows).
2. Apply the smallest change and cover it with tests.
3. Regenerate only `autonomy_after_conflict`, `belonging_norms`, and
   `fair_allocation`.
4. Rerun candidate selection against the five broad chunks plus accepted repair
   chunks.
5. Compare selected winners and gate counts against the current `7/10`
   availability, `4/10` core, `2/10` all-gate, and `7/10` lexical repaired
   baseline.
6. Add a dated run note with accepted/rejected repair candidates and residual
   failure modes.
7. Only after lexical, slack, source-diversity, component, and availability
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
- the planned study measures autonomy, perceived manipulation, psychological
  safety, fairness/procedural justice, and willingness to verify, not only
  which message sounds more cooperative.

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
