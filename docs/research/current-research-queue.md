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

The current bottleneck is no longer practical availability, source diversity,
or activation extraction. The newest source-diverse generated benchmark reaches
`bundle_ready` with activation metadata transfer accepted, but it still carries
high lexical-baseline warnings. The active bottleneck is now reducing broader
wording-level lexical separability while preserving the accepted text and
activation gates.

Recent accepted findings:

- `docs/research/2026-06-08-source-diverse-activation-ready-run.md`: a second
  independent constrained wording family and raw-output source-family bundle
  closed source diversity and source-held-out transfer. The two-source bundle
  passed lexical cue leakage, component, slack, availability, source diversity,
  fault/source transfer, activation metadata coverage, and activation metadata
  transfer at `Qwen/Qwen2.5-0.5B-Instruct` layer `-2`. The full audit status is
  `bundle_ready`, with two explicit warnings: `lexical_only` still reaches
  `0.950` mean held-out fault-class accuracy and `0.950` mean held-out source
  accuracy.
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

Activation extraction is no longer blocked for the generated-text benchmark.
However, activation results remain lexical-caveated until stronger wording
controls lower the lexical-only held-out baselines.

## Active Objective

Reduce lexical separability while preserving the new source-diverse
activation-ready generated benchmark.

The next operation should add a third wording-adversarial source family or
strengthen the lexical-baseline verifier. The new family should avoid reusing
the shared phrasing that now makes `lexical_only` strong across fault classes
and sources, especially repeated timing, visibility, public-record, immediate
revision, and balanced/proportional phrasing. It must still preserve:

- practical availability for all tested future-option paths;
- score and slack separation;
- source diversity without exact or near duplicates;
- activation metadata transfer readiness at a held-out metadata level;
- explicit generated-text and lexical-caveat claim boundaries.

## Definition Of Done

The active objective is complete when the repo can produce a source-diverse
generated benchmark with:

- generated audit status still `bundle_ready`;
- score, slack, component, availability, source diversity, and activation
  metadata transfer gates still passing;
- source and fault-class `lexical_only` warnings cleared or materially reduced
  below the warning threshold;
- no loss of all-eight-path coverage;
- a dated research note interpreting accepted, rejected, and caveated
  activation runs.

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

1. Add a third wording-adversarial source family or a stricter lexical-baseline
   gate.
2. Filter candidates through score, slack, lexical, availability, length, and
   formatting gates.
3. Rebuild the source-family bundle and rerun all generated benchmark audits.
4. Rerun activation metadata transfer on the accepted layer-sweep candidates.
5. Compare lexical-only fault/source held-out baselines against the current
   `0.950`/`0.950` warning baseline.
6. Add a dated run note with accepted, rejected, and caveated activation
   results.

## Decision Gates

Move to activation extraction only when all are true:

- lexical leakage readiness passes;
- slack preservation readiness passes;
- availability readiness passes;
- source diversity readiness passes for the intended claim;
- component margins still prefer genuine examples;
- generated data remains documented as generated-text evidence only.

Move to broader Modal generation only when:

- lexical-baseline warnings are the active target;
- the candidate source family is designed to lower wording separability, not to
  chase availability again;
- the current `bundle_ready` artifact is preserved as a baseline.

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
