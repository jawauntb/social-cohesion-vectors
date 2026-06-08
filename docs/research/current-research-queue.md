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
source-held-out lexical leakage, fault-class lexical leakage, activation
extraction, or same-family model replication for the generated-text benchmark.
The newest four-source generated benchmark reaches `bundle_ready` with zero
audit warnings and activation metadata transfer accepted on both
Qwen2.5-0.5B and Qwen2.5-7B. The active bottleneck is now whether the
procedural-justice distinction survives a small non-generated control benchmark
or, secondarily, a meaningfully different model family.

Recent accepted findings:

- `docs/research/2026-06-08-qwen7b-replication-run.md`: the zero-warning
  four-source generated benchmark replicated on `Qwen/Qwen2.5-7B-Instruct`.
  The full layer `-2` audit bundle is `bundle_ready` with zero warnings,
  activation metadata transfer accepted at `1.000` mean test accuracy over 40
  test pairs, mean margin `+41.175`, and minimum fold margin `+0.212`. Layers
  `-1`, `-4`, and `-8` are withheld because each had one leave-one-pair-out
  error.
- `docs/research/2026-06-08-cross-fault-lexical-hardening-run.md`: a fourth
  cross-fault lexical source family, `cross_fault_lexical_repair_v1`,
  preserved all strict local repair gates and produced a four-source bundle
  with zero audit warnings, availability accuracy `184/184`, cue-solved pairs
  `0/40`, best single lexical feature accuracy `0.638`, fault-class
  `lexical_only` held-out accuracy reduced from `0.933` to `0.700`, and
  Qwen2.5-0.5B layer `-2` activation metadata transfer accepted at `1.000`
  mean test accuracy over 40 test pairs. This is the current best generated
  benchmark baseline.
- `docs/research/2026-06-08-lexical-adversarial-source-family-run.md`: a third
  wording-adversarial source family, `lexical_adversarial_repair_v1`, preserved
  all strict local repair gates and produced a three-source bundle with
  source diversity ready, zero near duplicates, availability accuracy
  `138/138`, source `lexical_only` held-out accuracy reduced from `0.950` to
  `0.467`, and Qwen2.5-0.5B layer `-2` activation metadata transfer accepted
  at `1.000` mean test accuracy over 30 test pairs. The bundle remains
  lexical-caveated because fault-class `lexical_only` held-out accuracy is
  still `0.933`.
- `docs/research/2026-06-08-source-diverse-activation-ready-run.md`: a second
  independent constrained wording family and raw-output source-family bundle
  closed source diversity and source-held-out transfer. The two-source bundle
  passed lexical cue leakage, component, slack, availability, source diversity,
  fault/source transfer, activation metadata coverage, and activation metadata
  transfer at `Qwen/Qwen2.5-0.5B-Instruct` layer `-2`. The full audit status is
  `bundle_ready`, with two explicit lexical warnings. This finding is now
  superseded as the current baseline by the three-source lexical-adversarial
  run.
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

Activation extraction, lexical controls, and same-family model replication are
no longer blocked for the generated-text benchmark. However, activation results
remain generated-text claims until they also pass a non-generated control
benchmark and, preferably, a cross-family model replication.

## Active Objective

Add a small non-generated procedural-justice control benchmark.

The next operation should introduce hand-authored or otherwise non-generated
control examples with the same future-option paths: usable voice, evidence
access, appeal, dissent, non-retaliatory exit, accountability, and
proportionate repair under pressure. It must still preserve:

- practical availability for all tested future-option paths;
- score and slack separation;
- source diversity without exact or near duplicates;
- source and fault-class `lexical_only` held-out accuracy below the warning
  threshold;
- activation metadata transfer readiness at a held-out metadata level;
- explicit generated-text and cross-setting claim boundaries.

## Definition Of Done

The active objective is complete when the repo can produce a source-diverse
generated benchmark with:

- generated audit status still `bundle_ready`;
- score, slack, component, availability, source diversity, and activation
  metadata transfer gates still passing;
- source and fault-class `lexical_only` warnings cleared;
- no loss of all-eight-path coverage;
- a non-generated control benchmark, or a documented failed attempt with clear
  residuals;
- a dated research note interpreting accepted, rejected, and caveated
  replication runs.

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

1. Reuse the four-source `cross_fault_lexical_repair_v1` bundle as the current
   generated-text baseline.
2. Add a small non-generated procedural-justice control benchmark with the same
   future-option paths and explicit generated/non-generated provenance.
3. Run the existing lexical, slack, availability, source-diversity, and
   activation-transfer checks on the control benchmark where applicable.
4. If the control benchmark passes or fails informatively, consider a
   cross-family model replication next.
5. Compare the accepted Qwen0.5B and Qwen7B directions only after the control
   benchmark plan is documented.

## Decision Gates

Move to activation extraction only when all are true:

- lexical leakage readiness passes;
- slack preservation readiness passes;
- availability readiness passes;
- source diversity readiness passes for the intended claim;
- component margins still prefer genuine examples;
- generated data remains documented as generated-text evidence only.

Move to broader Modal generation only when:

- cross-setting replication fails because the generated source family is too
  narrow;
- the current zero-warning `bundle_ready` artifact is preserved as a baseline;
- the new generation plan targets replication robustness, not availability or
  lexical leakage that is already closed for the generated benchmark.

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
