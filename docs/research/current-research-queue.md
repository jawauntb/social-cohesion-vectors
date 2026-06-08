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
extraction, same-family model replication for the generated-text benchmark, the
first small non-generated control benchmark, a first expansion of that control,
or basic out-of-family separability. The newest four-source generated benchmark
reaches `bundle_ready` with zero audit warnings and activation metadata
transfer accepted on both Qwen2.5-0.5B and Qwen2.5-7B. The expanded
hand-authored procedural-justice control reaches `control_bundle_ready` on
Qwen2.5-0.5B, Qwen2.5-7B, and SmolLM2-1.7B. The generated benchmark has a
perfect SmolLM2 pair-LOO layer, Qwen7B-to-SmolLM2 same-prompt alignment is
accepted for both benchmarks, a pooled SmolLM2 generated+control direction
separates both benchmarks, and SmolLM2 bridge training passes held-out
source-family transfer in both domains. The active bottleneck is now minimal
bridge robustness: source-only directions remain domain-specific, but a bridge
direction works, so the next question is how little cross-domain bridge data is
needed before held-out-domain transfer becomes stable.

Recent accepted findings:

- `docs/research/2026-06-08-heldout-domain-bridge-audit.md`: the new
  held-out-domain bridge-training audit closes the SmolLM2 source-family
  transfer failure. Training on one full benchmark domain plus all but one
  source family from the other domain gives source holdout minimum accuracy
  `1.000`, source holdout minimum margin `+10.222`, target holdout minimum
  accuracy `1.000`, target holdout minimum margin `+32.374`, and zero failed
  pairs across eight held-out source-family folds.
- `docs/research/2026-06-08-joint-benchmark-direction-audit.md`: the
  cross-benchmark direction-transfer report now records per-pair margins,
  failed-pair tables, and a pooled joint direction diagnostic. On SmolLM2 layer
  `-2`, the source-only directions still fail cross-benchmark transfer, but the
  pooled generated+control direction separates both benchmarks: generated
  accuracy `1.000` with minimum margin `+21.173`, and control accuracy `1.000`
  with minimum margin `+50.555`. The residual is therefore source-only domain
  generalization, not absence of a shared separable direction.
- `docs/research/2026-06-08-out-of-family-replication-run.md`: the expanded
  hand-authored control replicates outside Qwen on
  `HuggingFaceTB/SmolLM2-1.7B-Instruct` layer `-2`, reaching
  `control_bundle_ready` with zero warnings, activation metadata transfer
  accuracy `1.000`, mean margin `+115.235`, and minimum fold margin `+34.080`.
  The generated benchmark also has a perfect SmolLM2 pair-LOO layer at `-2`,
  but its full activation-transfer bundle is not ready because the
  `privacy_bypass` held-out fold has one negative-margin `data_choice` pair
  from the primary generated source. Qwen7B-to-SmolLM2 same-prompt alignment is
  accepted for both benchmarks, but SmolLM2 generated/control direction
  transfer fails: generated-on-control accuracy `0.750` and control-on-generated
  accuracy `0.950`.
- `docs/research/2026-06-08-procedural-justice-control-expansion-run.md`: the
  non-generated procedural-justice control expanded to 16 hand-authored pairs
  across four source families while preserving zero audit warnings, all-eight
  future-option coverage, availability accuracy `68/68`, lexical cue-solved
  pairs `0/16`, fault-class `lexical_only` held-out accuracy `0.719`, and
  source `lexical_only` held-out accuracy `0.781`. Qwen2.5-0.5B layer `-1` and
  Qwen2.5-7B layer `-2` both reach `control_bundle_ready` with activation
  metadata transfer accepted at `1.000` mean test accuracy over 16 test pairs.
  Same-prompt Qwen0.5B-to-Qwen7B mapped transfer remains accepted in both
  directions. Qwen7B generated/control direction transfer also passes in both
  directions: generated direction on control accuracy `1.000` with minimum
  margin `+2.345`, and control direction on generated accuracy `1.000` with
  minimum margin `+1.700`.
- `docs/research/2026-06-08-procedural-justice-control-run.md`: a
  non-generated procedural-justice control benchmark now exports eight
  hand-authored pairs across two source families and four cases. It clears all
  pre-activation gates with zero audit warnings, availability accuracy `34/34`,
  lexical cue-solved pairs `0/8`, source `lexical_only` held-out accuracy
  `0.750`, and fault-class `lexical_only` held-out accuracy `0.500`.
  Qwen2.5-0.5B layer `-1` and Qwen2.5-7B layer `-2` both reach
  `control_bundle_ready` with activation metadata transfer accepted at `1.000`
  mean test accuracy over eight test pairs. Same-prompt Qwen0.5B-to-Qwen7B
  alignment on the control reaches linear CKA `0.845`, mutual kNN overlap
  `0.781`, and pair-LOO mapped accuracy `1.000` in both directions. Qwen7B
  generated/control direction transfer also passes in both directions:
  generated direction on control accuracy `1.000` with minimum margin `+2.345`,
  and control direction on generated accuracy `1.000` with minimum margin
  `+4.572`.
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

Activation extraction, lexical controls, same-family model replication, the
first non-generated control, its first source expansion, the first Qwen7B
generated/control direction-transfer check, and basic out-of-family separability
are no longer blocked. However, activation results remain text-benchmark claims
until bridge-ablation or minimal-bridge generated/control direction transfer
also survives the out-of-family setting, and human-facing gates are separately
validated.

## Active Objective

Measure minimal out-of-family bridge robustness.

The next operation should find how much generated/control bridge data SmolLM2
needs to preserve held-out source-family transfer. It must still preserve:

- practical availability for all tested future-option paths;
- score and slack separation;
- source diversity without exact or near duplicates;
- source and fault-class `lexical_only` held-out accuracy below the warning
  threshold;
- activation metadata transfer readiness at a held-out metadata level;
- generated/control direction-transfer checks where comparable accepted layers
  exist, with SmolLM2 minimal-bridge transfer as the active gate;
- explicit generated-text and cross-setting claim boundaries.

## Definition Of Done

The active objective is complete when the repo can produce a source-diverse
generated benchmark and non-generated control with:

- generated audit status still `bundle_ready`;
- non-generated control status still `control_bundle_ready`;
- score, slack, component, availability, source diversity, and activation
  metadata transfer gates still passing;
- source and fault-class `lexical_only` warnings cleared;
- no loss of all-eight-path coverage;
- an out-of-family minimal-bridge generated/control direction-transfer pass;
- generated/control direction-transfer readiness for any model setting where
  comparable source-only or held-out-domain layers exist;
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
- `src/social_cohesion_vectors/experiments/cross_benchmark_direction_transfer.py`
- `scripts/run_cross_benchmark_direction_transfer.py`
- `tests/test_cross_benchmark_direction_transfer.py`
- `src/social_cohesion_vectors/experiments/heldout_domain_direction_audit.py`
- `scripts/run_heldout_domain_direction_audit.py`
- `tests/test_heldout_domain_direction_audit.py`

## Next Sequence

1. Reuse the four-source `cross_fault_lexical_repair_v1` bundle as the current
   generated-text baseline.
2. Reuse the `procedural_justice_control_v2` bundle as the current
   non-generated control baseline.
3. Keep the accepted Qwen7B generated/control direction-transfer report on the
   expanded control as the current cross-benchmark alignment baseline.
4. Preserve SmolLM2 layer `-2` as the active out-of-family diagnostic model.
5. Use the per-pair failure tables and pooled joint-direction diagnostic in
   generated/control direction-transfer reports so the failing SmolLM2 cases
   are first-class audit artifacts.
6. Use the held-out-domain bridge-training audit as the current passing bridge
   baseline.
7. Add a minimal-bridge or bridge-ablation audit: vary how many source families
   or pairs from the opposite domain are included before held-out source-family
   transfer passes.
8. Target the current residuals: generated `privacy_bypass::data_choice`,
   generated cross-fault `deliberative_speed` and `fair_allocation`, and the
   control `privacy_exit`, `appeal_and_evidence`, and `harm_repair` rows that
   fail under the generated direction.
9. Rerun SmolLM2 generated/control direction transfer before adding more model
   families.
10. Keep human validation parked until generated, non-generated, cross-setting,
   and out-of-family gates agree.

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
- activation directions transfer across held-out fault classes, at least two
  model settings, and preferably one out-of-family model.
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
