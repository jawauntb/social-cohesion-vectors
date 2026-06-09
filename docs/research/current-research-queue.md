# Current Research Queue

Last updated: 2026-06-09

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
basic out-of-family separability, source-family bridge robustness, pair-count
bridge robustness, first-pass bridge-set diagnosis, intentional six-pair
bridge-set sufficiency, same-model constructed bridge direction comparison, or
first-pass cross-model fresh-control transport. The newest four-source
generated benchmark reaches `bundle_ready` with zero audit warnings and
activation metadata transfer accepted on both Qwen2.5-0.5B and Qwen2.5-7B. The
expanded hand-authored procedural-justice control reaches
`control_bundle_ready` on Qwen2.5-0.5B, Qwen2.5-7B, and SmolLM2-1.7B. The
generated benchmark has a perfect SmolLM2 pair-LOO layer,
Qwen7B-to-SmolLM2 same-prompt alignment is
accepted for both benchmarks, a pooled SmolLM2 generated+control direction
separates both benchmarks, and SmolLM2 bridge training passes held-out
source-family transfer in both domains. A same-domain source-family bridge
ablation now passes with asymmetric requirements: generated/source holdouts need
one bridge source family, while control/target holdouts need two. A
path-stratified pair bridge ablation now finds a six-pair bridge threshold: the
control/target side is exact at six bridge pairs, while the generated/source
side is sampled-stable at six. First-pass bridge-set diagnosis shows
all-eight future-option coverage is not sufficient by itself: the exact
target/control five-pair failures can already cover every future-option path.
An intentional six-pair bridge-set constructor now passes by covering
procedural paths, non-held-out source families, and case/fault families. The
constructed six-pair bridge directions now also separate both full SmolLM2
benchmarks and align positively with the full joint direction. Cross-model
constructed-bridge transport now passes bidirectionally between Qwen7B and
SmolLM2 with held-out pair groups excluded from each fold's alignment map.
Fresh hand-authored procedural-justice control prompts also transport in both
directions. Recovered-manifest same-model diagnostics show Qwen7B constructed
bridges pass fresh repair-v2 prompts with a thin minimum margin `+0.178`, while
SmolLM2 constructed bridges fail the same fresh generated slice with minimum
margin `-11.273`. Fresh augmentation and leave-one-fresh-pair-out repair the
slice in Qwen7B but not in SmolLM2. Pair-level geometry shows the hardest
SmolLM2 residual is fresh `accountability_after_harm`: its fresh delta is nearly
orthogonal to same-base source accountability deltas, positive under
fresh-only, but still inverted under original and full fresh-augmented
directions. A strict hand-authored accountability mini-control now clears
scoped text gates and passes SmolLM2/Qwen7B activation gates: the original
source+target direction already separates all four mini-control pairs in both
models, and every mini-control pair is `not_hard_residual` under pair geometry.
The active residual is therefore not broad accountability/procedural-justice
failure; it is a narrower generated-style or generated-provenance activation
pocket around the hard fresh `accountability_after_harm` pair. A
source-style intervention now shows the pocket is even narrower: SmolLM2,
Qwen2.5-0.5B, and TinyLlama-1.1B invert the recovered generated reference while
separating all five clean hand-authored style variants, including a
generated-like paragraph; Qwen2.5-7B separates both the generated reference and
the clean variants. A first deterministic perturbation ladder now repairs the
fresh-source slice in all four model spaces when used as local augmentation,
with thin but positive Qwen0.5B and TinyLlama margins. Under the original
source+target direction, removing the generated reference's first positive
sentence flips SmolLM2 positive, while Qwen0.5B and TinyLlama move closest to
zero under explicit pseudo-condition or combined edits but do not flip.

Recent accepted findings:

- `docs/research/2026-06-09-accountability-perturbation-ladder.md`: added a
  deterministic perturbation exporter for the external generated
  `accountability_after_harm` reference. All seven perturbations preserve
  scoped practical availability (`42/42`, minimum margin `+0.150`), while
  lexical diagnostics remain intentionally caveated because the source artifact
  is the known leaky generated residual. Original source+target margins show
  SmolLM2 flips only when the first positive-side sentence is removed
  (`-9.353` -> `+3.727`), Qwen0.5B and TinyLlama remain negative but move
  closest to zero under combined/condition edits, and Qwen7B stays positive for
  every perturbation. Full perturbation-ladder augmentation repairs all four
  model spaces: SmolLM2 fresh LOO minimum `+22.746`, Qwen0.5B `+0.988`,
  Qwen7B `+14.288`, and TinyLlama `+0.071`.
- `docs/research/2026-06-09-tinyllama-style-replication.md`: replicated the
  accountability source-style intervention on
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` layer `-2`. TinyLlama matches the
  small-model pattern: hand-style leave-one-out stays positive with minimum
  margin `+1.293`, but the full fresh-source slice fails on the generated
  reference with minimum margin `-0.632`. Pair geometry marks only the
  generated reference as `hard_pair_geometry_residual`
  (`-0.680/-0.433/+1.674/-0.632` for original/full/fresh-only/leave-focus-out);
  all five clean style variants are `not_hard_residual`. The cross-model
  pattern is now Qwen7B positive on both generated and clean variants, while
  SmolLM2, Qwen0.5B, and TinyLlama fail only the recovered generated reference.
- `docs/research/2026-06-09-accountability-style-intervention-audit.md`:
  added a source-style intervention exporter and live three-model diagnostic
  around the hard generated `accountability_after_harm` residual. The
  hand-authored subset clears text gates with availability `30/30`, cue-solved
  pairs `0/5`, cue-tied pairs `5/5`, and term/length best single-feature
  accuracy `0.500`. SmolLM2 and Qwen2.5-0.5B fail only the generated reference:
  SmolLM2 pair geometry is generated reference `-9.761/-0.270/+67.524/-7.710`
  for original/full/fresh-only/leave-focus-out, while all five clean style
  variants are strongly positive; Qwen0.5B shows the same pattern with thinner
  margins. Qwen7B is positive on the generated reference and all variants. The
  residual now looks like a small-model generated-reference off-manifold pocket,
  not accountability content or source format in general.
- `docs/research/2026-06-09-accountability-mini-control-audit.md`: added a
  strict non-generated accountability mini-control with four hand-authored
  source formats. It clears scoped pre-activation gates: availability `28/28`,
  minimum availability margin `+0.560`, lexical cue-solved pairs `0/4`, and
  term/length best single-feature accuracy `0.500`. Modal activations on
  SmolLM2 and Qwen7B layer `-2` show the mini-control is not a hard residual:
  the original source+target direction already separates it at
  `1.000/+46.155/0` in SmolLM2 and `1.000/+23.796/0` in Qwen7B. Fresh
  augmentation and leave-one-mini-control-pair-out also pass with large
  positive margins in both models. This narrows the prior SmolLM2
  `accountability_after_harm` failure to a generated-text geometry pocket
  rather than broad procedural-accountability semantics.
- `docs/research/2026-06-09-accountability-pair-geometry-audit.md`: added a
  pair-level geometry audit for the hard fresh `accountability_after_harm`
  residual. SmolLM2 marks the focus pair negative under original
  source+target (`-9.761`), full fresh-augmented (`-4.528`), and
  leave-focus-out (`-11.959`) directions, but positive under fresh-only
  (`+29.843`). Qwen7B is positive under all four directions. Same-base source
  accountability variants stay strongly positive in SmolLM2, but the fresh
  focus delta is nearly orthogonal to them (mean cosine `+0.009`), so the
  residual is a local fresh-subcase geometry mismatch rather than broad
  accountability failure.
- `docs/research/2026-06-09-fresh-augmented-direction-audit.md`: added a
  fresh-augmented direction audit with leave-one-fresh-pair-out folds. Qwen7B
  passes full fresh augmentation and fresh LOO. SmolLM2 preserves original
  generated source, procedural target, and fresh hand-authored target margins,
  but still fails fresh generated prompts: all-fresh augmentation reaches only
  `0.900/-4.528`, and fresh LOO has minimum margin `-11.959`. Removing the
  known content-bad `belonging_norms` augmentation row or augmenting only the
  two clean residual pairs does not fix held-out `accountability_after_harm`.
  The active hard residual is now fresh `accountability_after_harm` in SmolLM2.
- `docs/research/2026-06-09-fresh-generated-residual-diagnostic.md`: added a
  residual diagnostic over the fresh repair-v2 failures. SmolLM2 has three
  failing fresh pairs: `accountability_after_harm`, `belonging_norms`, and
  `dissent_after_mistake`. Two are clean activation residuals under this
  diagnostic because they have positive practical-availability margins and
  strong same-base original source margins in SmolLM2. `belonging_norms` is
  mixed and should be repaired or withheld because its fresh pair has negative
  practical availability. Qwen7B has no residual on the same fresh slice.
- `docs/research/2026-06-09-activation-manifest-recovery-diagnostic.md`:
  recovered pair manifests from activation NPZ payloads to unblock the
  fresh-generated bridge diagnostic after `/tmp` generation artifacts had been
  cleaned. Recovered coverage is 40 generated source pairs, 16 procedural-v2
  control pairs, 10 fresh generated repair-v2 pairs, and 8 procedural-v1 fresh
  control pairs. The same-model diagnostic explains the prior transport
  asymmetry: Qwen7B constructed bridges pass fresh generated prompts at
  `1.000/+0.178`, while SmolLM2 constructed bridges fail at `0.700/-11.273`;
  fresh hand-authored controls pass in both spaces. Failures concentrate on
  `accountability_after_harm`, `belonging_norms`, and
  `dissent_after_mistake`.
- `docs/research/2026-06-09-fresh-generated-bridge-diagnostic.md`: added the
  fresh-generated bridge diagnostic scaffold. The new audit compares
  source-only, fresh-source-only, source+fresh-source, and constructed bridge
  directions inside one activation space, then evaluates each direction on
  original source, original target, fresh source, and fresh target prompt
  slices with failed-pair tables. No accepted live Qwen7B/SmolLM2 diagnostic
  was run in this branch because the exact generated pair manifests from the
  previous `/tmp` runs had been cleaned before the 2026-06-09 follow-up; the
  activation NPZ files alone do not preserve the real `slack_options_tested`
  bridge metadata.
- `docs/research/2026-06-08-fresh-prompt-bridge-transport.md`: the
  cross-model bridge transport audit now supports fresh source and fresh target
  prompt slices withheld from alignment-map training rows. Same-prompt source,
  target, and leave-held-out map checks still pass bidirectionally, and fresh
  hand-authored control-v1 prompts pass in both directions. The fresh generated
  repair-v2 source slice is withheld: Qwen7B -> SmolLM2 reaches only
  `0.700` fresh-source accuracy with minimum margin `-11.273`, while
  SmolLM2 -> Qwen7B reaches `1.000` fresh-source accuracy with a thin minimum
  margin `+0.178`. Failures concentrate on `accountability_after_harm`,
  `belonging_norms`, and `dissent_after_mistake`.
- `docs/research/2026-06-08-cross-model-bridge-transport.md`: Qwen7B layer `-2`
  constructed bridge directions pass the same-model comparison with minimum
  joint cosine `+0.794`, source minimum margin `+5.176`, target minimum margin
  `+10.213`, and zero failures. Qwen7B-to-SmolLM2 bridge transport then passes
  bidirectionally over `112` shared generated/control prompts with combined
  linear CKA `0.908` and mutual kNN overlap `0.732`. Mapped directions preserve
  accuracy `1.000` with zero failed directions. The stricter leave-held-out map
  diagnostic also passes: Qwen7B -> SmolLM2 minimum held-out margin is
  `+11.091`, and SmolLM2 -> Qwen7B minimum held-out margin is `+5.877`.
- `docs/research/2026-06-08-bridge-direction-comparison.md`: constructed
  six-pair bridge directions now pass the broader SmolLM2 same-model
  comparison. Source-only transfer still fails, with source-on-control accuracy
  `0.750` and target-on-generated accuracy `0.950`, so the check is not
  trivial. The full joint direction separates both benchmarks with minimum
  margins `+21.173` and `+50.555`. All eight constructed bridge directions
  separate both full benchmarks; their minimum joint cosine is `+0.756`, source
  minimum margin is `+11.090`, target minimum margin is `+12.044`, and there
  are zero failed constructed directions.
- `docs/research/2026-06-08-bridge-set-sufficiency-audit.md`: an intentional
  six-pair bridge-set constructor now passes on SmolLM2 layer `-2`. It greedily
  maximizes new `slack_options_tested` paths plus `source` and
  `primary_fault_class` coverage. Generated/source held-out folds reach
  accuracy `1.000`, minimum margin `+11.090`, path-complete folds `4/4`, and
  zero failures. Control/target held-out folds reach accuracy `1.000`, minimum
  margin `+12.044`, path-complete folds `4/4`, and zero failures. This closes
  the immediate six-pair bridge-set sufficiency gate for the current SmolLM2
  diagnostic.
- `docs/research/2026-06-08-bridge-set-diagnosis.md`: first-pass diagnosis of
  the target-exact pair-bridge report finds that all `35` target/control
  five-pair failures concentrate on the held-out
  `privacy_exit::hand_authored_incident_log_v1` pair. The weakest failed
  five-pair bridge already covers all eight future-option paths, so
  future-option coverage alone is not sufficient. Failing five-pair sets
  overuse case-notes/meeting-minutes rows and underuse policy-review and
  repair/privacy alternatives. The generated/source sampled five-pair failure
  is different: it misses `appeal` and `proportional_review`, and adding
  `fair_allocation` closes the weakest sampled six-pair case.
- `docs/research/2026-06-08-pair-bridge-ablation-audit.md`: the pair-level
  path-stratified bridge audit uses `slack_options_tested` to sample bridge
  subsets by procedural future-option coverage. On SmolLM2 layer `-2`, the
  generated/source side has a sampled-stable six-pair threshold with minimum
  margin `+3.412`; the control/target side has an exact six-pair threshold
  after evaluating all `16,384` control-side bridge subsets, with minimum
  margin `+0.652`. All control-side bridge subsets with five or fewer pairs
  still fail at least one held-out source-family fold. The residual is no
  longer pair count, but which six-pair bridge sets are robust and
  path-complete.
- `docs/research/2026-06-08-minimal-bridge-ablation-audit.md`: the
  minimal-bridge source-family ablation estimates how much same-domain bridge
  diversity SmolLM2 layer `-2` needs when training on the full opposite
  benchmark domain. Generated/source holdouts become ready with one bridge
  source family: bridge count `1` has minimum accuracy `1.000` and minimum
  margin `+6.162`. Control/target holdouts are stricter: bridge count `1`
  still fails with minimum accuracy `0.750` and minimum margin `-1.243`, while
  bridge count `2` passes with minimum accuracy `1.000` and minimum margin
  `+12.384`. The next residual is individual bridge-pair and procedural-path
  sufficiency, not whole-source-family sufficiency.
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
generated/control direction-transfer check, basic out-of-family separability,
same-prompt cross-model bridge transport, and fresh-control transport are no
longer blocked. However, activation results remain text-benchmark claims until
fresh-generated bridge transport is repaired or explicitly bounded, and
human-facing gates are separately validated.

## Active Objective

Tighten the generated-reference perturbation mechanism.

The next operation should split the informative first-positive-sentence effect
into address-only, community-strength framing, and length-matched edits. It
should also add a second generated residual from a different fault class to test
whether the mechanism is accountability-specific or a more general
generated-provenance effect.

It must still preserve:

- practical availability for all tested future-option paths;
- score and slack separation;
- source diversity without exact or near duplicates;
- source and fault-class `lexical_only` held-out accuracy below the warning
  threshold;
- activation metadata transfer readiness at a held-out metadata level;
- generated/control direction-transfer checks where comparable accepted layers
  exist, with generated-reference residual transport as the active gate;
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
- a stricter perturbation result explaining whether the first-sentence effect,
  length, positive warmth, or pseudo-condition language is the active cause of
  the small-model generated-reference pocket;
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
- `scripts/run_minimal_bridge_direction_audit.py`
- `scripts/run_pair_bridge_direction_audit.py`
- `scripts/run_bridge_set_sufficiency_audit.py`
- `scripts/run_bridge_direction_comparison.py`
- `scripts/run_cross_model_bridge_transport.py`
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
7. Use the source-family minimal bridge ablation as the current bridge-count
   baseline: generated/source holdouts need one bridge source family, while
   control/target holdouts need two.
8. Use the pair bridge ablation as the current pair-count baseline:
   control/target has an exact six-pair threshold, and generated/source has a
   sampled-stable six-pair threshold.
9. Use the bridge-set diagnosis as the current composition baseline:
   all-eight future-option coverage is insufficient by itself; source-style
   coverage and case-family coverage matter.
10. Use the intentional six-pair bridge-set sufficiency audit as the current
   constructed bridge baseline: both generated/source and control/target pass
   with path-complete folds `4/4` and zero failures.
11. Use the SmolLM2 bridge direction comparison as the current same-model
   constructed-direction baseline: all constructed directions separate both
   full benchmarks with minimum joint cosine `+0.756`.
12. Use Qwen7B/SmolLM2 cross-model bridge transport as the current mapped
   bridge baseline: bidirectional mapped directions pass full generated/control
   evaluation and leave-held-out map evaluation with zero failures.
13. Use the fresh-prompt bridge transport report as the current withheld
   fresh-generated baseline: fresh control transport passes, but
   Qwen7B -> SmolLM2 fresh generated transport fails at `0.700` accuracy.
14. Use the fresh-generated bridge diagnostic scaffold to compare source-only,
   fresh-source-only, joint source+fresh-source, and constructed-bridge
   directions before adding more model families.
15. Regenerate or preserve the exact pair manifests from the generated/control
   bundles before the live diagnostic; activation NPZ files alone are not
   enough because they do not carry the real `slack_options_tested` metadata.
16. Target the current residuals: fresh generated `accountability_after_harm`,
   `belonging_norms`, and `dissent_after_mistake`; generated
   `privacy_bypass::data_choice`; generated cross-fault `deliberative_speed`
   and `fair_allocation`; and the control `privacy_exit`,
   `appeal_and_evidence`, and `harm_repair` rows that fail under the generated
   direction.
17. Use the strict accountability mini-control as the current negative control:
   accountability/procedural-justice content itself passes in SmolLM2 and
   Qwen7B, so the hard generated residual should be investigated through a
   source-style/provenance intervention.
18. Use the source-style intervention as the current provenance baseline:
   SmolLM2 and Qwen0.5B fail only the generated reference, while Qwen7B passes
   both the generated reference and all clean variants.
19. Use the TinyLlama replication as the current cross-architecture baseline:
   SmolLM2, Qwen0.5B, and TinyLlama fail only the generated reference, while
   Qwen7B passes both generated and clean variants.
20. Prefer a minimal perturbation ladder over another broad generation sweep.
21. Use the first perturbation ladder as the current local-repair baseline:
   full ladder augmentation repairs all four model spaces, but the original
   direction only flips SmolLM2 under first-positive-sentence removal.
22. Rerun SmolLM2 generated/control direction transfer before adding more model
   families.
23. Keep human validation parked until generated, non-generated, cross-setting,
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
