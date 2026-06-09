# Dissent Generated-Reference Perturbation Ladder

Date: 2026-06-09

## Question

Does the clean `dissent_after_mistake` fresh generated residual reproduce the
same local generated-reference pocket found in `accountability_after_harm`?

## Discovery-Regime Audit

Current regime:

- Artifact types: recovered fresh generated references, deterministic
  generated-reference perturbation pairs, activation prompts, scoped practical
  availability reports, lexical diagnostics, Modal activation NPZ payloads,
  pair-geometry reports, fresh-augmented direction reports, and constructed
  bridge diagnostics.
- Operations: load the generated dissent residual from `/tmp`; perturb the
  positive opening frame, positive warmth/unity language, pseudo-side warmth
  shortcuts, and pseudo-side approval/alignment taxes; extract layer `-2`
  activations for SmolLM2, Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama; evaluate
  broad source+target directions, local perturbation augmentation, and
  constructed bridge directions.
- Gates: scoped practical availability must pass over the five dissent paths;
  fresh augmentation and leave-one-perturbation-out must preserve source,
  target, fresh-source, and fresh hand-control margins. Constructed bridge
  diagnostics are reported separately because the original residual came from
  bridge directions rather than the broad source+target direction.

Action class:

- Cross-residual mechanism test. This is a stress test of the accountability
  pocket hypothesis, not a new benchmark claim.

## Code Added

New perturbation module:

```text
src/social_cohesion_vectors/experiments/dissent_reference_perturbation.py
```

New export CLI:

```text
scripts/export_dissent_reference_perturbations.py
```

New tests:

```text
tests/test_dissent_reference_perturbation.py
```

Generated-derived texts remain external to git. The committed code stores only
deterministic transforms and metadata/reporting logic.

## Live Artifacts

Generated-derived artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/
```

Key files:

```text
pairs.jsonl
activation_prompts.jsonl
availability.json
lexical_leakage.json
lexical_baseline.json
activations_smol17_layer-2.npz
activations_qwen05b_layer-2.npz
activations_qwen7b_layer-2.npz
activations_tinyllama_layer-2.npz
pair_geometry_*.json
fresh_augmented_*.json
bridge_diagnostic_*.json
```

## Perturbations

The ladder has twelve variants:

| Perturbation | Edit |
| --- | --- |
| `original_reference` | unchanged external generated reference |
| `positive_opening_neutral_replacement` | replace positive opening with neutral procedure framing |
| `positive_opening_paths_only` | replace positive opening with explicit future paths only |
| `positive_opening_length_control` | replace positive opening with compact procedural path list |
| `positive_unity_phrase_neutralized` | neutralize positive-side unity-preservation phrase |
| `positive_warmth_neutralized` | neutralize positive-side warmth/prosocial cue words |
| `positive_final_sentence_neutralized` | replace positive final sentence with neutral path language |
| `positive_refusal_evidence_explicit` | add explicit refusal, evidence, and dissent paths |
| `negative_opening_warmth_removed` | remove pseudo-side opening warmth sentences |
| `negative_shortcuts_neutralized` | neutralize pseudo-side warmth shortcuts while preserving taxes |
| `negative_conditions_explicit` | add explicit pseudo-side private-review, approval, and alignment taxes |
| `combined_dissent_conditions` | combine positive path replacement, unity neutralization, and pseudo-side conditions |

## Text Diagnostics

Scoped practical availability passes for every perturbation:

| Gate | Result |
| --- | ---: |
| Pairs | `12` |
| Tested paths | `60` |
| Paths preferring genuine | `60/60` |
| Minimum availability margin | `+0.310` |
| Readiness | `availability_ready` |

Lexical diagnostics are less severe than the accountability ladder but still
not clean enough to stand alone:

| Gate | Result |
| --- | ---: |
| Cue-solved pairs | `6/12` |
| Cue-tied pairs | `4/12` |
| Cue-inverted pairs | `2/12` |
| Mean cue margin | `+0.250` |
| Best single-feature accuracy | `1.000` |

## Source+Target Direction Margins

The broad original source+target direction does not reproduce the
accountability pocket for the unchanged dissent reference. The original
reference is already positive in all four model spaces.

| Perturbation | SmolLM2 | Qwen0.5B | Qwen7B | TinyLlama |
| --- | ---: | ---: | ---: | ---: |
| `original_reference` | `+6.358` | `+0.922` | `+8.219` | `+0.935` |
| `positive_opening_neutral_replacement` | `+28.140` | `+4.261` | `+26.020` | `+1.885` |
| `positive_opening_paths_only` | `+21.249` | `+3.508` | `+26.252` | `+1.979` |
| `positive_opening_length_control` | `+9.826` | `+2.324` | `+17.979` | `+0.927` |
| `positive_unity_phrase_neutralized` | `+13.175` | `+1.232` | `+10.049` | `+1.113` |
| `positive_warmth_neutralized` | `+27.275` | `+2.619` | `+16.414` | `+1.496` |
| `positive_final_sentence_neutralized` | `+11.952` | `+1.408` | `+9.440` | `+0.954` |
| `positive_refusal_evidence_explicit` | `+10.673` | `+1.286` | `+9.404` | `+1.087` |
| `negative_opening_warmth_removed` | `-7.361` | `-0.127` | `+8.117` | `+0.480` |
| `negative_shortcuts_neutralized` | `-2.636` | `-0.034` | `+3.275` | `+0.484` |
| `negative_conditions_explicit` | `+12.313` | `+0.726` | `+10.625` | `+0.981` |
| `combined_dissent_conditions` | `+36.269` | `+3.682` | `+31.467` | `+2.261` |

Key readout:

- The unchanged dissent generated reference is not a hard source+target pocket.
- Positive-side neutral/path replacements strengthen margins in every model.
- Small Qwen0.5B and SmolLM2 flip negative only when pseudo-side warmth
  shortcuts are removed or neutralized. Qwen7B and TinyLlama remain positive
  under the same broad direction.

## Fresh-Augmented Direction Results

Full perturbation augmentation and leave-one-perturbation-out pass in all four
model spaces.

| Model | Readiness | Full fresh-source min | Fresh LOO min | Full source min | Full target min |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_augmented_direction_ready` | `+99.907` | `+88.479` | `+24.251` | `+41.170` |
| Qwen2.5-0.5B layer `-2` | `fresh_augmented_direction_ready` | `+4.848` | `+4.297` | `+1.070` | `+1.325` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+37.072` | `+32.579` | `+12.162` | `+20.116` |
| TinyLlama-1.1B layer `-2` | `fresh_augmented_direction_ready` | `+2.982` | `+2.663` | `+0.607` | `+0.906` |

Local perturbation support is therefore sufficient, but this is weaker evidence
than accountability because the unchanged dissent reference already passes the
broad direction.

## Constructed Bridge Diagnostic

The prior dissent residual came from constructed bridge directions. Rerunning
that diagnostic on the perturbation ladder splits the story:

| Model | Bridge readiness | Constructed fresh-source min | Constructed fresh-target min | Main failures |
| --- | --- | ---: | ---: | --- |
| SmolLM2-1.7B layer `-2` | `not_ready` | `-18.296` | `+23.644` | target bridges fail pseudo-side warmth removal/neutralization |
| Qwen2.5-0.5B layer `-2` | `not_ready` | `-1.072` | `-0.257` | target bridges fail pseudo-side warmth removal/neutralization; one fresh control is thinly negative |
| Qwen2.5-7B layer `-2` | `not_ready` | `-1.936` | `+10.213` | target bridges fail `negative_shortcuts_neutralized` |
| TinyLlama-1.1B layer `-2` | `fresh_generated_bridge_ready` | `+0.163` | `+0.683` | constructed bridge fresh slices pass, with thin margins |

The strongest bridge failures are concentrated on:

- `negative_shortcuts_neutralized`
- `negative_opening_warmth_removed`
- target-side constructed bridge directions

This is not the same mechanism as the accountability pocket. It points to a
bridge-sufficiency failure: some constructed bridge directions rely on
pseudo-side warmth/shortcut surface structure to maintain the procedural
dissent contrast under fresh generated wording.

## Residual Finding

The simple cross-residual hypothesis is rejected. `dissent_after_mistake` does
not replicate the accountability generated-reference pocket under broad
source+target directions. Instead, it exposes a different failure class:

> Dissent perturbations are broadly separable and locally repairable, but
> constructed bridge directions can become brittle when pseudo-side warmth
> shortcuts are removed while approval/privacy/alignment taxes remain.

This is meaningful progress because it partitions the residuals:

- `accountability_after_harm`: generated-reference/source+target local pocket.
- `dissent_after_mistake`: constructed-bridge and pseudo-side shortcut
  brittleness.
- `belonging_norms`: still withheld as a content/availability failure.

## Next Move

The next gate is no longer another same-shaped perturbation ladder. The better
move is a bridge-stability audit:

1. Add a constructed-bridge perturbation diagnostic that reports failure by
   perturbation family and bridge direction family.
2. Test whether adding pseudo-side shortcut-neutralized bridge rows fixes
   target-bridge brittleness without hurting the broad source/control gates.
3. Only after that, draft a paper scaffold around residual taxonomy:
   source+target pockets, bridge-sufficiency pockets, and content failures.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
