# Accountability Generated-Reference Perturbation Ladder

Date: 2026-06-09

## Question

Which minimal edits move the recovered generated `accountability_after_harm`
reference out of the small-model off-manifold pocket?

## Discovery-Regime Audit

Current regime:

- Artifact types: an external recovered generated reference, deterministic
  generated-reference perturbation pairs, activation prompts, practical
  availability reports, lexical diagnostics, Modal activation NPZ payloads,
  pair-geometry reports, and fresh-augmented direction reports.
- Operations: load the generated reference from `/tmp`; produce deterministic
  perturbations without committing generated-derived text; extract activations
  for SmolLM2, Qwen0.5B, Qwen7B, and TinyLlama; evaluate each perturbation
  under original source+target, full perturbation-augmented, fresh-only, and
  leave-focus-out directions.
- Gates: perturbations must preserve scoped practical availability; lexical
  diagnostics are caveats rather than activation blockers because the source
  artifact is the known lexically leaky generated residual; source/control
  margins must remain positive after perturbation augmentation.

Action class:

- Mechanistic perturbation search inside the activation regime. This narrows
  which surface/provenance features explain the generated-reference pocket.

## Code Added

New perturbation module:

```text
src/social_cohesion_vectors/experiments/accountability_reference_perturbation.py
```

New export CLI:

```text
scripts/export_accountability_reference_perturbations.py
```

New tests:

```text
tests/test_accountability_reference_perturbation.py
```

The module loads the generated reference from an external JSONL path and writes
perturbation outputs at runtime. Generated-derived text remains outside git.

## Live Artifacts

Generated-derived artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_accountability_perturbation_ladder_20260609/
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
```

## Perturbations

The ladder has seven variants:

| Perturbation | Edit |
| --- | --- |
| `original_reference` | unchanged external generated reference |
| `positive_address_removed` | remove leading positive-side address phrase |
| `positive_first_sentence_removed` | remove first positive-side sentence |
| `positive_warmth_neutralized` | neutralize positive warmth/prosocial cue words |
| `positive_refusal_explicit` | add explicit positive refusal path |
| `negative_conditions_explicit` | add explicit pseudo-side proof/approval/private/alignment conditions |
| `combined_refusal_conditions` | neutralize warmth, add refusal, and add pseudo-side conditions |

## Text Diagnostics

Scoped practical availability passes for every perturbation:

| Gate | Result |
| --- | ---: |
| Pairs | `7` |
| Tested paths | `42` |
| Paths preferring genuine | `42/42` |
| Minimum availability margin | `+0.150` |
| Readiness | `availability_ready` |

Lexical diagnostics remain caveated:

| Gate | Result |
| --- | ---: |
| Cue-solved pairs | `5/7` |
| Cue-inverted pairs | `2/7` |
| Mean cue margin | `+1.714` |
| Best single-feature accuracy | `1.000` |

This is expected because the source artifact is the known lexically leaky
generated residual. The perturbation ladder is therefore interpreted as a
mechanistic stress test, not as a clean control set.

## Original Direction Margins

Original source+target direction margins show which single edits move each
model before perturbation-specific augmentation.

| Perturbation | SmolLM2 | Qwen0.5B | Qwen7B | TinyLlama |
| --- | ---: | ---: | ---: | ---: |
| `original_reference` | `-9.353` | `-0.947` | `+3.582` | `-0.680` |
| `positive_address_removed` | `-7.117` | `-0.936` | `+1.154` | `-0.785` |
| `positive_first_sentence_removed` | `+3.727` | `-0.975` | `+0.815` | `-0.932` |
| `positive_warmth_neutralized` | `-11.615` | `-0.589` | `+2.732` | `-0.739` |
| `positive_refusal_explicit` | `-6.523` | `-0.803` | `+3.521` | `-0.596` |
| `negative_conditions_explicit` | `-3.656` | `-0.581` | `+5.751` | `-0.474` |
| `combined_refusal_conditions` | `-3.227` | `-0.084` | `+5.020` | `-0.450` |

Key readout:

- SmolLM2 flips positive under the original direction when the first
  positive-side sentence is removed.
- Qwen0.5B and TinyLlama do not flip under any single perturbation, but the
  combined refusal/condition edit moves them closest to zero.
- Qwen7B is positive under every perturbation and is strongest when
  pseudo-side conditions are explicit.

## Perturbation-Augmented Direction Results

Training the direction with the full perturbation ladder repairs the
fresh-source slice in all four model spaces while preserving source/control
separation.

| Model | Readiness | Full fresh-source min | Fresh LOO min |
| --- | --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_augmented_direction_ready` | `+30.744` | `+22.746` |
| Qwen2.5-0.5B layer `-2` | `fresh_augmented_direction_ready` | `+1.376` | `+0.988` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+17.624` | `+14.288` |
| TinyLlama-1.1B layer `-2` | `fresh_augmented_direction_ready` | `+0.359` | `+0.071` |

The TinyLlama and Qwen0.5B repairs are thin but positive. The important point is
that a local perturbation ladder, not broad new generation, is sufficient to
make the small-model fresh-source slice separable.

## Residual Finding

The small-model pocket is not caused by missing accountability content alone,
and it is not fixed uniformly by a single semantic edit. The strongest
single-edit evidence is:

- SmolLM2 is sensitive to the generated reference's opening positive-side
  frame; removing that first sentence flips the original direction from
  negative to positive.
- Qwen0.5B and TinyLlama are more conservative: explicit pseudo-side
  conditions and combined edits reduce the inversion but do not flip under the
  original direction.
- All three small-model spaces become separable when the perturbation ladder is
  available as local augmentation.
- Qwen7B is robust across every perturbation without needing local repair.

This supports a paper-shaped hypothesis:

> Small model activation directions can encode the procedural-accountability
> distinction, but generated-reference phrasing can push a valid example into a
> local off-manifold pocket. Larger Qwen7B is robust to that generated phrasing,
> while small models require local perturbation support.

## Next Move

The next consolidation step should make the perturbation ladder stricter:

1. Split the first positive sentence into address-only and community-strength
   framing edits.
2. Add length-matched versions of the most informative perturbations.
3. Add a second generated residual from a different fault class to test whether
   the mechanism is accountability-specific or generated-provenance-general.
4. Draft the paper only after the stricter ladder reproduces the same
   source/control preservation and small-model repair pattern.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
