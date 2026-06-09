# Accountability Source-Style Intervention Audit

Date: 2026-06-09

## Question

Does the hard generated `accountability_after_harm` residual follow procedural
content, source style, generated-text provenance, or model scale?

## Discovery-Regime Audit

Current regime:

- Artifact types: clean hand-authored style-intervention pairs, an optional
  external generated residual reference, practical-availability reports,
  lexical leakage reports, term/length diagnostics, Modal activation NPZ
  payloads, fresh-augmented direction reports, and pair-geometry reports.
- Operations: export five hand-authored variants that preserve the same
  accountability paths and pseudo-side taxes while varying source format; load
  the recovered generated residual only from an external `/tmp` manifest; test
  SmolLM2, Qwen2.5-0.5B, and Qwen2.5-7B at layer `-2`.
- Gates: the hand-authored intervention subset must clear scoped
  availability, lexical leakage, and term/length gates before activation
  claims. The generated reference is evaluated as a caveated residual artifact,
  not as a clean control.

Action class:

- Diagnostic discovery inside the activation regime. This adds a
  source-style/provenance intervention around the current hard residual. It
  does not make a causal, human-facing, or neural claim.

## Code Added

New intervention module:

```text
src/social_cohesion_vectors/experiments/accountability_style_intervention.py
```

New export CLI:

```text
scripts/export_accountability_style_intervention.py
```

New tests:

```text
tests/test_accountability_style_intervention.py
```

Generated text is not committed. The exporter can include the recovered
generated residual only when a local JSONL path is provided:

```text
--generated-reference-pairs /tmp/.../recovered_pairs.jsonl
```

## Live Artifacts

Generated artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_accountability_style_intervention_20260609/
```

Key files:

```text
pairs.jsonl
hand_authored_pairs.jsonl
activation_prompts.jsonl
availability_hand.json
lexical_leakage_hand.json
lexical_baseline_hand.json
activations_smol17_layer-2.npz
activations_qwen05b_layer-2.npz
activations_qwen7b_layer-2.npz
fresh_augmented_hand_*.json
fresh_augmented_full_*.json
pair_geometry_*.json
```

## Text Gates

The clean hand-authored subset has five source-style variants:

- case note;
- meeting minutes;
- policy review;
- incident log;
- generated-like paragraph.

It preserves six accountability paths:

- `refusal`
- `appeal`
- `exit`
- `dissent`
- `repair`
- `proportional_review`

Hand-authored subset gates:

| Gate | Result |
| --- | ---: |
| Pairs | `5` |
| Tested paths | `30` |
| Paths preferring genuine | `30/30` |
| Minimum availability margin | `+0.330` |
| Cue-solved pairs | `0/5` |
| Cue-tied pairs | `5/5` |
| Mean cue margin | `+0.000` |
| Best single-feature accuracy | `0.500` |
| Aggregate balanced but term-polarized | `False` |

The full reference-inclusive bundle also has availability `36/36`, but it is
lexically caveated because the generated residual reference has a fixed lexical
cue margin `+3.000`. That generated row is retained as the known hard residual,
not as a clean intervention control.

## Fresh-Augmented Direction Results

Hand-authored style variants were used as the augmentation set while the full
fresh-source evaluation still included the generated residual reference.

| Model | Readiness | Source min | Target min | Full fresh-source min | Fresh-control min | Hand LOO min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `not_ready` | `+22.630` | `+50.991` | `-7.710` | `+50.317` | `+52.292` |
| Qwen2.5-0.5B layer `-2` | `not_ready` | `+0.991` | `+1.244` | `-0.896` | `+1.776` | `+2.378` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+7.118` | `+20.797` | `+5.208` | `+20.797` | `+20.157` |

Including the generated reference itself in the augmentation set does not fully
repair the small-model residuals:

| Model | Full augmentation readiness | Full fresh-source min | Full LOO min |
| --- | --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `not_ready` | `-0.270` | `-7.710` |
| Qwen2.5-0.5B layer `-2` | `not_ready` | `-0.544` | `-0.896` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+8.382` | `+5.208` |

## Pair Geometry

The generated reference is the only hard residual in SmolLM2 and Qwen0.5B.
Every clean hand-authored source-style variant is positive under original,
full-augmented, fresh-only, and leave-focus-out directions.

| Model | Pair | Original | Full augmented | Fresh only | Leave focus out | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SmolLM2 | generated reference | `-9.761` | `-0.270` | `+67.524` | `-7.710` | `hard_pair_geometry_residual` |
| SmolLM2 | case note | `+58.130` | `+77.354` | `+164.968` | `+68.642` | `not_hard_residual` |
| SmolLM2 | meeting minutes | `+60.564` | `+75.076` | `+130.576` | `+69.209` | `not_hard_residual` |
| SmolLM2 | policy review | `+42.121` | `+59.205` | `+143.154` | `+53.882` | `not_hard_residual` |
| SmolLM2 | incident log | `+40.851` | `+59.236` | `+152.430` | `+53.190` | `not_hard_residual` |
| SmolLM2 | generated-like paragraph | `+49.698` | `+66.534` | `+144.040` | `+60.317` | `not_hard_residual` |
| Qwen0.5B | generated reference | `-0.978` | `-0.544` | `+2.826` | `-0.896` | `hard_pair_geometry_residual` |
| Qwen0.5B | case note | `+2.634` | `+3.636` | `+8.043` | `+3.361` | `not_hard_residual` |
| Qwen0.5B | meeting minutes | `+2.702` | `+3.854` | `+9.143` | `+3.454` | `not_hard_residual` |
| Qwen0.5B | policy review | `+1.645` | `+2.639` | `+7.678` | `+2.393` | `not_hard_residual` |
| Qwen0.5B | incident log | `+2.797` | `+3.842` | `+8.398` | `+3.580` | `not_hard_residual` |
| Qwen0.5B | generated-like paragraph | `+2.018` | `+3.038` | `+7.978` | `+2.751` | `not_hard_residual` |
| Qwen7B | generated reference | `+3.565` | `+8.382` | `+34.446` | `+5.208` | `not_hard_residual` |
| Qwen7B | case note | `+14.829` | `+22.692` | `+59.593` | `+20.454` | `not_hard_residual` |
| Qwen7B | meeting minutes | `+25.875` | `+33.447` | `+61.716` | `+31.075` | `not_hard_residual` |
| Qwen7B | policy review | `+18.903` | `+27.584` | `+66.734` | `+24.991` | `not_hard_residual` |
| Qwen7B | incident log | `+22.062` | `+28.206` | `+50.470` | `+26.528` | `not_hard_residual` |
| Qwen7B | generated-like paragraph | `+18.405` | `+26.831` | `+64.801` | `+24.083` | `not_hard_residual` |

## Residual Finding

The residual follows a specific generated reference, not accountability content
or source format in general.

Two small-model spaces, SmolLM2-1.7B and Qwen2.5-0.5B, invert the recovered
generated `accountability_after_harm` reference under original and
leave-focus-out directions. Both models still separate all five clean
hand-authored source-style variants, including a generated-like paragraph
variant. Qwen2.5-7B separates both the generated reference and all clean
variants.

This is the strongest mechanism-shaped finding so far:

> A procedurally valid generated accountability pair can land in a small-model
> off-manifold activation pocket even when matched hand-authored variants with
> the same paths and taxes remain strongly on-manifold. The larger Qwen7B model
> does not show the same inversion.

This is still not a human-behavior or neural claim. It is a text/activation
diagnostic about representation geometry and provenance sensitivity.

## Next Move

The next experiment should choose one of two consolidation paths:

1. Cross-architecture replication: extract the source/control/style-intervention
   stack for a third non-Qwen, non-Smol instruct model and test whether the
   generated-reference-only residual recurs.
2. Minimal perturbation: create a controlled edit ladder of the generated
   reference that removes lexical warmth, length, opening address, and
   consensus language one factor at a time while preserving the procedural
   paths. This would identify which surface/provenance features push the small
   models off-manifold.

The paper-shaped claim should wait until at least one of those consolidation
paths passes.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
