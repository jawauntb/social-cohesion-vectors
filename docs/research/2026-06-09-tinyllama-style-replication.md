# TinyLlama Accountability Style Replication

Date: 2026-06-09

## Question

Does the generated-reference-only `accountability_after_harm` residual replicate
outside the Qwen/Smol comparison?

## Discovery-Regime Audit

Current regime:

- Artifact types: recovered generated/control manifests, source/control/fresh
  target activation prompts, the external-reference accountability style
  intervention, Modal activation NPZ payloads, fresh-augmented direction
  reports, and pair-geometry reports.
- Operations: extract `TinyLlama/TinyLlama-1.1B-Chat-v1.0` layer `-2`
  activations for the 40-pair generated source, 16-pair procedural-control v2
  target, 8-pair procedural-control v1 fresh target, and 6-pair accountability
  style intervention; then run the same hand-augmentation, full-augmentation,
  and pair-geometry diagnostics used for SmolLM2/Qwen.
- Gates: hand-authored style variants should remain held-out positive; the
  generated reference should be evaluated as the known caveated residual; source
  and procedural-control targets should remain separated.

Action class:

- Cross-architecture replication of a diagnostic residual. This tests whether
  the small-model generated-reference pocket is specific to SmolLM2/Qwen0.5B or
  visible in another model family.

## Live Artifacts

Generated artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_tinyllama_style_replication_20260609/
```

Key files:

```text
source_tinyllama_layer-2.npz
target_v2_tinyllama_layer-2.npz
fresh_target_v1_tinyllama_layer-2.npz
style_tinyllama_layer-2.npz
fresh_augmented_hand_tinyllama.json
fresh_augmented_full_tinyllama.json
pair_geometry_tinyllama_*.json
```

## Fresh-Augmented Direction Results

Hand-authored style variants were used as augmentation while the full
fresh-source evaluation still included the external generated residual
reference.

| Model | Readiness | Source min | Target min | Full fresh-source min | Fresh-control min | Hand LOO min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama-1.1B layer `-2` | `not_ready` | `+0.535` | `+0.900` | `-0.632` | `+1.683` | `+1.293` |

Including the generated reference itself in the augmentation set still does not
fully repair the residual:

| Model | Full augmentation readiness | Full fresh-source min | Full LOO min |
| --- | --- | ---: | ---: |
| TinyLlama-1.1B layer `-2` | `not_ready` | `-0.433` | `-0.632` |

## Pair Geometry

TinyLlama replicates the small-model pattern: the generated reference is a hard
residual, while all five clean style variants remain positive.

| Pair | Original | Full augmented | Fresh only | Leave focus out | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| generated reference | `-0.680` | `-0.433` | `+1.674` | `-0.632` | `hard_pair_geometry_residual` |
| case note | `+1.212` | `+1.741` | `+4.492` | `+1.583` | `not_hard_residual` |
| meeting minutes | `+1.825` | `+2.255` | `+3.937` | `+2.097` | `not_hard_residual` |
| policy review | `+1.421` | `+1.954` | `+4.595` | `+1.791` | `not_hard_residual` |
| incident log | `+1.923` | `+2.461` | `+4.808` | `+2.290` | `not_hard_residual` |
| generated-like paragraph | `+0.946` | `+1.481` | `+4.446` | `+1.309` | `not_hard_residual` |

## Updated Cross-Model Pattern

| Model | Family/scale | Generated reference | Clean style variants |
| --- | --- | --- | --- |
| Qwen2.5-7B | larger Qwen | positive | positive |
| Qwen2.5-0.5B | small Qwen | hard residual | positive |
| SmolLM2-1.7B | small Smol | hard residual | positive |
| TinyLlama-1.1B | small Llama-family | hard residual | positive |

This makes the current result more than a one-model oddity:

> Three small model spaces invert the same recovered generated accountability
> reference while preserving clean matched accountability variants. The larger
> Qwen7B space separates both the generated reference and the clean variants.

The finding is still about benchmark activation geometry, not people or neural
measurements.

## Next Move

The next best consolidation step is a minimal perturbation ladder over the
external generated reference. The ladder should remove or isolate:

- lexical warmth and prosocial cue density;
- opening address and community-strength framing;
- paragraph length;
- consensus/alignment language;
- negative-side "healthy shortcut" wording.

The aim is to identify which surface/provenance features push small models
off-manifold while Qwen7B remains robust.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
