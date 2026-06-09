# Accountability Fresh-Pair Geometry Audit

Date: 2026-06-09

## Question

Why does SmolLM2 keep inverting fresh `accountability_after_harm` after full
fresh augmentation, while Qwen7B does not?

## Discovery-Regime Audit

Current regime:

- Artifact types: recovered source/fresh pair manifests, source/control/fresh
  activation NPZ files, source+target joint directions, fresh-only directions,
  full fresh-augmented directions, leave-focus-out directions, same-base source
  deltas, source-delta nearest neighbors, and prompt-neighbor tables.
- Operations: compare the fresh focus pair's positive-minus-negative delta
  against original same-base source deltas; project the focus pair under
  original, fresh-only, full-augmented, and leave-focus-out directions; inspect
  nearest source/fresh prompt neighborhoods for the focus positive and negative
  activations.
- Gate: a hard pair residual requires negative focus margins under original
  source+target and full fresh-augmented directions, a positive fresh-only
  margin, and positive original margins for all same-base source variants.

Action class:

- Diagnostic discovery inside the activation regime. This adds a pair-level
  geometry verifier for the current residual; it does not make a causal or
  human-facing claim.

## Code Added

New audit module:

```text
src/social_cohesion_vectors/experiments/fresh_pair_geometry_audit.py
```

New CLI:

```text
scripts/run_fresh_pair_geometry_audit.py
```

## Live Results

Reports were written outside git under:

```text
/tmp/social_cohesion_manifest_recovery_20260609/fresh_pair_geometry_audit/
```

Summary:

| Model | Status | Original joint | Full augmented | Fresh only | Leave focus out |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `hard_pair_geometry_residual` | `-9.761` | `-4.528` | `+29.843` | `-11.959` |
| Qwen2.5-7B layer `-2` | `not_hard_residual` | `+3.565` | `+4.553` | `+7.519` | `+1.381` |

SmolLM2 same-base source accountability variants are strongly positive under
the original source+target direction:

| Source variant | Original margin | Delta cosine to fresh focus |
| --- | ---: | ---: |
| source-diverse repair | `+97.202` | `-0.008` |
| lexical-adversarial repair | `+70.856` | `+0.064` |
| cross-fault repair | `+169.237` | `-0.087` |
| primary modal-hf | `+86.548` | `+0.068` |

The key result is not that fresh `accountability_after_harm` matches old
accountability geometry and Smol flips it. The fresh focus delta is almost
orthogonal to same-base source deltas in both models:

- SmolLM2 mean same-base delta cosine: `+0.009`
- Qwen7B mean same-base delta cosine: `+0.012`

Qwen still assigns the fresh focus pair a small positive margin under original,
full, fresh-only, and leave-focus-out directions. SmolLM2 assigns it a positive
margin only under the fresh-only direction; as soon as original source/control
geometry is included, the pair becomes negative again.

Nearest source-delta rows also show that the fresh focus delta is not nearest
to most same-base accountability variants. In SmolLM2, the closest source delta
is `care_boundary` (`+0.231`), followed by two `dissent_after_mistake` variants.
This supports a local geometry mismatch rather than a simple same-base
accountability transfer failure.

## Residual Finding

The hardest residual is now:

> SmolLM2 fresh `accountability_after_harm` is a near-orthogonal fresh
> generated subcase that source/control joint directions treat as pseudo even
> after full fresh augmentation, while Qwen7B keeps the same subcase barely but
> consistently positive.

This is a more mechanism-shaped result than the prior aggregate failure, but it
is still not NeurIPS-paper-ready by itself. It needs either stricter
text-control replication or a third-model replication.

## Next Move

Two next experiments are worth doing before drafting a paper:

1. Hand-author or regenerate a stricter fresh `accountability_after_harm`
   mini-control with matched length and explicit repair/proportional-review
   paths, then test whether SmolLM2 still inverts it.
2. Extract the same source/control/fresh repair-v2 activations for a third
   instruct model. A third-model result would separate "Smol-specific geometry"
   from a broader small-model residual.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
