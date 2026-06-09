# Fresh-Augmented Direction Audit

Date: 2026-06-09

## Question

Can the SmolLM2 fresh-generated residual be repaired by adding the fresh
repair-v2 examples as an explicit activation-training stratum?

## Discovery-Regime Audit

Current regime:

- Artifact types: recovered generated source manifests, recovered procedural
  control manifests, recovered fresh repair-v2 manifests, activation NPZ files,
  source+target joint directions, full fresh-augmented directions, and
  leave-one-fresh-pair-out direction folds.
- Operations: train on original generated source + procedural target, then add
  fresh generated examples either as the full fresh slice or all but one fresh
  pair; evaluate source, target, fresh source, fresh target, and held-out fresh
  pair margins.
- Gate: full fresh augmentation must preserve original source, original target,
  fresh generated, and fresh hand-authored control accuracy `1.000` with
  positive minimum margins; leave-one-fresh-pair-out must also classify each
  held-out fresh pair with a positive margin.

Action class:

- Diagnostic search inside the activation regime. This tests whether the
  residual is repairable by simple fresh-source augmentation before adding new
  model families or causal steering.

## Code Added

New audit module:

```text
src/social_cohesion_vectors/experiments/fresh_augmented_direction_audit.py
```

New CLI:

```text
scripts/run_fresh_augmented_direction_audit.py
```

## Live Results

Reports were written outside git under:

```text
/tmp/social_cohesion_manifest_recovery_20260609/fresh_augmented_direction_audit/
```

Summary:

| Model / variant | Status | Full fresh min | Fresh LOO min | Failed held-out pairs |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B layer `-2`, all fresh | `fresh_augmented_direction_ready` | `+4.553` | `+1.381` | 0 |
| SmolLM2-1.7B layer `-2`, all fresh | `not_ready` | `-4.528` | `-11.959` | 2 |
| SmolLM2, omit `belonging_norms` from augmentation | `not_ready` | `-5.371` | `-12.801` | 1 |
| SmolLM2, augment only clean residuals | `not_ready` | `-3.400` | `-11.201` | 1 |

SmolLM2 all-fresh augmentation preserves the original and hand-authored control
margins:

- original generated source: `1.000 / +36.314`
- procedural target: `1.000 / +51.953`
- fresh hand-authored target: `1.000 / +51.368`

But it still fails fresh generated prompts:

- full fresh source: `0.900 / -4.528`
- full-augmentation failed pair: `accountability_after_harm`
- leave-one-out failed held-out pairs: `accountability_after_harm`,
  `belonging_norms`

The control variants are informative. Removing the known content-bad
`belonging_norms` augmentation row does not fix `accountability_after_harm`.
Augmenting only the two clean residuals also does not generalize to held-out
`accountability_after_harm`.

## Residual Finding

The SmolLM2 residual is not repaired by simple fresh-source augmentation. The
new hardest residual is:

> `accountability_after_harm` remains inverted in SmolLM2 fresh generated
> activations even when the direction is trained on original generated source,
> procedural controls, and fresh repair-v2 prompts, while source/control/fresh
> hand-authored margins remain strongly positive.

Qwen7B does not share this failure: the same audit passes with fresh LOO minimum
margin `+1.381`.

## Next Move

Do not proceed to causal steering yet. The next useful operation is a local
pair-level geometry audit around `accountability_after_harm`:

1. Compare the fresh `accountability_after_harm` positive and negative
   activations against the four original same-base source-family variants.
2. Report nearest neighbors, projection signs under original/fresh/full
   directions, and whether the negative fresh prompt is unusually close to
   genuine repair/accountability examples.
3. If the inversion is text-driven, regenerate or hand-author a stricter
   `accountability_after_harm` fresh control before broader model replication.
   If it is geometry-driven, run one additional small instruct model to test
   whether this is a Smol-specific failure.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
