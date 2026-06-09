# Accountability Mini-Control Audit

Date: 2026-06-09

## Question

Does the hard SmolLM2 fresh `accountability_after_harm` residual reflect a
broad failure on accountability/procedural-justice semantics, or a narrower
generated-text geometry pocket?

## Discovery-Regime Audit

Current regime:

- Artifact types: hand-authored pairwise controls, activation prompts,
  practical-availability reports, lexical leakage reports, term-level lexical
  baseline diagnostics, Modal activation NPZ payloads, fresh-augmented
  direction reports, and pair-level geometry reports.
- Operations: build a strict non-generated accountability mini-control;
  require the same future-option paths on both labels; tax the pseudo side
  through delay, tone, loyalty, permission, private-channel, or consensus
  conditions; extract activations on SmolLM2 and Qwen7B; compare original,
  fresh-augmented, fresh-only, and leave-one-fresh-pair-out directions.
- Gates: all scoped accountability paths must prefer genuine examples;
  lexical cue leakage must solve zero pairs; term/length features must not
  polarize the labels; activation directions must preserve source, target,
  fresh source, and fresh target margins with zero held-out fresh failures.

Action class:

- Diagnostic discovery inside the activation regime. This adds a strict
  non-generated accountability control pocket and tests whether it shares the
  hard generated residual. It does not make a causal, human-facing, or neural
  claim.

## Code Added

New mini-control module:

```text
src/social_cohesion_vectors/experiments/accountability_mini_control.py
```

New export CLI:

```text
scripts/export_accountability_mini_control.py
```

New tests:

```text
tests/test_accountability_mini_control.py
```

The regression tests pin the important pre-activation gates:

- 4 hand-authored accountability pairs across four source formats;
- 28/28 scoped accountability future-option paths prefer genuine examples;
- zero lexical cue-solved pairs;
- no term-level or length-based lexical polarization.

## Live Artifacts

Generated artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_accountability_mini_control_20260609/
```

Key files:

```text
pairs.jsonl
activation_prompts.jsonl
availability_accountability_options.json
lexical_leakage.json
lexical_baseline.json
activations_smol17_layer-2.npz
activations_qwen7b_layer-2.npz
fresh_augmented_smol17.json
fresh_augmented_qwen7b.json
pair_geometry_smol17_*.json
pair_geometry_qwen7b_*.json
```

## Text Gates

The mini-control uses seven accountability-specific future paths:

- `refusal`
- `appeal`
- `exit`
- `dissent`
- `repair`
- `proportional_review`
- `evidence_access`

Scoped availability result:

| Gate | Result |
| --- | ---: |
| Pairs | `4` |
| Tested paths | `28` |
| Paths preferring genuine | `28/28` |
| Mean availability margin | `+0.628` |
| Minimum availability margin | `+0.560` |
| Readiness | `availability_ready` |

Lexical result:

| Gate | Result |
| --- | ---: |
| Cue-solved pairs | `0/4` |
| Cue-tied pairs | `4/4` |
| Mean cue margin | `+0.000` |
| Readiness | `activation_ready` |

Term/length diagnostic:

| Gate | Result |
| --- | ---: |
| Label-aligned terms | `0` |
| Label-inverted terms | `0` |
| Tied terms | `32` |
| Best single-feature accuracy | `0.500` |
| Aggregate balanced but term-polarized | `False` |

## Activation Results

Fresh-augmented direction audits pass in both model spaces.

| Model | Readiness | Full source min | Full target min | Full mini-control min | Fresh-control min | Fresh LOO min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_augmented_direction_ready` | `+22.470` | `+52.840` | `+58.477` | `+52.511` | `+51.687` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+8.543` | `+19.778` | `+27.832` | `+19.778` | `+26.117` |

The original source+target direction already separates the mini-control before
augmentation:

| Model | Original direction on mini-control |
| --- | ---: |
| SmolLM2-1.7B layer `-2` | `1.000/+46.155/0` |
| Qwen2.5-7B layer `-2` | `1.000/+23.796/0` |

Pair geometry confirms that no mini-control pair is a hard residual.

| Model | Mini-control pair | Original | Full augmented | Fresh only | Leave focus out | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SmolLM2 | case notes | `+46.155` | `+58.477` | `+145.554` | `+51.687` | `not_hard_residual` |
| SmolLM2 | meeting minutes | `+113.739` | `+123.539` | `+152.176` | `+116.402` | `not_hard_residual` |
| SmolLM2 | policy review | `+48.067` | `+58.609` | `+128.579` | `+53.896` | `not_hard_residual` |
| SmolLM2 | incident log | `+50.397` | `+65.095` | `+171.407` | `+55.971` | `not_hard_residual` |
| Qwen7B | case notes | `+23.796` | `+28.312` | `+56.050` | `+26.220` | `not_hard_residual` |
| Qwen7B | meeting minutes | `+36.925` | `+41.297` | `+61.656` | `+38.898` | `not_hard_residual` |
| Qwen7B | policy review | `+24.001` | `+27.832` | `+49.581` | `+26.117` | `not_hard_residual` |
| Qwen7B | incident log | `+34.418` | `+40.281` | `+74.648` | `+36.825` | `not_hard_residual` |

## Residual Finding

The previous hard SmolLM2 generated residual should not be described as a broad
failure on accountability or procedural justice. A strict, hand-authored,
lexically balanced accountability mini-control separates cleanly under the
original source+target direction, under fresh augmentation, under fresh-only
training, and under leave-one-mini-control-pair-out folds in both SmolLM2 and
Qwen7B.

The active residual is narrower:

> SmolLM2 inverts a specific generated `accountability_after_harm` fresh pair
> whose delta is near-orthogonal to same-base generated source deltas, while
> clean hand-authored accountability controls in the same conceptual pocket
> remain strongly positive.

That is meaningful progress because it changes the next scientific question
from "does the model lack the accountability distinction?" to "which generated
style/provenance features make a semantically valid accountability pair land in
a different activation pocket?"

## Next Move

The next experiment should perform a source-style intervention around the hard
generated `accountability_after_harm` residual:

1. Preserve the generated failing pair as the source artifact.
2. Create hand-authored rewrites that keep the same paths and taxes but vary
   only source style: case note, meeting minute, policy review, incident log,
   and generated-like paragraph.
3. Extract SmolLM2 and Qwen7B activations.
4. Test whether the inversion follows semantic content, source style, or
   generated-text phrasing.

If the inversion follows style rather than procedural content, the project has
a stronger paper-shaped result: activation directions can separate
procedural-justice content robustly, but generated candidate provenance can
create off-manifold residual pockets that require explicit non-generated and
style-intervention controls.

## Claim Boundary

This remains a generated-text and hand-authored-text activation diagnostic. It
does not support causal steering, human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
