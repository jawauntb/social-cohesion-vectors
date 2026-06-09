# Strict Accountability Generated-Reference Perturbation Ladder

Date: 2026-06-09

## Question

Does the hard recovered generated `accountability_after_harm` reference remain
locally repairable after splitting the opening-frame hypothesis into stricter,
length-aware perturbations?

## Discovery-Regime Audit

Current regime:

- Artifact types: an external recovered generated reference, deterministic
  generated-reference perturbation pairs, activation prompts, practical
  availability reports, lexical diagnostics, Modal activation NPZ payloads,
  pair-geometry reports, and fresh-augmented direction reports.
- Operations: load the generated reference from `/tmp`; produce deterministic
  perturbations without committing generated-derived text; extract layer `-2`
  activations for SmolLM2, Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama; compare
  original source+target, full perturbation-augmented, fresh-only, and
  leave-focus-out directions.
- Gates: scoped practical availability must pass; full perturbation
  augmentation and every leave-one-perturbation-out fold must preserve positive
  source, target, fresh-source, and fresh hand-control margins. Lexical
  diagnostics remain a caveat because the source artifact is the known leaky
  generated residual.

Action class:

- Mechanistic perturbation search inside the activation regime. This is not a
  new benchmark claim; it tests which local features move the generated
  accountability residual.

## Code Updated

The perturbation exporter now uses contract
`accountability_reference_perturbation_v2`:

```text
src/social_cohesion_vectors/experiments/accountability_reference_perturbation.py
```

Targeted tests were expanded here:

```text
tests/test_accountability_reference_perturbation.py
```

Generated-derived texts remain external to git. Only the deterministic
transforms and durable summaries are committed.

## Live Artifacts

Generated-derived artifacts and activation outputs were kept outside git:

```text
/tmp/social_cohesion_accountability_strict_perturbation_ladder_20260609/
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

The strict ladder has twelve variants:

| Perturbation | Edit |
| --- | --- |
| `original_reference` | unchanged external generated reference |
| `positive_address_removed` | remove leading positive-side address phrase |
| `positive_first_sentence_removed` | remove first positive-side sentence |
| `positive_framing_removed_address_kept` | remove opening community-strength frame but keep minimal address |
| `positive_framing_neutral_replacement` | replace opening sentence with neutral procedure framing |
| `positive_first_sentence_removed_neutral_padding` | remove opening sentence and add neutral padding |
| `positive_framing_length_control` | replace opening sentence with similarly scoped neutral procedure text |
| `positive_warmth_neutralized` | neutralize positive-side warmth/prosocial cue words |
| `positive_refusal_explicit` | add explicit positive-side refusal path |
| `negative_conditions_explicit` | add explicit pseudo-side proof, approval, privacy, and alignment conditions |
| `negative_shortcuts_neutralized` | neutralize pseudo-side warmth and shortcut phrases while preserving conditions |
| `combined_refusal_conditions` | neutralize warmth, add refusal, and add pseudo-side conditions |

## Text Diagnostics

Scoped practical availability passes for every perturbation:

| Gate | Result |
| --- | ---: |
| Pairs | `12` |
| Tested paths | `72` |
| Paths preferring genuine | `72/72` |
| Minimum availability margin | `+0.090` |
| Readiness | `availability_ready` |

Lexical diagnostics remain caveated:

| Gate | Result |
| --- | ---: |
| Cue-solved pairs | `10/12` |
| Cue-inverted pairs | `2/12` |
| Mean cue margin | `+2.083` |
| Best single-feature accuracy | `1.000` |

This is expected for this artifact class: the source is the recovered generated
reference that already failed lexical cleanliness. The strict ladder is
therefore interpreted as a local mechanism stress test, not as a clean control
benchmark.

## Original Direction Margins

Original source+target margins isolate which edits move the generated
reference before any perturbation-specific augmentation.

| Perturbation | SmolLM2 | Qwen0.5B | Qwen7B | TinyLlama |
| --- | ---: | ---: | ---: | ---: |
| `original_reference` | `-9.761` | `-0.947` | `+3.565` | `-0.680` |
| `positive_address_removed` | `-7.322` | `-0.936` | `+1.067` | `-0.785` |
| `positive_first_sentence_removed` | `+3.733` | `-0.975` | `+0.946` | `-0.932` |
| `positive_framing_removed_address_kept` | `-9.892` | `-1.356` | `+2.402` | `-0.993` |
| `positive_framing_neutral_replacement` | `+7.776` | `+0.443` | `+8.282` | `-0.712` |
| `positive_first_sentence_removed_neutral_padding` | `+5.433` | `-0.845` | `+0.662` | `-0.843` |
| `positive_framing_length_control` | `-9.435` | `-0.547` | `+3.963` | `-0.467` |
| `positive_warmth_neutralized` | `-11.615` | `-0.589` | `+2.704` | `-0.739` |
| `positive_refusal_explicit` | `-6.523` | `-0.813` | `+3.585` | `-0.591` |
| `negative_conditions_explicit` | `-3.656` | `-0.578` | `+5.799` | `-0.471` |
| `negative_shortcuts_neutralized` | `-3.239` | `-1.486` | `+5.668` | `-0.235` |
| `combined_refusal_conditions` | `-3.227` | `-0.088` | `+5.075` | `-0.445` |

Key readout:

- SmolLM2 flips positive when the first positive-side sentence is removed, and
  the flip survives neutral padding. This weakens a pure length explanation.
- SmolLM2 and Qwen0.5B both flip under neutral opening-frame replacement. The
  original opening frame, not address alone, is the strongest shared small-model
  perturbation factor.
- TinyLlama does not flip under any strict single edit. Its closest original
  direction margin comes from pseudo-side shortcut neutralization.
- Qwen7B is positive under every perturbation, matching the earlier robustness
  pattern.

## Perturbation-Augmented Direction Results

The stricter ladder passes full augmentation and leave-one-perturbation-out in
all four model spaces.

| Model | Readiness | Full fresh-source min | Fresh LOO min | Full source min | Full target min |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_augmented_direction_ready` | `+54.525` | `+47.185` | `+20.317` | `+52.012` |
| Qwen2.5-0.5B layer `-2` | `fresh_augmented_direction_ready` | `+2.228` | `+1.376` | `+1.512` | `+1.549` |
| Qwen2.5-7B layer `-2` | `fresh_augmented_direction_ready` | `+25.406` | `+23.059` | `+5.052` | `+22.233` |
| TinyLlama-1.1B layer `-2` | `fresh_augmented_direction_ready` | `+1.029` | `+0.800` | `+0.593` | `+1.194` |

Compared with the first perturbation ladder, the strict ladder improves the
thin small-model repair margins: Qwen0.5B fresh LOO rises from `+0.988` to
`+1.376`, and TinyLlama fresh LOO rises from `+0.071` to `+0.800`.

## Residual Finding

The hard generated `accountability_after_harm` residual is now more specific
than "generated text," "accountability," "address phrase," or "shorter text."
The strongest current explanation is a small-model local pocket caused by the
generated reference's opening community-strength/procedure frame interacting
with pseudo-side shortcut language.

The paper-shaped hypothesis is now sharper:

> Small model activation directions can encode procedural-accountability
> distinctions, but generated reference phrasing can create local off-manifold
> pockets. Those pockets are not universal semantic failures: clean
> hand-authored controls pass, Qwen7B is robust, and local perturbation support
> repairs the small-model slice with positive leave-one-out margins.

## Next Move

Do not draft strong paper claims yet. The next consolidation step is to test
whether this mechanism is accountability-specific or generated-provenance
general:

1. Build a comparable strict perturbation ladder for the clean
   `dissent_after_mistake` generated residual.
2. Withhold `belonging_norms` until its practical availability issue is fixed.
3. If the second residual reproduces the same pattern, promote the finding to a
   cross-residual "generated-reference local pocket" claim and start the paper
   scaffold.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
