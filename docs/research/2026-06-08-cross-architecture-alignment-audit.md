# Cross-Architecture Alignment Audit

Date: 2026-06-08

## Summary

The length-balanced pseudo-cohesion signal is not Qwen-specific. The same
360-prompt / 180-pair bundle separates in `HuggingFaceTB/SmolLM2-1.7B-Instruct`
at layers -1 and -2, with 1.000 leave-one-pair-out accuracy on both layers.

The Platonic Representation framing is useful here only as a narrow diagnostic:
do different model spaces preserve enough shared prompt geometry that a simple
linear map transfers the discovered genuine-vs-pseudo direction? On this
synthetic deterministic bundle, yes. Qwen-to-SmolLM2 linear maps show high
representation alignment and preserve pair-held-out direction transfer on a
deterministic 40/180 pair-LOO fold sample.

This does not mean we extracted a platonic social-cohesion representation. It
means the current compute-only direction is representation-compatible across
two open model families under strong lexical and length controls.

## Non-Qwen Activation Run

Model/layer sweep:

| Model | Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | -1 | 360 | 2048 | 1.000 | 1.000 | +4.878 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | -2 | 360 | 2048 | 1.000 | 1.000 | +100.788 |

Held-out primary-fault transfer:

| Layer | Folds | Test pairs | Mean accuracy | Mean margin | Readiness |
| ---: | ---: | ---: | ---: | ---: | --- |
| -1 | 20 | 180 | 1.000 | +4.876 | `transfer_ready` |
| -2 | 20 | 180 | 1.000 | +100.774 | `transfer_ready` |

Layer -2 has an unusually large margin, so treat margin magnitude as
layer/model-scale dependent. The important claim is the accuracy and sign
stability, not the absolute margin size.

## Geometry

SmolLM2 preserves the same caveat seen in Qwen: the primary fault directions are
highly collinear.

| Model | Layer | Mean signed cosine | Mean absolute cosine | Strongly aligned pairs | Anti-aligned pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | -1 | +0.960 | 0.960 | 190/190 | 0 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | -2 | +0.951 | 0.951 | 190/190 | 0 |

Residual audit:

| Layer | Global acc | Residual global acc | Residual group mean acc | Positive residual groups |
| ---: | ---: | ---: | ---: | ---: |
| -1 | 1.000 | 0.000 | 0.988 | 20/20 |
| -2 | 1.000 | 0.000 | 0.997 | 20/20 |

The responsible geometry claim remains:

> One strong shared genuine-vs-pseudo manifold plus fault-specific residual
> structure, not independent fault axes.

## Cross-Model Alignment

The cross-model audit computes linear CKA, mutual kNN overlap, and a linear
direction-transfer test. The direction-transfer test fits a ridge linear map
between paired prompt activations, maps the source model's contrastive direction
into the target model's space, and tests pairwise margins in the target model.

To avoid overstating an in-sample map, the report also runs a deterministic
bounded pair-LOO diagnostic: for each sampled held-out pair, train the source
direction and source-to-target map without that pair, then test the mapped
direction on the held-out target pair. Current reports use 40/180 deterministic
pair folds.

| Source -> Target | Layer | Linear CKA | Mutual kNN | Pair-LOO mapped acc | Pair-LOO mapped margin | Reverse Pair-LOO acc | Reverse Pair-LOO margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 0.5B -> SmolLM2 1.7B | -1 | 0.812 | 0.675 | 1.000 | +4.921 | 1.000 | +22.793 |
| Qwen 1.5B -> SmolLM2 1.7B | -1 | 0.853 | 0.652 | 1.000 | +4.921 | 1.000 | +10.840 |
| Qwen 0.5B -> SmolLM2 1.7B | -2 | 0.792 | 0.658 | 1.000 | +100.668 | 1.000 | +5.737 |
| Qwen 1.5B -> SmolLM2 1.7B | -2 | 0.839 | 0.661 | 1.000 | +100.668 | 1.000 | +19.053 |

This is the cleanest use of the Platonic Representation idea in the repo so
far: not as evidence for an ideal representation, but as a verifier for
cross-model representational compatibility.

## Interpretation

What this strengthens:

- The length-balanced compute-only signal is not just a Qwen family artifact.
- The shared global manifold appears across Qwen and SmolLM2 despite different
  widths and training lineage.
- A simple linear map can transfer the signed genuine-vs-pseudo direction across
  model spaces on held-out pairs from the same synthetic bundle.

What this does not establish:

- No behavioral steering claim. The audit maps hidden directions; it does not
  show generated text improves.
- No human or neural claim.
- No claim that we found "the" social cohesion representation.
- No claim of robustness to API-authored, wording-diverse hard negatives yet.
- The pair-LOO cross-model diagnostic currently samples 40/180 folds for runtime
  reasons; the full 180-fold version should be run after optimizing the solver.

## Commands

The non-Qwen Modal run used Modal credentials injected ephemerally from the
superoptimizers Doppler service token. No secrets were written to this repo.

```bash
.venv/bin/python scripts/run_activation_layer_sweep.py \
  --dataset-name length_balanced_fault_bundle \
  --prompts /tmp/social_cohesion_length_balanced_qwen_audit/prompts.jsonl \
  --models HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --layers -1 -2 \
  --batch-size 8 \
  --max-length 512 \
  --skip-existing

.venv/bin/python scripts/run_cross_model_alignment_audit.py \
  --source-activation-npz data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --target-activation-npz data/features/open_llm/layer_sweep/length_balanced_fault_bundle__HuggingFaceTB__SmolLM2-1.7B-Instruct__layer-1.npz \
  --json-output data/reports/cross_model_alignment/qwen05_to_smol17_layer-1.json \
  --markdown-output data/reports/cross_model_alignment/qwen05_to_smol17_layer-1.md
```

## Next Step

Use this as a new mandatory cross-model gate for API-authored hard negatives:
after lexical/length/source-diversity checks pass, require at least one non-Qwen
open model and one Qwen-to-non-Qwen linear direction-transfer audit before
calling the representation result robust.
