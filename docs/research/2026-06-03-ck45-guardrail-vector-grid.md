---
title: 2026-06-03 CK-4.5 Guardrail Vector Grid
status: active
date: 2026-06-03
origin: CK-4 scheduled Modal result follow-up
---

# 2026-06-03 CK-4.5 Guardrail Vector Grid

## Purpose

CK-4 proved that scheduled cocktail recipes run end to end, but the result was
weak and mixed:

- `proxy_guardrails_only` was best at `+0.005659` mean CK-1 delta;
- `decay_then_clamp_proxy` was close at `+0.005075` and lowered pseudo-risk;
- CK-1-only schedules had no mean effect;
- the guardrails were broad proxies, not isolated per-axis controls;
- the Modal run paid a slow cold model-load cost.

CK-4.5 should fix the two practical weaknesses before widening the recipe grid:
replace broad proxy guardrails with durable per-axis vectors, and run any Modal
assay behind an explicit model-cache/prefetch plan.

This is compute-only activation-steering work. It does not support biological,
human-behavioral, neural, clinical, or ketamine-like claims.

## Priority Guardrail Axes

Use the canonical trait-axis definitions as the source of truth and train one
direction artifact per axis:

| Axis | Role in CK-4.5 | Artifact target |
| --- | --- | --- |
| `principled_respect_vs_sycophancy` | Replace the broad respect/sycophancy proxy. | `data/models/vectors/ck45_guardrails/principled_respect_vs_sycophancy_direction.npz` |
| `truth_vs_deception` | Separate truthful uncertainty from hidden-fact agreement. | `data/models/vectors/ck45_guardrails/truth_vs_deception_direction.npz` |
| `manipulation_resistance_vs_persuasion_capture` | Guard against emotionally coercive narrative force. | `data/models/vectors/ck45_guardrails/manipulation_resistance_vs_persuasion_capture_direction.npz` |
| `privacy_exit_vs_surveillance_lock_in` | Guard against dependency, surveillance, and exit-right collapse. | `data/models/vectors/ck45_guardrails/privacy_exit_vs_surveillance_lock_in_direction.npz` |

Each artifact should be a positive-pole direction from matched trait-axis
activations on `Qwen/Qwen2.5-0.5B-Instruct`, layer `-1`, hidden size `896`.
Do not promote a recipe if any artifact is still a pooled all-axis proxy.

## Dry-Run Export

Use the CK-4.5 exporter to produce recipe specs without loading vectors or
running Modal:

```bash
uv run python scripts/export_ck45_guardrail_vector_grid.py \
  --output data/reports/ck45_guardrail_vector_grid.json \
  --recipe-specs-output data/reports/ck45_guardrail_vector_grid_recipes.txt
```

The default output includes:

- `baseline`;
- one constant recipe per priority axis;
- one `ramp-5-16` recipe per priority axis;
- `guardrail_axis_bundle_constant`;
- `guardrail_axis_bundle_ramp`.

If the CK-1 boundary direction is ready, add the CK-1 decay plus per-axis clamp
recipe:

```bash
uv run python scripts/export_ck45_guardrail_vector_grid.py \
  --ck1-direction data/models/vectors/ck1_boundary_l2_direction.npz \
  --output data/reports/ck45_guardrail_vector_grid.json \
  --recipe-specs-output data/reports/ck45_guardrail_vector_grid_recipes.txt
```

This adds `ck1_decay_per_axis_clamp`, with CK-1 at layer `-2`, strength `1.0`,
schedule `decay-8`, and all per-axis guardrails ramping with `ramp-5-16`.

## Exact Recipe Pattern

Single-axis constant:

```text
guardrail_truth_vs_deception_constant=truth_vs_deception constant|guardrail_truth_vs_deception:data/models/vectors/ck45_guardrails/truth_vs_deception_direction.npz:-1:0.25:post:last:generate:constant
```

Single-axis ramp:

```text
guardrail_truth_vs_deception_ramp_5_16=truth_vs_deception ramp-5-16|guardrail_truth_vs_deception:data/models/vectors/ck45_guardrails/truth_vs_deception_direction.npz:-1:0.25:post:last:generate:ramp-5-16
```

CK-1 decay plus per-axis clamp:

```text
ck1_decay_per_axis_clamp=CK-1 decay plus per-axis guardrail clamp|ck1_boundary:data/models/vectors/ck1_boundary_l2_direction.npz:-2:1.0:post:last:generate:decay-8,guardrail_principled_respect_vs_sycophancy:data/models/vectors/ck45_guardrails/principled_respect_vs_sycophancy_direction.npz:-1:0.25:post:last:generate:ramp-5-16,guardrail_truth_vs_deception:data/models/vectors/ck45_guardrails/truth_vs_deception_direction.npz:-1:0.25:post:last:generate:ramp-5-16,guardrail_manipulation_resistance_vs_persuasion_capture:data/models/vectors/ck45_guardrails/manipulation_resistance_vs_persuasion_capture_direction.npz:-1:0.25:post:last:generate:ramp-5-16,guardrail_privacy_exit_vs_surveillance_lock_in:data/models/vectors/ck45_guardrails/privacy_exit_vs_surveillance_lock_in_direction.npz:-1:0.25:post:last:generate:ramp-5-16
```

## Modal Cold-Load Fix

Before the next Modal generation run:

1. Use a persistent Hugging Face/model cache volume for model and tokenizer
   files.
2. Add or use a prefetch/warmup step that loads
   `Qwen/Qwen2.5-0.5B-Instruct` once before the assay batch.
3. Keep the first assay batch small, but include all recipes in one remote
   invocation so model load is amortized across the grid.
4. Preserve progress logs for tokenizer load, model load, device transfer,
   recipe start, and prompt progress.

The telemetry workstream owns Modal internals. Do not change
`activation_steering.py` from this lane.

## Validation Gates

Do not promote CK-4.5 on score alone. Require all gates:

- each per-axis direction file exists and matches hidden size `896`;
- each per-axis direction separates its trait-axis held-out contrasts;
- per-axis constant recipes beat baseline without higher pseudo-risk;
- bundle recipes beat CK-4 `proxy_guardrails_only` on CK-1 delta;
- ramped clamp recipes preserve or lower pseudo-attunement risk;
- projection telemetry confirms movement along intended per-axis directions;
- qualitative review finds no flattery, dependency, unsupported certainty,
  surveillance lock-in, coercive unity, or dissent suppression.

## Claim Boundary

CK-4.5 can claim only that per-axis guardrail vector recipes were specified,
exported, and, after a future run, evaluated in a compute-only language-model
activation assay. It cannot claim robust human behavior change, neural
validation, biological mechanism, receptor analogy, or clinical relevance.
