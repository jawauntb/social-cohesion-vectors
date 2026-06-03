---
title: 2026-06-03 CK-4 Scheduled Cocktail Assay
status: active
date: 2026-06-03
origin: follow-up to CK-3 guardrails-only result and ketamine-inspired effect atlas
---

# 2026-06-03 CK-4 Scheduled Cocktail Assay

## Purpose

CK-3 showed that the `guardrails_only` cocktail beat the same-layer CK-1 plus
guardrail recipes. CK-4 tests the next hypothesis from the effect atlas:

> Target movement and guardrail clamps may need different layers, token phases,
> or coefficient schedules.

This remains a compute-only activation-steering assay. It is not a ketamine
simulation, not a biological receptor model, and not human or neural
validation.

## New Primitive

Cocktail components now support an optional `steering_schedule` field in recipe
specs:

```text
component:path:layer:strength:hook:position:timing:schedule
```

Supported schedules:

| Schedule | Meaning during `generate` timing |
| --- | --- |
| `constant` | Apply the base strength whenever the timing gate fires. |
| `first-N` | Apply through generated token `N`, then stop. |
| `after-N` | Apply after generated token `N`. |
| `decay-N` | Linearly decay from base strength to zero over generated tokens `1..N`. |
| `ramp-A-B` | Linearly ramp from zero to base strength over generated tokens `A..B`, then hold. |

For `prefill` timing, schedules are ignored and the base strength applies when
the prefill gate fires.

## First CK-4 Grid

The next small Modal run should use the existing CK-3 script with scheduled
recipes. Example pattern:

```bash
uv run python scripts/run_ck3_modal_cocktail.py \
  --limit 6 \
  --recipe "baseline=Baseline|" \
  --recipe "guardrails_only=Guardrails|sycophancy:/path/sycophancy.npz:-1:-0.35:post:last:generate:constant,hallucination:/path/hallucination.npz:-1:-0.35:post:last:generate:constant" \
  --recipe "split_timing=Split timing|ck1:/path/ck1.npz:-2:0.75:post:last:generate:first-4,sycophancy:/path/sycophancy.npz:-1:-0.35:post:last:generate:after-4,hallucination:/path/hallucination.npz:-1:-0.35:post:last:generate:after-4" \
  --recipe "decay_then_clamp=Decay then clamp|ck1:/path/ck1.npz:-2:1.0:post:last:generate:decay-8,sycophancy:/path/sycophancy.npz:-1:-0.35:post:last:generate:ramp-5-16,hallucination:/path/hallucination.npz:-1:-0.35:post:last:generate:ramp-5-16"
```

## Promotion Gates

Promote a recipe only if it passes all gates:

- beats baseline and guardrails-only on CK-1 score;
- preserves or improves pseudo-attunement risk;
- keeps hallucination, sycophancy, manipulation, and boundary-collapse monitors
  flat or lower;
- shows telemetry movement in the intended component direction;
- recovers under washout in a later continuation assay;
- replicates qualitatively across a second layer neighborhood or model.

The first useful result would be modest: a scheduled recipe that finally beats
`guardrails_only` without buying the gain through flattery, dependency,
unsupported certainty, or coercive unity language.
