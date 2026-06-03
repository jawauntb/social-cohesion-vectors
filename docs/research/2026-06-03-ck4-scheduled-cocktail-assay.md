---
title: 2026-06-03 CK-4 Scheduled Cocktail Assay
status: completed
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

The CK-4 runbook now has a wrapper that builds these scheduled recipe specs
from named direction artifacts before any Modal call:

```bash
uv run python scripts/run_ck4_scheduled_modal_cocktail.py \
  --dry-run \
  --direction ck1=/path/ck1.npz \
  --direction sycophancy=/path/sycophancy.npz \
  --direction hallucination=/path/hallucination.npz
```

See [2026-06-03 CK-4 Scheduled Modal Runbook](2026-06-03-ck4-modal-runbook.md)
for dry-run review, optional `--run-modal` handoff, and compute-only claim
boundaries.

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

## Modal Result

The first CK-4 scheduled proxy grid ran on Modal after the wrapper and CK-3
runner were instrumented for visible remote progress logs.

Result note:
[2026-06-03 CK-4 Scheduled Modal Run Results](2026-06-03-ck4-scheduled-modal-run-results.md).

The result is mixed. The best recipe was `proxy_guardrails_only`, not a
scheduled CK-1 recipe. It improved mean CK-1 score by only `+0.006` over
baseline. `decay_then_clamp_proxy` was close at `+0.005` and lowered mean
pseudo-attunement risk, but neither result is strong enough to promote.

The practical win is methodological: CK-4 now has scheduled recipe execution,
remote progress logs, report output, and transition-record export. The
scientific win is narrower: this specific scheduled CK-1 pulse did not add a
clear benefit over proxy guardrails on the six held-out prompts.
