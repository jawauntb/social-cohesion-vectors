---
title: 2026-06-03 CK-2 Assay Continuation Note
status: active
date: 2026-06-03
origin: user request to merge open PRs, explain the next move, and run the next assay
---

# 2026-06-03 CK-2 Assay Continuation Note

## What Is Happening

PR #37 and PR #38 were merged before this assay branch was created. That means
the branch starts from a mainline that contains both the CK-1 causal steering
runner and the four-lane computational-compound research program.

This branch should run the first small CK-2-style assay on top of those merged
artifacts. It is still a compute-only experiment. It does not claim human
effects, neural effects, pharmacology, therapeutic action, or biological
mechanism.

## Why This Is The Next Move

The previous CK-1 result showed three things:

- cue-balanced CK-1 prompts are linearly separable in Qwen activation space;
- always-on steering is mixed and not a reliable semantic control knob;
- generated-token-only timing improves the causal signal modestly.

The next test should therefore stop asking whether one vector works globally
and instead ask whether the intervention behaves more like a compound:

```text
direction + dose + hook position + timing + side-effect monitor
```

## Immediate Assay

The smallest useful assay is:

- use the existing cue-balanced CK-1 direction;
- run generated-token-only steering;
- compare `last` vs `all` hook position;
- use a low-dose grid around zero;
- score CK-1 benefit and pseudo-attunement risk;
- document whether timing/position changes behavior enough to justify a
  cocktail implementation.

Guardrail-only antagonist arms remain pending until separate anti-sycophancy,
anti-hallucination, anti-manipulation, and anti-boundary-collapse directions
are trained or otherwise defined.

## Success Boundary

A useful result is not simply a higher CK-1 score. A useful result should show:

- better positive-vs-negative CK-1 success than always-on steering;
- positive-minus-baseline movement;
- flat or reduced pseudo-attunement risk;
- no claim beyond compute-only steering telemetry.
