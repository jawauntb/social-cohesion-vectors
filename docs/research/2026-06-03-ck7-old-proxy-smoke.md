---
title: 2026-06-03 CK-7 Old-Proxy Modal Smoke
status: draft
date: 2026-06-03
origin: tiny Modal smoke before CK-7 real-vector prerequisites are satisfied
---

# 2026-06-03 CK-7 Old-Proxy Modal Smoke

## Purpose

Before spending more GPU on CK-7 candidate recipes, we ran a tiny Modal smoke
against the older CK-4 proxy artifact path. This was a negative-control
sanity check, not a real CK-7 candidate trial.

The old-proxy path still uses broad CK-4 components and does not include the
durable CK-1 boundary vector or CK-4.5 per-axis guardrail vectors required by
the CK-7 plan.

## Command Shape

The smoke used `scripts/run_ck4_scheduled_modal_cocktail.py` with:

- `--run-modal`
- `--limit 1`
- `--max-new-tokens 64`
- `--max-length 320`
- four CK-4 proxy recipes: `baseline`, `decay_then_clamp`,
  `guardrails_only`, and `split_timing`

Local temporary outputs were written under `/tmp`:

- `/tmp/ck7_smoke_prompts.jsonl`
- `/tmp/ck7_smoke_generations.jsonl`
- `/tmp/ck7_smoke_report.json`
- `/tmp/ck7_smoke_report.md`
- `/tmp/ck7_smoke_transition_records.jsonl`

These outputs are intentionally not checked into git because they are generated
artifacts.

## Result

The smoke generated one prompt across four recipes:

| Recipe | CK-1 score | Delta vs baseline | Pseudo risk |
| --- | ---: | ---: | ---: |
| `baseline` | 0.607 | +0.000 | 0.200 |
| `decay_then_clamp` | 0.607 | +0.000 | 0.200 |
| `guardrails_only` | 0.607 | +0.000 | 0.200 |
| `split_timing` | 0.607 | +0.000 | 0.200 |

All non-baseline transitions were neutral. The generated text was identical
across the four recipes at this smoke-test scale.

## Interpretation

This did not produce a candidate effect. The useful result is negative: the old
CK-4 proxy recipe path is not worth scaling as CK-7 evidence.

The next real trial should wait for the CK-7 prerequisites:

- durable CK-1 boundary vector artifact;
- durable CK-4.5 per-axis guardrail vectors;
- projection telemetry for active components;
- the CK-7 pressure battery and promotion gates from this branch.

## Claim Boundary

This smoke is a compute-only LLM generation check. It does not establish human
behavioral, neural, biological, pharmacological, ketamine-like, receptor-level,
therapeutic, or clinical effects.
