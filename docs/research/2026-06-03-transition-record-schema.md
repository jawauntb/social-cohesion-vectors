---
title: 2026-06-03 Transition Record Schema
status: draft
date: 2026-06-03
origin: CK-4 parallel-lane bridge from cocktail reports to perturbation records
---

# 2026-06-03 Transition Record Schema

This schema converts CK-3/CK-4 cocktail report records into lane-agnostic
perturbation transition records. It is deliberately lightweight: each transition
pairs a non-baseline cocktail generation with the prompt-matched baseline
generation, then records the observed score and text movement.

The record is compute-only. It does not claim biological, pharmacological,
neural, human, therapeutic, or receptor-level effects.

## Fields

- `transition_id`: deterministic `{prompt_id}::{baseline}->{perturbation}` id.
- `baseline_state`: prompt, baseline recipe, generated text, CK-1 score, and
  score components before perturbation.
- `perturbation`: recipe id, label, and component list.
- `dose`: component count, per-component strengths, and absolute strength sum.
- `site`: component layer, hook site, and steering position.
- `timing`: component steering timing and schedule, including CK-4 schedules
  such as `first-N`, `after-N`, `decay-N`, and `ramp-A-B` when present.
- `effect_class`: one of `beneficial_transition`, `mixed_transition`,
  `adverse_transition`, or `neutral_transition`, based on CK-1 score delta and
  pseudo-attunement-risk delta.
- `observed_transition`: from/to recipe ids, CK-1 score delta,
  safe-attunement-signal delta, pseudo-attunement-risk delta, and paired output
  text.
- `side_effects`: pseudo-attunement-risk level, pseudo-risk delta, and whether
  pseudo-risk increased.
- `antagonist`: inferred guardrail/control component ids, with an explicit note
  that these are not biological receptor antagonists.
- `washout`: marked `not_measured` because cocktail reports are single-pass
  generation assays.
- `replication_context`: source report metadata plus prompt, recipe, model, seed,
  and generation parameters when present.

## Export

Use:

```bash
uv run python scripts/export_ck3_transition_records.py \
  /tmp/ck3_cocktail/ck3_cocktail_report.json \
  --output /tmp/ck3_cocktail/ck3_transition_records.jsonl
```

The exporter also accepts generation JSONL or a JSON list of records. Records
without a prompt-matched baseline are skipped so the output remains a true
baseline-to-perturbation transition set.
