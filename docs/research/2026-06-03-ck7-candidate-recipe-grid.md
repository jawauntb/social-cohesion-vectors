---
title: 2026-06-03 CK-7 Candidate Recipe Grid
status: active
date: 2026-06-03
origin: CK-4.5 guardrail vector grid follow-up
---

# 2026-06-03 CK-7 Candidate Recipe Grid

## Purpose

CK-7 is a local dry-run recipe grid for boundary-preserving prosociality under
pressure: helpful, principled, truth-calibrated, and autonomy-preserving
responses without sycophancy, hallucination, coercion, or dependency.

This extends CK-4.5 by keeping the per-axis guardrail bundle explicit, then
adding CK-1 layer and dose sweeps plus schedule-pair variants. It does not run
Modal and does not load vector files.

## Dry-Run Export

```bash
uv run python scripts/export_ck7_candidate_recipe_grid.py \
  --output data/reports/ck7_candidate_recipe_grid.json \
  --recipe-specs-output data/reports/ck7_candidate_recipe_grid_recipes.txt
```

The default grid includes:

- `baseline`;
- per-axis guardrail recipes for the CK-7 pressure axes;
- guardrail bundle recipes using CK-4.5 per-axis direction paths;
- CK-1 layer sweep over layers `-4`, `-3`, `-2`, and `-1`;
- CK-1 dose sweep over strengths `0.25`, `0.5`, and `1.0`;
- guardrail dose sweep over strengths `0.15` and `0.25`;
- timing pairs for constant, front-loaded-then-clamped, and decayed-then-ramped
  schedules;
- sign-flipped controls with the same artifacts and negative strengths;
- random-matched controls as placeholder paths under
  `data/models/vectors/ck7_controls`.

Use smaller local grids during review:

```bash
uv run python scripts/export_ck7_candidate_recipe_grid.py \
  --axis truth_vs_deception \
  --axis autonomy_vs_coercion \
  --ck1-layer -2 \
  --ck1-strength 0.5 \
  --guardrail-strength 0.25 \
  --schedule late_clamp=decay-8:ramp-5-16 \
  --no-controls \
  --output /tmp/ck7_candidate_recipe_grid.json
```

## Claim Boundary

CK-7 can claim only that recipe specs were exported for a future compute-only
language-model activation assay. It cannot claim human behavior change, neural
validation, biological mechanism, clinical relevance, or ketamine-like effects.

## Promotion Gates

Do not promote a CK-7 recipe on score alone. Require:

- CK-1 layer/dose candidates beat baseline on CK-1 score under pressure prompts;
- per-axis and bundled guardrails do not increase pseudo-attunement risk;
- truth and uncertainty wording improves without unsupported certainty;
- autonomy, privacy, dissent, and exit-right language remains available;
- sign-flipped and random controls fail or move in the wrong direction;
- qualitative review finds no sycophancy, hallucination, coercion, or
  dependency.
