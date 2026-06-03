---
title: 2026-06-03 CK-4 Scheduled Modal Runbook
status: active
date: 2026-06-03
origin: CK-4 scheduled cocktail wrapper for named direction artifacts
---

# 2026-06-03 CK-4 Scheduled Modal Runbook

## Purpose

CK-4 is a compute-only scheduled activation-cocktail assay. It builds
CK-3-compatible recipe specs from named local direction artifacts, then can
either dry-run those specs or explicitly hand them to the existing CK-3 Modal
runner.

Do not treat CK-4 recipes as ketamine simulations, receptor models, human
behavioral evidence, neural evidence, or clinical claims. The artifact names are
mechanistic hypotheses for model activation steering only.

## Wrapper

The wrapper is:

```bash
uv run python scripts/run_ck4_scheduled_modal_cocktail.py
```

It expects named direction artifacts:

```text
--direction ck1=/path/to/ck1_direction.npz
--direction sycophancy=/path/to/sycophancy_direction.npz
--direction hallucination=/path/to/hallucination_direction.npz
```

The default CK-4 grid emits four recipes:

| Recipe | Components |
| --- | --- |
| `baseline` | no steering |
| `guardrails_only` | `sycophancy` and `hallucination`, constant guardrail clamps |
| `split_timing` | `ck1` first, then guardrail clamps after the initial token window |
| `decay_then_clamp` | decaying `ck1` pulse, ramped guardrail clamps |

## Schedule Grammar

The wrapper validates the same schedule strings consumed by the CK-3 Modal
runner:

| Schedule | Meaning during `generate` timing |
| --- | --- |
| `constant` | Apply the base strength whenever the timing gate fires. |
| `first-N` | Apply through generated token `N`, then stop. |
| `after-N` | Apply after generated token `N`. |
| `decay-N` | Linearly decay from base strength to zero over generated tokens `1..N`. |
| `ramp-A-B` | Linearly ramp from zero to base strength over generated tokens `A..B`, then hold. |

For `prefill` timing, schedules are not a generated-token policy; the base
strength applies when the prefill gate fires.

## Dry Run

Use dry-run first. It does not import Modal, does not load `.npz` direction
contents, and does not generate model outputs.

```bash
uv run python scripts/run_ck4_scheduled_modal_cocktail.py \
  --dry-run \
  --direction ck1=/path/to/ck1_direction.npz \
  --direction sycophancy=/path/to/sycophancy_direction.npz \
  --direction hallucination=/path/to/hallucination_direction.npz \
  --specs-output data/reports/ck4_scheduled_cocktail_specs.json \
  --recipe-specs-output data/reports/ck4_scheduled_cocktail_recipes.txt
```

Review the printed `--recipe` lines and the JSON spec before any GPU run. The
JSON output is a recipe construction artifact only.

## Optional Modal Handoff

Only use `--run-modal` when the dry-run specs are reviewed and the intended
direction artifact paths exist. This calls the existing CK-3 Modal runner with
the generated `--recipe` arguments.

```bash
uv run python scripts/run_ck4_scheduled_modal_cocktail.py \
  --run-modal \
  --limit 6 \
  --direction ck1=/path/to/ck1_direction.npz \
  --direction sycophancy=/path/to/sycophancy_direction.npz \
  --direction hallucination=/path/to/hallucination_direction.npz \
  --json-output data/reports/ck4_scheduled_cocktail_report.json \
  --markdown-output data/reports/ck4_scheduled_cocktail_report.md
```

Do not run Modal from this runbook unless compute budget, credentials, prompt
limit, and output paths are intentional.

## Promotion Gates

Promote a scheduled recipe only if the resulting compute report shows:

- improvement over both `baseline` and `guardrails_only`;
- flat or lower pseudo-attunement risk;
- no increase in hallucination, sycophancy, manipulation, or boundary-collapse
  monitor signals;
- telemetry movement in the intended component direction;
- qualitative replication across a neighboring layer or model before any
  stronger claim.

These gates remain computational. Human, in-person, Prolific, fMRI, EEG, fNIRS,
or hyperscanning validation would be required before making claims about real
human effects.
