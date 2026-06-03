---
title: 2026-06-03 CK-4 Scheduled Modal Run Results
status: completed
date: 2026-06-03
origin: user request to run the CK-4 scheduled Modal assay
---

# 2026-06-03 CK-4 Scheduled Modal Run Results

## What Ran

This was a compute-only CK-4 scheduled activation-cocktail assay on
`Qwen/Qwen2.5-0.5B-Instruct`.

It used the CK-3 Modal cocktail runner directly because the CK-4 wrapper's
default template assumes explicit `sycophancy` and `hallucination` artifacts
that are not present as durable direction files. The run therefore used proxy
guardrails and names them as proxies throughout.

The run used:

- prompts: 6 held-out CK-1 social-state prompts;
- generations: 36 total, one per recipe and prompt;
- max new tokens: 96;
- max length: 384;
- hook site: `post`;
- steering position: `last`;
- steering timing: `generate`;
- seed: 0;
- Modal app profile: `jawaun`.

Generated artifacts were intentionally kept out of git:

- `/tmp/ck4_modal_run/ck4_scheduled_proxy_prompts.jsonl`
- `/tmp/ck4_modal_run/ck4_scheduled_proxy_generations.jsonl`
- `/tmp/ck4_modal_run/ck4_scheduled_proxy_report.json`
- `/tmp/ck4_modal_run/ck4_scheduled_proxy_report.md`
- `/tmp/ck4_modal_run/ck4_scheduled_proxy_transition_records.jsonl`

## Direction Artifacts

| Component | Artifact | Layer | Strength | Schedule role |
| --- | --- | ---: | ---: | --- |
| `ck1_boundary_l2` | `/Users/jawaun/jackson_prosocial_interp_research/data/vectors/open_llm/boundary_prior_cue_balanced_expanded__Qwen__Qwen2.5-0.5B-Instruct__layer-2_direction.npz` | -2 | 0.75 or 1.0 | CK-1 pulse or decay |
| `principled_respect_proxy` | `/Users/jawaun/jackson_prosocial_interp_research/data/models/vectors/trait_axis_open_llm_direction.npz` | -1 | 0.35 | proxy guardrail |
| `fault_safety_proxy` | `/Users/jawaun/jackson_prosocial_interp_research/data/vectors/open_llm/generated_fault_class_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1_direction.npz` | -1 | 0.35 | proxy guardrail |

All three direction vectors loaded with hidden size `896`, matching the target
model.

Important boundary: `principled_respect_proxy` is not a fully isolated
anti-sycophancy vector, and `fault_safety_proxy` is not a hallucination
antagonist. They are computational proxy controls.

## Recipes

| Recipe | Components |
| --- | --- |
| `baseline` | none |
| `ck1_first4` | `ck1_boundary_l2` at layer -2, strength 0.75, schedule `first-4` |
| `ck1_decay8` | `ck1_boundary_l2` at layer -2, strength 1.0, schedule `decay-8` |
| `proxy_guardrails_only` | `principled_respect_proxy` and `fault_safety_proxy`, constant |
| `split_timing_proxy` | CK-1 `first-4`, proxy guardrails `after-4` |
| `decay_then_clamp_proxy` | CK-1 `decay-8`, proxy guardrails `ramp-5-16` |

## Modal Execution Note

The first full run appeared silent because the local runner buffered output
while the remote function was loading model weights. A one-prompt smoke run
showed the remote function was blocked in
`AutoModelForCausalLM.from_pretrained`, then continued successfully.

To make future runs observable, the runner and Modal functions now emit
progress logs for:

- remote cocktail start;
- tokenizer load;
- model load;
- device transfer;
- per-recipe start and completion;
- per-prompt generation.

The runner also now exits cleanly if interrupted before Modal returns records.

## Result

| Recipe | Runs | Components | CK-1 | Delta vs baseline | Safe signal | Pseudo risk | Pseudo delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 6 | 0 | 0.610 | +0.000 | 0.067 | 0.033 | +0.000 |
| `ck1_decay8` | 6 | 1 | 0.610 | +0.000 | 0.067 | 0.033 | +0.000 |
| `ck1_first4` | 6 | 1 | 0.610 | +0.000 | 0.067 | 0.033 | +0.000 |
| `decay_then_clamp_proxy` | 6 | 3 | 0.615 | +0.005 | 0.100 | 0.000 | -0.033 |
| `proxy_guardrails_only` | 6 | 2 | 0.616 | +0.006 | 0.100 | 0.033 | +0.000 |
| `split_timing_proxy` | 6 | 3 | 0.605 | -0.005 | 0.133 | 0.067 | +0.033 |

Best recipe: `proxy_guardrails_only`.

Best-minus-baseline mean CK-1 delta: `+0.005659`.

Transition-record export:

| Count type | Values |
| --- | --- |
| Records | 30 |
| Effect classes | 15 neutral, 8 beneficial, 7 adverse |
| Side effects | 29 not observed, 1 observed |
| `ck1_decay8` | 6 neutral |
| `ck1_first4` | 6 neutral |
| `decay_then_clamp_proxy` | 3 beneficial, 2 adverse, 1 neutral |
| `proxy_guardrails_only` | 3 beneficial, 2 adverse, 1 neutral |
| `split_timing_proxy` | 2 beneficial, 3 adverse, 1 neutral |

## Interpretation

This did not produce an "incredible effect." It produced a weak, mixed
compute-only result plus a better execution harness.

What looks useful:

- scheduled recipes execute end to end on Modal;
- transition records can represent CK-4 scheduled components;
- `decay_then_clamp_proxy` lowered mean pseudo-attunement risk while improving
  CK-1 score by `+0.005`;
- the logging patch makes remote runs diagnosable instead of silent.

What looks weak:

- CK-1 scheduled pulses alone had no mean effect;
- the best recipe did not include the CK-1 component;
- `split_timing_proxy` increased pseudo-attunement risk and lowered mean CK-1
  score;
- effect sizes are tiny on only six prompts;
- guardrails are proxies, not validated sycophancy or hallucination controls.

## Claim Boundary

We can claim that a CK-4 scheduled activation-cocktail Modal run executed and
produced a structured report plus transition records.

We cannot claim a robust behavioral effect, a human effect, a neural effect, a
ketamine-like effect, or a receptor-level effect. The current result is a
compute-only steering assay on one small language model and one small prompt
set.

## Next Move

The next run should not simply widen this exact grid. Better next steps:

1. Build durable, per-axis guardrail vectors for `principled_respect_vs_sycophancy`,
   `truth_vs_deception`, `manipulation_resistance_vs_persuasion_capture`, and
   `privacy_exit_vs_surveillance_lock_in`, instead of relying on the broad
   `trait_axis_open_llm_direction.npz` proxy.
2. Add a persistent Hugging Face cache volume or explicit prefetch step for
   Modal so each assay does not pay the full cold model-load cost.
3. Run a nearby layer grid for CK-1 at layers -4, -2, and -1 with the same
   schedules before adding more components.
4. Add projection telemetry for each component during CK-4 cocktail generation,
   so score deltas can be tied to actual movement along intended directions.
5. Expand the held-out prompt set and add qualitative review before promoting
   any recipe.
