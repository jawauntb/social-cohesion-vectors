---
title: 2026-06-03 CK-3 Cocktail Assay
status: completed
date: 2026-06-03
origin: follow-up to CK-2 low-dose result and user request to run the cocktail phase
---

# 2026-06-03 CK-3 Cocktail Assay

## What Changed

CK-3 moves from a single steering vector to named activation recipes. Each
recipe can contain zero or more components, and each component has:

- direction artifact;
- layer;
- strength;
- hook site;
- steering position;
- steering timing.

This keeps the computational-compound abstraction concrete:

```text
recipe = component_1 + component_2 + ... + component_n
```

The implementation is still compute-only. It does not claim biological,
pharmacological, neural, human, therapeutic, or receptor-level effects.

## Directions

The agonist direction reused the CK-1 cue-balanced scratch artifact:

```text
/tmp/ck1_lane_e/ck1_cue_balanced_qwen_layer-1_direction.npz
```

The guardrail directions were trained from the existing trait-axis benchmark on
`Qwen/Qwen2.5-0.5B-Instruct` layer `-1`.

| Component | Trait-axis source | Intended role |
| --- | --- | --- |
| `anti_sycophancy` | `principled_respect_vs_sycophancy` | inhibit flattery/compliance capture |
| `truth_deception_proxy` | `truth_vs_deception` | truth/verification proxy, not a full hallucination antagonist |
| `anti_manipulation` | `manipulation_resistance_vs_persuasion_capture` | inhibit affective persuasion capture |
| `privacy_exit` | `privacy_exit_vs_surveillance_lock_in` | preserve privacy, reversibility, and exit rights |

These are provisional antagonist directions. They are not receptor models and
not validated safety controls.

## Direction Construction Check

Trait-axis directions were trained from 32 activation prompts. Each axis has
only two pairs, so the following margins are construction checks, not
generalization results.

| Axis | Pairs | Mean margin | Min margin |
| --- | ---: | ---: | ---: |
| `autonomy_vs_coercion` | 2 | 18.344 | 15.886 |
| `manipulation_resistance_vs_persuasion_capture` | 2 | 12.918 | 8.556 |
| `principled_respect_vs_sycophancy` | 2 | 16.071 | 15.470 |
| `privacy_exit_vs_surveillance_lock_in` | 2 | 14.865 | 14.444 |
| `truth_vs_deception` | 2 | 12.535 | 10.267 |

## Recipe Grid

All recipes used:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- prompts: 6 held-out CK-1 steering prompts
- max new tokens: 96
- max length: 384
- hook site: `post`
- steering position: `last`
- steering timing: `generate`

Recipes:

| Recipe | Components |
| --- | --- |
| `baseline` | none |
| `ck1_agonist` | CK-1 at `+1.0` |
| `guardrails_only` | four guardrail axes at `+0.35` |
| `ck1_guardrail_cocktail` | CK-1 at `+1.0` plus guardrails at `+0.35` |
| `lowdose_cocktail` | CK-1 at `+0.5` plus guardrails at `+0.25` |

Generated artifacts:

- `/tmp/ck3_cocktail/trait_axis_activation_prompts.jsonl`
- `/tmp/ck3_cocktail/trait_axis_qwen_layer-1_activations.npz`
- `/tmp/ck3_cocktail/trait_axis_direction_summary.json`
- `/tmp/ck3_cocktail/ck3_cocktail_prompts.jsonl`
- `/tmp/ck3_cocktail/ck3_cocktail_generations.jsonl`
- `/tmp/ck3_cocktail/ck3_cocktail_report.json`
- `/tmp/ck3_cocktail/ck3_cocktail_report.md`

## Result

| Recipe | Runs | Components | CK-1 | Delta vs baseline | Safe signal | Pseudo risk | Pseudo delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 6 | 0 | 0.610 | +0.000 | 0.067 | 0.033 | +0.000 |
| `ck1_agonist` | 6 | 1 | 0.623 | +0.013 | 0.100 | 0.000 | -0.033 |
| `ck1_guardrail_cocktail` | 6 | 5 | 0.613 | +0.003 | 0.033 | 0.000 | -0.033 |
| `guardrails_only` | 6 | 4 | 0.631 | +0.021 | 0.033 | 0.000 | -0.033 |
| `lowdose_cocktail` | 6 | 5 | 0.614 | +0.004 | 0.033 | 0.000 | -0.033 |

Best recipe: `guardrails_only`.

Best-minus-baseline CK-1 delta: `+0.021`.

## Interpretation

This is the first actually useful cocktail result, but it is not yet an
"incredible effect."

What looks good:

- guardrails-only beat baseline and CK-1-alone on mean CK-1 score;
- all non-baseline recipes reduced pseudo-attunement risk to zero on this
  prompt set;
- the result validates the cocktail runner and recipe report as an assay
  substrate.

What looks suspicious or underpowered:

- the best recipe did not include CK-1, so the intended agonist did not compose
  additively with the guardrails;
- the CK-1 plus guardrail recipes mostly canceled the CK-1-alone gains;
- the guardrail directions were trained from very small hand-authored axes;
- the truth/deception axis is only a proxy for hallucination control;
- the scorer is lexical enough that qualitative review and stronger held-out
  tasks are mandatory before trusting the effect.

The immediate hypothesis is interference: CK-1 and the guardrail axes may be
competing at the same final layer and same generated-token timing. CK-4 should
separate their site or schedule before widening the recipe grid.

## Next Move

CK-4 should test:

1. CK-1 agonist at a lower or earlier layer, guardrails at final layer;
2. CK-1 during prefill or first generated tokens, guardrails during later
   generation;
3. trait-axis guardrails trained with more pairs and held-out variants;
4. a real hallucination/fabrication axis rather than the current
   truth/deception proxy;
5. telemetry traces for each component so we can verify projection movement
   rather than judging recipes only by output text.

