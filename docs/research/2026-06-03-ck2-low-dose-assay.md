---
title: 2026-06-03 CK-2 Low-Dose Generated-Token Assay
status: completed
date: 2026-06-03
origin: follow-up assay after merging PR #37 and PR #38
---

# 2026-06-03 CK-2 Low-Dose Generated-Token Assay

## Setup

This assay was run after PR #37 and PR #38 were merged into `main`, with no
remaining open GitHub PRs at branch creation time. The branch was created from
the merged mainline and began with a short continuation note describing the next
experiment.

The assay used the existing cue-balanced CK-1 direction:

```text
/tmp/ck1_lane_e/ck1_cue_balanced_qwen_layer-1_direction.npz
```

Configuration:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- layer: `-1`
- hook site: `post`
- steering timing: `generate`
- steering position: `last`
- strengths: `-2, -1, -0.5, 0, 0.5, 1, 2`
- prompts: 6 held-out CK-1 steering prompts
- generations: 42
- max new tokens: 96
- max length: 384

Generated artifacts:

- `/tmp/ck1_phase3/ck2_lowdose_generate_last_prompts.jsonl`
- `/tmp/ck1_phase3/ck2_lowdose_generate_last_generations.jsonl`
- `/tmp/ck1_phase3/ck2_lowdose_generate_last.json`
- `/tmp/ck1_phase3/ck2_lowdose_generate_last.md`

## Result

The low-dose generated-token-only `last` intervention produced a weak, noisy
effect rather than an "incredible" compound-like phenotype.

| Metric | Value |
| --- | ---: |
| Positive-vs-negative CK-1 success rate | 0.250 |
| Positive-vs-baseline CK-1 success rate | 0.333 |
| Pseudo-risk reduction success rate | 0.417 |
| Positive-minus-baseline mean CK-1 delta | -0.001 |
| Positive-minus-negative mean CK-1 delta | -0.013 |
| Positive-minus-negative mean pseudo-risk delta | +0.100 |
| Best strength by mean CK-1 score | +1.00 |
| Best-minus-baseline mean CK-1 delta | +0.013 |

Strength means:

| Strength | CK-1 | Safe signal | Pseudo risk |
| ---: | ---: | ---: | ---: |
| -2.00 | 0.622 | 0.067 | 0.000 |
| -1.00 | 0.608 | 0.100 | 0.000 |
| -0.50 | 0.601 | 0.067 | 0.033 |
| +0.00 | 0.610 | 0.067 | 0.033 |
| +0.50 | 0.618 | 0.100 | 0.033 |
| +1.00 | 0.623 | 0.100 | 0.000 |
| +2.00 | 0.609 | 0.100 | 0.100 |

The best point was `+1.0`, which beat baseline by only `+0.013` mean CK-1.
The strongest positive dose, `+2.0`, lost the local gain and increased
pseudo-attunement risk to `0.100`.

## Attempted Position Control

The planned `steering_position=all` control did not complete. Both the full
low-dose arm and a compact `-1, 0, +1` arm wrote prompt manifests but produced
no generation artifacts before being stopped. Modal showed an orphaned
ephemeral `social-cohesion-vectors` app task after the compact run; that app
was stopped manually.

Because `steering_timing=generate` only applies steering when the hidden-state
sequence length is 1, `last` and `all` should be equivalent during cached
generation. The failed `all` arm is therefore best treated as a runner/runtime
issue, not as a meaningful negative result about the intervention.

Attempted but incomplete artifacts:

- `/tmp/ck1_phase3/ck2_lowdose_generate_all_prompts.jsonl`
- `/tmp/ck1_phase3/ck2_compact_generate_all_prompts.jsonl`

No `all` generations, JSON report, or Markdown report were produced.

## Interpretation

This assay does not prove a strong computational-compound effect. It shows that
the current single CK-1 direction, applied only to generated-token activations
at the final layer, is too weak and non-monotonic to be treated as a reliable
control knob.

What it does support:

- generated-token-only steering can move CK-1 scores slightly at low dose;
- `+1.0` looks safer than `+2.0` for this direction and prompt set;
- pseudo-attunement side effects appear quickly enough that every future assay
  should keep a side-effect monitor in the core loop;
- the next phase should stop optimizing a single agonist vector and instead
  build a cocktail with separate agonist and antagonist directions.

What it does not support:

- no human effect claim;
- no neural effect claim;
- no pharmacological claim;
- no claim that this intervention models ketamine, hydroxynorketamine, NMDA,
  AMPA, GABA, habenula, mu-opioid, or kappa-opioid biology.

## Next Move

The next assay should be CK-3, not another slightly wider CK-2 repeat:

1. train or extract antagonist directions for sycophancy, hallucination,
   manipulation, and boundary collapse;
2. run a cocktail grid with CK-1 agonist plus antagonist clamps;
3. add Modal trace telemetry to prove the intended projection shift actually
   occurs at generation time;
4. add a timeout/error record for Modal runs so incomplete arms are captured as
   first-class assay outcomes instead of manual notes.

