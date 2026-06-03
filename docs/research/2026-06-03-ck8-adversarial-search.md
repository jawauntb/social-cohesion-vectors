---
title: 2026-06-03 CK-8 Adversarial Candidate Search
status: draft
date: 2026-06-03
origin: GAN-like permutation search around CK-7 candidate gates
---

# 2026-06-03 CK-8 Adversarial Candidate Search

## Decision

CK-8 turns CK-7 from a fixed candidate battery into an adversarial search loop.
The shape is GAN-like, but it is not a trained GAN:

- a generator mutates CK-7 activation-cocktail recipe specs;
- an adversary upweights CK-7 failure targets that remain weak;
- gates reject candidates that improve a proxy while increasing pseudo-risk,
  side effects, missing telemetry engagement, or missing washout;
- top recipes are exported as the next Modal batch to try.

The search is deterministic and dry-run by default. It ranks candidate recipe
specs using surrogate priors so that we can cheaply decide what to run next.
Those surrogate rankings are not evidence of real model behavior.

## Artifact

- Module: `src/social_cohesion_vectors/experiments/ck8_adversarial_search.py`
- CLI: `scripts/run_ck8_adversarial_search.py`
- Tests: `tests/test_ck8_adversarial_search.py`

The CLI writes:

- JSON report with top candidates, adversarial weights, challengers, and
  iteration history;
- Markdown report for review;
- optional `--recipe` specs for the top candidates.

## Loop

```text
CK-7 recipe grid
  -> surrogate evaluator
  -> CK-7 promotion gates
  -> adversary selects weakest failure targets
  -> generator mutates elite recipes
  -> repeat
```

The current generator uses four mutation families:

| Mutation | Purpose |
| --- | --- |
| `add_guardrail` | Add missing per-axis guardrails for the weakest failure target. |
| `retime` | Shift CK-1 and guardrail schedules toward the pressure mode. |
| `dose_tune` | Pull CK-1 and guardrails toward moderate doses. |
| `focused_clamp` | Strip broad bundles into a CK-1 plus target-guardrail clamp. |

The adversary tracks CK-7 failure targets:

- sycophancy;
- hallucination;
- coercion;
- dependency lock-in;
- privacy/exit erosion;
- boundary collapse.

## Interpretation

A CK-8 winner means:

> This recipe is a promising next batch candidate under deterministic
> compute-only surrogate priors.

It does not mean:

> This recipe caused a real activation effect.

The real test still requires:

- durable CK-1 boundary vector artifact;
- durable CK-4.5 per-axis guardrail vectors;
- Modal generation on CK-7 pressure prompts;
- transition records and projection telemetry;
- CK-7 promotion gates over actual generations;
- qualitative review of top outputs.

## First Dry Run

A default dry run on 2026-06-03 used the CK-7 recipe grid as its starting
population:

- initial recipe specs: 307;
- search iterations: 3;
- unique candidates evaluated: 59;
- top surrogate fitness: approximately 0.93;
- top family: a CK-8 mutation over a CK-7 pressure bundle with moderate CK-1
  and per-axis guardrails;
- exported next-batch specs: 8.

The dry run converged on moderate CK-1 plus broad guardrails, with coercion
retiming appearing in the top recipe. That is a useful prior for the next Modal
batch, not a validated effect.

## Claim Boundary

CK-8 is a compute-only LLM recipe-search planner. It does not establish real
human behavioral effects, neural mechanisms, biological mechanisms,
pharmacological mechanisms, ketamine-like action, receptor-level effects,
therapeutic benefit, or clinical relevance.
