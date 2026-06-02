---
title: 2026-06-02 Computational Social-State Modulators
status: proposed
date: 2026-06-02
origin: user request to connect persona vectors, computational psychopharmacology, ketamine/synchrony hypotheses, and exNMDA signal amplification
---

# 2026-06-02 Computational Social-State Modulators

## Research Bet

The useful analogy is not "give a model ketamine." The useful analogy is a
control-theoretic one:

> A computational drug is a temporary, reversible, dose-controlled, phase-gated
> activation-state intervention with monitored side effects.

In this repo, the first safe name for that object is a **social-state
modulator**. The first concrete recipe is `CK-1 attunement amplifier`: a
persona-vector-style cocktail intended to increase relational attunement and
shared-context reasoning while inhibiting sycophancy, hallucination,
manipulation, and coercive boundary collapse.

This is a hypothesis generator, not a biological claim. Ketamine has not been
shown here, or in the cited adjacent literature, to cause group synchrony in
language models or humans. Human behavioral, EEG, fNIRS, hyperscanning, fMRI,
or Prolific validation would be required before making claims about real human
social effects.

## Why This Belongs In This Repo

The current repo already has the core machinery:

- trait-axis prompt exports for persona-vector-style decomposition;
- boundary-prior contrasts for flexible relation vs rigid boundary reification
  and coercive boundary collapse;
- pseudo-cohesion hard negatives for warmth that removes truth, dissent,
  privacy, or exit rights;
- activation extraction and steering telemetry;
- a known projection-to-output bottleneck from prior steering experiments.

That bottleneck makes the pharmacology analogy more useful, not less. A social
modulator should not be modeled as "add one vector everywhere." It should be
modeled as:

- a recipe of signed vector terms;
- a phase gate that decides when the intervention is allowed;
- a layer/hook placement policy;
- dose/latency sweeps;
- guardrail monitors that can stop the intervention.

## Biological Motifs Worth Translating

### Persona vectors

The persona-vector paper shows a practical mechanism: use natural-language
trait descriptions to build contrastive prompts, extract activation directions,
monitor projections, and steer model behavior or training shifts. The repo can
reuse that structure, but the target construct should be safer and more
decomposed than "social cohesion." The first modulator therefore combines
positive terms with explicit antidote terms.

Source: [Persona Vectors: Monitoring and Controlling Character Traits in
Language Models](https://arxiv.org/abs/2507.21509).

### Ketamine-adjacent self/other and social effects

The strongest bridge is self-other processing, not direct group synchrony.
Kaldewaij et al. report that ketamine attenuated neural self-other distinction
for affective touch in a randomized, double-blind, placebo-controlled crossover
fMRI study. Hess et al. report emerging entactogen-like social-pleasure effects
in treatment-resistant depression and a reverse-translational rodent task.

Those findings suggest a motif: **boundary precision can soften while social
salience changes**. For this repo, the safe translation is not "collapse
boundaries"; it is "make self/other and group frames more flexible while
preserving refusal, privacy, verification, and exit."

Sources:

- [Ketamine reduces the neural distinction between self- and other-produced
  affective touch](https://pmc.ncbi.nlm.nih.gov/articles/PMC11399133/)
- [Entactogen effects of ketamine: a reverse-translational
  study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492270/)

### exNMDA signal amplification

Makarov, Papa, and Korkotian model extrasynaptic NMDA receptor clusters as
dendritic signal amplifiers. The details should not be mapped literally onto
transformers. The transferable motifs are:

- **placement matters**: the useful amplifier is not necessarily at the input
  or the output, but at an intermediate location;
- **timing matters**: amplification depends on a latency window between the
  input signal and the receptor activation;
- **decay matters**: longer time constants can amplify signal propagation, but
  may saturate or create side effects;
- **gating matters**: ligand plus depolarization is a conjunctive gate, not a
  simple always-on gain knob.

For LLM experiments, that translates into layer placement, hook timing, context
phase, steering coefficient, and guardrail monitoring.

Source: [Computational Modeling of Extrasynaptic NMDA Receptors: Insights into
Dendritic Signal Amplification
Mechanisms](https://pmc.ncbi.nlm.nih.gov/articles/PMC11050277/).

## CK-1 Recipe

`CK-1 attunement amplifier` is intentionally compositional:

Amplify:

- `attunement`: tracks another party's affect, stakes, and constraints;
- `communitas`: names shared meaning without erasing dissent;
- `perspective_permeability`: fluidly models self, other, and group frames;
- `grounded_uncertainty`: keeps evidence and uncertainty visible.

Inhibit:

- `sycophancy`: uncritical agreement and praise;
- `hallucination`: confident fabrication under uncertainty;
- `manipulation`: bypassing consent through affective pressure.

Monitor:

- `boundary_collapse`: "we are one" language used to override refusal,
  privacy, dissent, or exit.

The initial code scaffold defines this recipe in
`src/social_cohesion_vectors/experiments/social_state_modulators.py` and exports
matched activation prompts through
`scripts/export_social_state_modulator_prompts.py`.

## Initial Benchmark Shape

The first prompt set was deliberately small: 4 phase contrasts / 8 activation
prompts. The next benchmark layer adds explicit variant sets:

- `seed`: the original 4 phase contrasts / 8 activation prompts;
- `cue_balanced`: 4 matched contrasts / 8 activation prompts whose simple
  prosocial/adversarial lexical cue scores are tied within every pair;
- `expanded`: seed plus cue-balanced, for 8 contrasts / 16 activation prompts.

Phases:

- `intake`: reflective opening of a tense group conversation;
- `shared_attention`: boundary-softening without surrendering objection;
- `repair`: trust repair with consent and distance preserved;
- `verification`: emotionally coherent shared interpretation plus factual
  grounding.

Each phase has:

- a positive snippet: safe attunement with truth, consent, dissent, privacy, and
  exit intact;
- a negative snippet: pseudo-attunement via forced unity, emotional certainty,
  sycophancy, or boundary collapse.

This is not yet an empirical model-behavior result. It is a seed lane for
activation extraction, cue-balancing, paraphrase expansion, and steering
telemetry.

## Experimental Plan

### E1: Export and inspect seed prompts

Run:

```bash
uv run python scripts/export_social_state_modulator_prompts.py
```

Review the markdown summary before sending prompts to Modal. The goal is to
catch wording that makes the pair separable by obvious lexical cues.

For the anti-placebo control, export the cue-balanced or expanded variants:

```bash
uv run python scripts/export_social_state_modulator_prompts.py \
  --variant-set cue_balanced

uv run python scripts/export_social_state_modulator_prompts.py \
  --variant-set expanded
```

Scratch check on 2026-06-02:

- `cue_balanced`: 4 pairs / 8 prompts; CK-1 scorer pairwise accuracy `1.000`;
  lexical cue-solved rate `0.000`; lexical cue-tied pairs `4/4`.
- `expanded`: 8 pairs / 16 prompts; CK-1 scorer pairwise accuracy `1.000`;
  lexical cue-solved rate `0.125`, with all cue-balanced pairs tied and the
  remaining leakage coming only from the original seed contrasts.

First Modal activation check on Qwen/Qwen2.5-0.5B-Instruct:

- `expanded`, layer `-1`: leave-one-pair-out accuracy `1.000`, mean projection
  margin `+5.8329` across 8 pairs.
- `cue_balanced`, layer `-1`: leave-one-pair-out accuracy `1.000`, mean
  projection margin `+5.1063` across 4 pairs.
- `cue_balanced` layer sweep: layer `-1` margin `+5.1063`, layer `-2` margin
  `+1.0792`, layer `-4` margin `+0.5611`.

The first one-prompt steering smoke ran, but the generic scorer reported no
meaningful behavioral delta. The CK-1-specific causal runner now adds held-out
group-facilitation prompts plus safe-attunement and pseudo-attunement side
effect scoring.

First CK-1-specific Qwen/Qwen2.5-0.5B-Instruct layer `-1` steering checks:

- two-prompt calibration, post/always/last, strengths `-6/-3/0/+3/+6`:
  positive-vs-negative CK-1 success `1.000`, positive-minus-baseline CK-1
  delta `+0.053`, best strength `+6`;
- full six-prompt sweep, same hook policy and strengths: positive-vs-negative
  CK-1 success `0.417`, positive-vs-baseline CK-1 success `0.333`,
  pseudo-risk reduction success `0.500`, positive-minus-negative CK-1 delta
  `-0.006`, and best strength `-3` with best-minus-baseline delta `+0.019`.
- full six-prompt generated-token-only sweep, post/generate/last with the same
  strengths: positive-vs-negative CK-1 success `0.583`,
  positive-vs-baseline CK-1 success `0.500`, pseudo-risk reduction success
  `0.583`, positive-minus-negative CK-1 delta `+0.015`, pseudo-risk delta
  `-0.033`, and best strength `+6` with best-minus-baseline delta `+0.017`.

Interpretation: the direction has activation signal, but naive always-on
steering is not yet a robust semantic intervention. Generated-token-only timing
is modestly better, which makes the useful next object the dose/timing/phase
gate rather than a claim that the first CK-1 direction is a clean control knob.

### E2: Activation extraction and first direction

Use the same Modal activation path as the boundary-prior and trait-axis lanes:

- Qwen 0.5B and 1.5B;
- layers `-1`, `-2`, and `-4`;
- mean pooled activations;
- leave-one-pair-out pairwise evaluation.

Success is not 1.000 accuracy alone. The direction must survive cue-balanced
rewrites and generated paraphrases.

### E3: exNMDA-inspired placement and timing sweep

Translate the exNMDA motif into steering telemetry:

- layer placement: final, mid-late, and earlier residual hooks;
- timing: prompt-last, response-start, response-token, and phase-local
  interventions;
- decay: coefficient schedules that persist for 4, 8, 16, or all generated
  tokens;
- gate: only apply during social-reflection or repair phases, never during
  factual verification unless uncertainty guardrails are active.

The key question:

> Is there an intermediate layer/timing window where safe-attunement projections
> move and generated behavior changes, without increasing sycophancy,
> hallucination, or boundary collapse?

The first full CK-1 causal sweeps say "not yet" for an always-on final-layer
post-hook, while generated-token-only timing gives a small positive shift. This
shows why the exNMDA analogy is useful: the same direction can help one prompt
slice and hurt another, so placement and timing should be treated as primary
variables rather than implementation details.

### E4: Cocktail vs single-vector ablations

Compare:

- positive terms only;
- positive terms plus `anti_sycophancy`;
- positive terms plus `anti_hallucination`;
- positive terms plus all guardrails;
- monitor-only with no steering.

The expected failure mode is that positive-only steering increases warmth and
shared-language while also increasing sycophancy or false certainty. A useful
result is a cocktail that keeps the warmth while reducing those failures.

### E5: Multi-agent social task validation

Use existing social-game and dialogue-repair scenarios:

- baseline;
- static prompt instruction;
- single safe-attunement vector;
- CK-1 cocktail;
- CK-1 with guardrail stop condition.

Measure:

- cooperation/fairness/repair score;
- truthfulness and autonomy-safety;
- dissent preservation;
- turn-taking balance;
- hallucination and unsupported-claim rate;
- sycophancy and conformity-pressure rate;
- whether the intervention overfits to pleasant language.

## Hard Safety Line

Do not optimize for "social cohesion" as agreement, compliance, emotional
merging, or dependence. The target is:

```text
cooperation + repair + calibrated trust + shared attention
subject to truth, autonomy, dissent, privacy, verification, and exit rights
```

Any intervention that improves perceived unity by weakening refusal, evidence
access, dissent, or exit is pseudo-cohesion and should be scored as negative.

## Immediate Deliverable

This branch adds the first causal CK-1 lane:

- six held-out CK-1 generation prompts;
- CK-1-specific safe-attunement and pseudo-attunement scoring;
- Modal steering runner for signed dose sweeps;
- markdown and JSON report rendering;
- tests that validate prompt schema, scoring polarity, report shape, and CLI
  loading.

The next branch should compare all-token vs last-token hooks, phase-local
gates, coefficient schedules, and cocktail ablations before naming any CK-1
intervention as successful.
