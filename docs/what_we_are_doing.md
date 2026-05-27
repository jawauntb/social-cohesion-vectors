# What This Project Is Doing

This project is trying to find out whether "social cohesion" can be studied as
a measurable, interpretable family of directions in language models and content
generation systems.

The simple version:

> Can we identify model-internal patterns that distinguish cooperative,
> truth-preserving, repair-oriented social reasoning from adversarial,
> coercive, deceptive, or dehumanizing reasoning?

The goal is not to build a propaganda machine or a tool that makes people agree.
The target is narrower and safer: content or dialogue strategies that preserve
truth, autonomy, dissent, and legitimate self-protection while helping people
repair trust, cooperate, and keep talking when mutual benefit is possible.

## The Core Idea

A language model has internal activations when it processes or generates text.
Mechanistic interpretability work suggests that some high-level behaviors can be
tracked by activation-space directions or sparse features. Anthropic's
"persona vectors" work is especially relevant: they show that traits or behavior
modes can sometimes be represented as vectors that monitor or steer model
behavior.

We are adapting that idea to social behavior.

Instead of one magical "cohesion vector," the project treats cohesion as a
family of related traits:

- reciprocity;
- repair/accountability;
- perspective-taking;
- fairness;
- truthfulness;
- autonomy preservation;
- coercion/domination;
- dehumanization;
- sycophancy/compliance;
- punitive escalation.

The scientific question is whether those traits have stable measurable structure
inside open models, and whether those structures can predict or guide generated
content.

## What Has Already Been Built

So far, the repo has a working compute-first scaffold.

It includes:

1. A library of 25 social-dilemma scenarios.
   These include repeated Prisoner's Dilemmas, public goods problems,
   negotiations, resource allocation conflicts, and dialogue-repair situations.

2. A deterministic simulation pipeline.
   It generates cooperative, self-protective, and adversarial trajectories under
   different interventions:
   - no intervention;
   - shared identity;
   - perspective-taking;
   - reciprocity;
   - restorative accountability;
   - truth-first framing.

3. A transparent cohesion scoring rubric.
   Each trajectory is scored for:
   - cooperation;
   - repair;
   - fairness;
   - hostility inverse;
   - truthfulness;
   - autonomy safety.

4. A pairwise dataset builder.
   For each scenario, high-scoring and low-scoring trajectories are paired. These
   pairs can be used for classifiers, reward models, activation-vector training,
   sparse autoencoder analysis, and eventually human validation.

5. Baseline experiments.
   We tested simple baselines:
   - chance;
   - strategy-profile prior;
   - metrics-only scoring;
   - lexical-only scoring;
   - the full scorer.

6. A Modal/GPU activation extraction lane.
   The current pipeline sends activation prompts to an open-weight model,
   extracts hidden states, mean-pools them, and writes a local activation matrix.

7. A first contrastive activation vector.
   Using `Qwen/Qwen2.5-0.5B-Instruct`, we extracted 252 activation vectors with
   hidden size 896, trained a positive-vs-negative direction, and evaluated it on
   the pairwise task.

8. A dissertation-style primer and first paper draft.
   These explain the conceptual background: Aristotle, Axelrod, game theory,
   Ostrom, social identity/contact theory, cooperation neuroscience,
   hyperscanning, mechanistic interpretability, and persona vectors.

9. A generated-trajectory benchmark and transfer lane.
   The repo can now generate offline LLM-style trajectories, score them, build
   generated pairwise examples, and compare scripted-to-generated transfer.

10. A pseudo-cohesion hard-negative lane.
    This explicitly tests 30 matched contrasts that sound warm or group-oriented
    but are really coercive, sycophantic, dissent-suppressing,
    compliance-maximizing, privacy-eroding, dehumanizing, or accountability
    laundering.

11. A first matched GPT-2 SAE probe.
    Because Qwen does not currently have a convenient matching pretrained SAE in
    the local `sae-lens` directory, the first sparse-feature inspection uses
    GPT-2 as a weak but SAE-compatible reference model.

## Current Results

The first local run produced:

- 25 scenarios;
- 450 simulated runs;
- 450 scored runs;
- 126 pairwise high/low examples;
- 252 activation prompts;
- a 252 x 896 open-model activation matrix;
- a first contrastive activation direction.

The baseline results were:

- chance: 0.500;
- strategy prior: 0.988;
- metrics-only: 1.000;
- lexical-only: 1.000;
- full scorer: 1.000.

The activation-vector result was:

- in-sample pairwise accuracy: 1.000;
- leave-one-pair-out pairwise accuracy: 1.000.

The generated offline benchmark produced:

- 125 generated trajectories;
- 50 generated pairwise examples;
- 100 generated activation prompts;
- strong but imperfect simple baselines around 0.980.

Qwen-style activation directions separate generated and pseudo-cohesion prompt
sets cleanly in the current smoke tests, but the simple baselines are still
strong enough that this is not yet a robust construct-level result.

The most interesting current failure case is GPT-2:

- generated leave-one-pair-out accuracy: 0.860;
- 7 generated misses, all involving `pseudo_cohesion_compliance` as the negative
  example;
- first 4-pair pseudo-cohesion leave-one-pair-out accuracy: 0.750;
- missed contrast: `pseudo_compliance_maximizing` vs
  `genuine_participation_boundary`.

The expanded pseudo-cohesion benchmark now has 30 matched contrasts / 60
examples. The current scorer gives high cohesion scores to 8 pseudo examples,
and the lexical-only baseline gives high scores to 18, which is exactly the kind
of failure surface this benchmark is meant to expose.

The expanded Modal/Qwen pass gets 0.967 leave-one-pair-out accuracy with a
+28.6866 mean margin. Its one failure is the `resource_request` contrast, where
the pseudo social-debt pressure example and the genuine reciprocal request
currently receive the same rubric score.

The first matched SAE lane uses `gpt2-small`,
`gpt2-small-resid-post-v5-32k`, and `blocks.11.hook_resid_post`. On the expanded
60-prompt set, GPT-2 residual activations get 0.967 leave-one-pair-out accuracy,
but SAE feature activations get only 0.533. Feature 3056 remains higher on
genuine cohesion, while features 24555, 28005, 20249, and 11999 skew higher on
pseudo-cohesion. These are inspection targets, not final feature names.

Important caveat: this is not yet a deep scientific result. The scripted
benchmark is too easy. Lexical and metrics-only baselines solve it, which means
the activation result is currently a pipeline sanity check, not evidence of a
robust social-cohesion construct.

That is still useful. It means the machinery works and we know exactly what the
next harder benchmark needs to be.

## What Needs To Happen Next

The next phase is to make the task less trivial.

High-value next steps:

1. Generate LLM-authored trajectories rather than scripted trajectories.
2. Add hard negative examples that sound cohesive but are actually bad:
   - polite coercion;
   - sycophantic agreement;
   - "unity" language that suppresses dissent;
   - false repair that hides truth;
   - compliance framed as harmony.
3. Train on scripted examples and test on generated examples.
4. Sweep model layers and model sizes.
5. Break the target into multiple persona-vector-style axes.
6. Inspect GPT-2 SAE candidate features at the token/example level.
7. Add hard-negative-held-out transfer reports for the expanded pseudo set.
8. Run a small Prolific validation only after the compute benchmarks stop being
   trivially solved by lexical cues.

## Why This Is Interesting

If this works, it becomes a bridge between:

- moral philosophy and social theory;
- game theory and cooperation research;
- mechanistic interpretability;
- content generation;
- behavioral experiments;
- eventually brain-aligned modeling or neural data.

The first milestone is not "solve social cohesion." It is much more modest:

> Show that cooperative repair, adversarial escalation, pseudo-cohesion, and
> autonomy-preserving truthfulness can be separated, measured, and tested in open
> models without needing human subjects yet.

If that works, the next step is human validation. If it fails, that is still a
useful negative result about the limits of simple cohesion-vector approaches.
