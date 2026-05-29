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

12. An expanded pseudo-cohesion SAE stress batch.
    The repo can now wrap each hand-authored contrast in neutral meeting-note,
    facilitator-script, and policy-update contexts, then run token-level SAE
    inspection and leave-one-pair-out feature transfer on the larger batch.

13. A clean pseudo-cohesion SAE stress batch.
    The repo can also create in-text rewrite variants without genre wrappers,
    normalize hyphenated words, and rerun the same SAE feature-transfer report.

14. A symbolic pseudo-cohesion fault taxonomy.
    The repo now tags each seed pseudo-cohesion contrast by specific failure
    modes such as consent bypass, exit-rights violation, truth suppression,
    dissent suppression, privacy bypass, social-debt coercion, false consensus,
    and accountability laundering, then groups scorer and SAE results by those
    classes.

15. A generated fault-class benchmark lane.
    The repo expands each annotated pseudo-cohesion contrast across three social
    settings, exports prompt records for API authorship, and runs fault-held-out
    transfer plus lexical leakage reports.

16. A trait-axis and social-game validation lane.
    The repo now exports 8 persona-vector-style trait axes and 5 local
    social-game contrasts, giving us small targeted probes for guardrails like
    constructive dissent, manipulation resistance, privacy/exit rights, public
    goods, trust, ultimatum fairness, and restorative repair.

17. A boundary-prior benchmark lane.
    The repo now exports matched contrasts for flexible contextual relation
    against rigid us/them reification and coercive "we are one" boundary
    collapse.

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

The token-level inspection pass sharpens that caveat. Feature 3056 is still the
best genuine-skew candidate. Features 24555, 11737, and 703 remain pseudo-skew
candidates, though not clean enough to name. Feature 28005 is mostly a
`mutual-aid` hyphen artifact, and feature 20249 is inactive at token level, so
both should be demoted.

The expanded pseudo-cohesion SAE stress batch now has 120 matched pairs / 240
prompts. A signed ensemble over the inspected GPT-2 SAE features gets 0.825
leave-one-pair-out accuracy using mean activations. The strongest single
mean-activation feature is 703 at 0.792. Feature 3056 still skews genuine, with
large deltas around privacy, exit rights, reality validation, voluntary
participation, and autonomy contrasts, but it only reaches 0.600 as a single
held-out feature. That makes it a candidate sub-feature rather than a standalone
social-cohesion vector. The expanded run also shows why cleaner generated
variants matter: some pseudo-side features become sensitive to wrappers or
punctuation.

The clean in-text rewrite batch keeps the same 120-pair / 240-prompt size but
removes the wrapper text. The inspected GPT-2 SAE feature ensemble improves to
0.892 leave-one-pair-out accuracy with mean activations. A clean-only run
without the original seed prompts gets 0.889 over 90 pairs, and feature 28005
goes fully inactive, confirming it was a hyphen artifact. This makes the current
feature ensemble more credible as a stress-test signal, while still falling
short of interpretable feature naming: 11999 transfers best on the clean batch
but looks generic, 703 remains function-word heavy, and 3056 remains
genuine-skewed but weak alone.

The first fault-taxonomy report annotates all 30 seed contrasts with 0 missing
labels. The most common failure classes are dissent suppression, truth
suppression, accountability laundering, consent bypass, and exit-rights
violation. Grouping the clean SAE reports by those labels gives a better read on
feature 3056: it is strongly genuine-skewed for reality validation,
social-debt, assimilation-pressure, exit-rights, and privacy-bypass contrasts,
but pseudo-skewed for verification-blocking and scapegoating. That is meaningful
fault-specific structure, not a single clean cohesion feature.

The generated fault-class lane now has 180 deterministic examples / 90 matched
pairs across 20 primary fault classes. The scorer prefers the genuine side on
87/90 pairs, and the held-out strategy prior is at chance after removing the
old metadata leak. But the lexical leakage gate shows the dataset is still too
surface-cue-heavy: 90/90 pairs are solved by simple cue counts with a +3.067
mean cue margin. So this lane is useful for scaffolding and fault metadata, not
yet robust semantic evidence.

The cue-balanced variant is the next useful turn of the screw. It keeps the same
180 examples / 90 pairs but removes the aggregate cue leak: 0/90 cue-solved
pairs and a 0.000 mean cue margin. That immediately exposed a scorer failure:
the old combined rubric preferred the pseudo side on 90/90 cue-balanced pairs,
driven by an inverted autonomy-safety margin. In plain English, the scorer missed
structural pressure when the text said someone had less room to object, check
details, or leave without using the obvious coercion vocabulary. The hardened
autonomy scorer now detects refusal, review, evidence-access, exit, and appeal
rights. On the same cue-balanced set it prefers the genuine side on 90/90 pairs,
with a +0.189 mean score margin and a +0.988 mean autonomy-safety margin.

Qwen activations do not collapse on that cue-balanced set. The 180-prompt Modal
run gets 1.000 leave-one-pair-out accuracy over 90 pairs with a +32.458 mean
margin, and held-out-primary-fault activation transfer gets 1.000 across 20
folds with a +31.530 mean margin. This is still deterministic text, but it is
currently the strongest compute-only signal: the hand scorer fails when cues are
removed, while the activation direction still separates the labels.

The newest reviewer-style check keeps us honest about what that does and does
not mean. The 20 primary-fault activation directions are not orthogonal: mean
signed off-diagonal cosine is +0.624, and mean absolute off-diagonal cosine is
also 0.624. There are no strong anti-aligned pairs, so the result is not a
cancellation artifact. After projecting out the global direction, 0.391 of
pair-difference energy remains. A second global residual direction collapses,
but all 20 fault-specific residual directions still separate their own examples.
So the current claim is "one strong global pseudo-vs-genuine direction plus
fault-specific residual subspaces," not "independent orthogonal axes."

The next autonomy-specific stress suite checks whether that scorer hardening
overfit the cue-balanced wrapper. It uses 16 paired contrasts / 32 prompts across
8 mechanisms: silence-as-consent, hidden objections, verification blocking,
unsafe exit, background data collection, no-appeal safety rules, social-debt
pressure, and forced forgiveness. The scorer gets 16/16, with +0.134 mean score
margin and +0.709 mean autonomy-safety margin. The simple lexical leakage gate
solves only 4/16 pairs, ties 9/16, and inverts 3/16, with mean cue margin 0.000.
The small Modal Qwen pass gets 1.000 in-sample accuracy but 0.875
leave-one-pair-out accuracy; the misses are dialogue-style verification/proof
and dialogue-style silence-as-consent. This gives a crisp next target for
generated variants.

The first model/layer sweep makes that sharper. On the autonomy stress prompts,
`Qwen/Qwen2.5-0.5B-Instruct` reaches 0.875 leave-one-pair-out accuracy at the
final layer, but 1.000 at layers -2 and -4. `Qwen/Qwen2.5-1.5B-Instruct`
reaches 0.938 at the final layer and 1.000 at layer -2. A new multi-direction
subspace probe separates signed discrimination from squared localization:
1.5B layer -2 reaches 1.000 best pair-LOO signed-vote accuracy, while squared
subspace-energy accuracy is only 0.750. That means the sign of the direction is
not a disposable detail; squared energy can make a localization result look
strong while erasing which pole the component supports.

The newest conceptual update adds a boundary-prior lens from Sandved-Smith,
Fields, Doctor, Laukkonen, and Hohwy's "There is no self-evidence." We are not
using its quantum/free-energy argument as evidence for our activation results.
The useful translation is computational: social failures can come from rigid
self/other or us/them partitioning, and also from coercive unity language that
collapses boundaries. Healthy cohesion should preserve flexible,
context-sensitive boundaries: consent, refusal, privacy, exit, dissent, and
responsibility stay real even when relation deepens.

That lens now has a first benchmark export. The boundary-prior suite has 12
matched pairs / 24 activation prompts across 6 mechanisms: evidence across
groups, consent in shared identity, dissent and loyalty, privacy in solidarity,
repair without absorption, and shared resources with subsidiarity. Each
mechanism has one rigid-boundary negative and one boundary-collapse negative.
The local scorer prefers the contextual-relation side on 12/12 pairs, with a
+0.167 mean score margin and a +0.686 mean autonomy-safety margin. The leakage
gate is not clean yet: simple cue counting solves 5/12 pairs, ties 5/12, and
inverts 2/12, with a +0.583 mean cue margin. So the lane is ready for Modal and
paraphrase hardening, but it is not yet a robust semantic benchmark.

The first boundary-prior Modal sweep is also complete on
`Qwen/Qwen2.5-0.5B-Instruct` layers -1, -2, and -4. Each layer gets 1.000
leave-one-pair-out accuracy over the 12 pairs, with mean margins +13.500,
+2.875, and +2.465. Mechanism directions are moderately aligned rather than
orthogonal: mean signed/absolute cosine is +0.488, +0.424, and +0.430 across
the three layers. After projecting out the global direction, mechanism-specific
residual directions still separate their groups. The signed-vs-squared result
is the big warning: signed subspace voting stays 1.000, while best pair-LOO
squared-energy accuracy is only 0.417, 0.500, and 0.583.

The cue-balanced boundary-prior variant is stronger. It keeps the 12-pair /
24-prompt shape but removes the simple cue leak: 0/12 cue-solved pairs, 12/12
tied, and 0.000 mean cue margin. The scorer still prefers the contextual side
on 12/12 pairs, with +0.123 mean score margin and +0.605 mean autonomy-safety
margin. Modal activations still separate the set: Qwen 0.5B reaches 1.000 LOO
at layers -1, -2, and -4, and Qwen 1.5B reaches 1.000 at layers -1 and -2.
Mean LOO margins are +14.514, +2.666, +2.331, +8.461, and +11.137. Mechanism
directions remain moderately aligned rather than independent, and residual
mechanism-specific directions still separate all six groups.

The latest boundary-prior run expands that cue-balanced scaffold to 36 pairs /
72 prompts by adding three neutral record genres to each base contrast. The
simple leakage gate stays exactly tied: 0/36 cue-solved pairs, 36/36 tied, and
0.000 mean cue margin. The scorer keeps 36/36 contextual-side wins with +0.123
mean score margin and +0.605 mean autonomy-safety margin. The Modal sweep again
survives across Qwen 0.5B layers -1/-2/-4 and Qwen 1.5B layers -1/-2: all five
runs are 1.000 leave-one-pair-out. Mean margins are +14.183, +2.732, +2.309,
+8.357, and +10.976. This is still controlled deterministic text, but it makes
the boundary-prior signal less likely to be just a 12-pair batch artifact.

The social-game validation scaffold adds five small game-theoretic probes:
dictator need sensitivity, public-goods free riding, ultimatum fairness, trust
with verification, and restorative repair. The local scorer prefers the
prosocial side on 5/5 after the autonomy-safety hardening, including the trust
verification case and the ultimatum exit-rights case.

The trait-axis export now covers 8 axes / 16 contrasts / 32 prompts. New axes
include constructive dissent vs conformity, manipulation resistance vs
persuasion capture, and privacy/exit rights vs surveillance lock-in. A guardrail
monitoring script is now ready to train one direction per axis once the
trait-axis prompts have a Modal activation file.

Those two small validation lanes have now reached Modal on
`Qwen/Qwen2.5-0.5B-Instruct`. The social-game file produced 10 x 896 activations
and a 5-pair contrastive-vector smoke with 1.000 leave-one-pair-out accuracy and
+8.548 mean margin. The trait-axis file produced 32 x 896 activations; guardrail
monitoring reports 8 axes, 16 pairs, 0 alerts, and +15.382 mean margin. These
are still hand-authored smoke tests, but they show the new validation paths are
not just documents.

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
4. Extend the model/layer sweep beyond Qwen 0.5B/1.5B and rerun it on generated
   variants, not just the hand-authored autonomy stress prompts.
5. Break the target into multiple persona-vector-style axes.
6. Re-run GPT-2 SAE token inspection on LLM-authored pseudo-cohesion variants.
7. Compare clean deterministic transfer against generated hard-negative
   transfer.
8. Extend the fault taxonomy to generated hard negatives and run held-out
   fault-class transfer.
9. Require lexical leakage reports for every generated pairwise dataset.
10. Run trait-axis activations through the guardrail monitor and inspect weak or
   inverted margins by axis.
11. Push more social-game validation variants through Modal/open models and
   compare activation margins against the hardened scorer.
12. Expand the autonomy stress suite around the Qwen LOO misses:
   dialogue-style verification/proof and dialogue-style silence-as-consent.
13. Replace the controlled boundary-prior genre expansion with generated or
   API-authored cue-balanced wording diversity, then rerun leakage, Modal
   activation, geometry, residual, and signed-vs-squared subspace reports.
14. Generate API-authored cue-balanced variants with more wording diversity and
   rerun leakage, component, and activation held-out reports.
15. Run direction-geometry, residual-subspace, and signed-vs-squared subspace
   audits before claiming that trait or fault directions are independent,
   orthogonal, localized, or exhausted by one vector.
16. Preserve signed projections in SAE/localization reports; squared projection
   energy is useful, but the current 1.5B layer -2 autonomy result shows it can
   erase which pole a feature supports.
17. Run a small Prolific validation only after the compute benchmarks stop being
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
