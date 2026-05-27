# Learning Candidate Social Cohesion Vectors From Simulated Social Dilemmas

## Abstract

We introduce an early-stage compute-first pipeline for studying whether
cooperative social orientation can be represented as interpretable directions in
language-model activation space. The current system defines 25 social-dilemma
scenarios spanning repeated Prisoner's Dilemmas, public goods, negotiation,
dialogue repair, and resource allocation. Deterministic scripted agents generate
cooperative, self-protective, and adversarial trajectories under six
intervention conditions. A transparent rubric scores each trajectory for
cooperation, repair, fairness, hostility, truthfulness, and autonomy safety, then
exports pairwise high/low examples for baseline ranking, activation capture, and
future human validation. The initial local run produced 450 trajectories, 126
pairwise probe examples, and 252 activation prompts. These results are
preliminary and deliberately non-behavioral: they establish a reproducible
scaffold and sanity baselines, not evidence that the learned directions affect
humans. The next experiments replace scripted text with open-model generated
trajectories, extract hidden-state activations on GPU, train contrastive
directions, and evaluate whether activation-space projections generalize beyond
lexical artifacts.

## 1. Introduction

Language and multimodal generative models are increasingly able to produce
persuasive content. Most optimization pipelines target engagement, preference,
retention, conversion, or task success. This project explores a different target:
whether content generation can be steered toward agency-preserving social
cohesion.

The central scientific question is not whether there is a single moral vector.
There almost certainly is not. The question is whether social modes such as
reciprocity, repair, fairness, truthfulness, dehumanization, coercion, and
punitive escalation have measurable structure in model behavior and activations.
If they do, we can ask whether combinations of these directions predict or steer
outputs toward healthier conflict trajectories without degrading truthfulness or
autonomy.

## 2. Conceptual Background

The project combines four lineages.

First, Aristotle's ethical and political frame treats human flourishing as partly
social. A good life is not merely preference satisfaction; it is activity in
accordance with virtue, developed in relation and institutions.

Second, Axelrod's iterated-game work shows how cooperation can emerge among
self-interested agents when interactions repeat and strategies can condition on
prior behavior. Repetition, reciprocity, forgiveness, and clarity are central.

Third, social psychology shows that group identity, intergroup contact, status,
and threat shape cooperation beyond payoff matrices. Cohesion must therefore be
distinguished from conformity and from majority pressure.

Fourth, mechanistic interpretability and persona-vector work suggest that
high-level behavioral traits can sometimes be represented as directions or sparse
features in model activation space. This motivates open-model activation capture,
contrastive direction training, and a trait-family view rather than a single
monolithic "cohesion" direction. The relevant Anthropic persona-vector angle is
especially concrete for this project: trait vectors can be used to monitor model
state before output is emitted, can be composed or steered in activation space,
and need guardrails so "cohesion" does not collapse into compliance,
sycophancy, or dissent suppression.

## 3. Methods

### 3.1 Scenario Benchmark

The current benchmark includes 25 hand-authored scenarios across five kinds:

- iterated Prisoner's Dilemma;
- public goods;
- negotiation;
- dialogue repair;
- resource allocation.

Each scenario specifies agents, goals, risks, cooperative actions, defecting
actions, and success metrics.

### 3.2 Scripted Simulation

For each scenario, deterministic scripted agents generate trajectories under
three strategy profiles:

- cooperative;
- self-protective;
- adversarial.

Each profile is crossed with six intervention conditions:

- no intervention;
- shared identity;
- perspective-taking;
- reciprocity;
- restorative accountability;
- truth-first framing.

The first local run generated 450 trajectories.

### 3.3 Scoring

Each trajectory is scored with explicit components:

- cooperation;
- repair;
- fairness;
- hostility inverse;
- truthfulness;
- autonomy safety.

The combined score is a weighted aggregate with bonuses for cooperative repair
and penalties for hostility or manipulation risk. This score is interpretable
but not yet externally validated.

### 3.4 Pairwise Dataset

Within each scenario, high-scoring and low-scoring trajectories are paired when
their score margin exceeds a threshold. The first local run produced 126
pairwise examples and 252 activation prompt records.

### 3.5 Activation Capture

The Modal lane targets open-weight causal language models. For each activation
prompt, the system captures hidden states at a configured layer, mean-pools over
tokens, and stores activation vectors with sample IDs, labels, and target
scores. Contrastive directions can then be trained from high/low examples.

## 4. Current Compute-Only Results

The current scripted run is a sanity benchmark.

- Scenarios: 25
- Runs: 450
- Pairwise examples: 126
- Activation prompts: 252
- Mean pairwise score margin: approximately 0.272
- Minimum pairwise margin: approximately 0.154
- Maximum pairwise margin: approximately 0.558

Mean cohesion score by strategy profile:

- adversarial: 0.620
- self-protective: 0.736
- cooperative: 0.855

Mean cohesion score by intervention:

- none: 0.694
- reciprocity: 0.727
- truth-first: 0.740
- shared identity: 0.746
- perspective-taking: 0.752
- restorative accountability: 0.762

These numbers are not claims about humans. They verify that the scripted
benchmark creates graded trajectories and that interventions move scores in the
expected direction under the current rubric.

## 5. Baselines

The current pairwise ranking baselines are deliberately simple. On the 126
pair scripted dataset:

- chance: 0.500;
- strategy-profile prior: 0.988, 95% bootstrap CI [0.972, 1.000];
- metrics-only composite: 1.000;
- lexical-only heuristic: 1.000;
- full-scorer sanity check: 1.000.

In the current scripted data, the task is easy. Metrics-only and lexical-only
baselines recover the pair labels almost perfectly. This is expected because the
dataset is generated from scripted agents and a transparent rubric. The result
is useful as a warning: the first real benchmark must use held-out
LLM-generated trajectories or human labels so that surface markers cannot solve
the task.

## 6. Activation And Persona-Vector Roadmap

The first GPU activation experiment was run with `Qwen/Qwen2.5-0.5B-Instruct`,
using final-layer hidden states mean-pooled over tokens. The resulting activation
matrix has 252 prompt rows and 896 dimensions. A contrastive positive-vs-negative
direction gives:

- in-sample pairwise accuracy: 1.000 on 126 pairs;
- leave-one-pair-out pairwise accuracy: 1.000 on 126 pairs;
- leave-one-pair-out mean positive-minus-negative projection margin: +26.4385.

This is not yet strong evidence because the source trajectories remain scripted
and lexically separable. It does show that the open-model activation lane is
working end to end: local JSONL prompts → Modal GPU hidden states → local NPZ →
contrastive vector → pairwise evaluation.

The roadmap below separates completed artifacts from pending experiments. All
items marked pending require new runs before any result should be reported.

### 6.1 Generated Trajectories

Status: partial.

The next benchmark should replace or augment scripted trajectories with
open-model generated trajectories. Each generated run should preserve the
scenario, strategy pressure, and intervention condition, but should not reuse the
scripted lexical templates. The goal is to test whether the scorer and activation
directions identify cooperative repair, reciprocity, truthfulness, and autonomy
safety when the wording is less obvious.

The current local artifact includes 125 generated trajectories from the offline
fallback generator, covering 25 scenarios and five trajectory styles. These have
now been converted into 125 scored generated runs, 50 generated pairwise
examples, and 100 generated activation prompts. On this first generated
benchmark, the strategy-prior, metrics-only, and lexical-only baselines each
reach approximately 0.980 pairwise accuracy, while the full scorer reaches
1.000. API-generated trajectories from an external model remain pending.

Next artifacts: external-model generated trajectories, harder generated
pseudo-cohesion examples, and a report where lexical, metrics-only, and
metadata-prior baselines no longer nearly solve the benchmark.

### 6.2 Pseudo-Cohesion Hard Negatives

Status: complete for the expanded hand-authored probe.

The benchmark needs adversarial examples that sound cooperative while violating
the target construct. The first hard-negative families are:

- polite coercion framed as team unity;
- sycophantic agreement that hides evidence;
- compliance maximization framed as harmony;
- dissent suppression framed as repair;
- false accountability that protects status rather than repairing harm.

These examples should be treated as negative examples even when they contain
positive social language. They are the main guardrail against learning a vector
for agreeableness, obedience, or generic warmth instead of agency-preserving
cohesion.

The expanded hard-negative run uses 60 hand-authored examples: 30
pseudo-cohesion cases and 30 matched genuine contrasts. The current scorer
assigns high scores to 8 pseudo-cohesion examples, and the lexical-only baseline
assigns high scores to 18 pseudo-cohesion examples. These are expected failure
cases for the next scorer and dataset iteration, not evidence that the pseudo
cases are actually cohesive.

The first matched GPT-2 SAE pass on the expanded prompt set gives a useful
caveat. Mean residual activations separate the 30 pairs with 0.967
leave-one-pair-out accuracy, but SAE feature activations reach only 0.533. The
aggregate feature ranking still surfaces candidate sparse features, including
3056 on the genuine side and 24555/28005/20249/11999 on the pseudo side, but
these require token- and example-level inspection before any feature naming.

A Qwen 0.5B Modal pass on the same expanded prompts also reaches 0.967
leave-one-pair-out accuracy with a +28.6866 mean margin. Its single failure is
the resource-request contrast, where the pseudo social-debt pressure example and
the genuine reciprocal-request example currently receive the same rubric score.

Token-level SAE inspection further constrains the interpretation. Feature 3056
remains the best genuine-skew candidate, while 24555, 11737, and 703 remain
pseudo-skew candidates. Feature 11737 is the most semantically suggestive
pseudo-side feature because it activates on `you`/`comply` in the
autonomy/coercion contrast. Features 28005 and 20249 should be demoted: 28005 is
mostly a single `mutual-aid` hyphen artifact, and 20249 is inactive when
activations are encoded token by token.

Next artifacts: generated pseudo-cohesion examples, token-level SAE feature
inspection, and hard-negative-held-out transfer reports.

### 6.3 Transfer Splits

Status: partial.

The current transfer report includes scripted held-out scenario-id,
scenario-kind, and explicit scripted/generated pair-set folds. These are still
largely solved by lexical-only, metrics-only, and strategy-prior baselines, so
they remain sanity checks rather than evidence of robust generalization.
Scripted-to-generated transfer currently gives 0.960 for lexical-only and
metrics-only baselines and 0.980 for the strategy prior. Generated-to-scripted
transfer remains 1.000 for lexical-only and metrics-only baselines.

Next artifacts: intervention-held-out, hard-negative-held-out, and
external-generated transfer evaluations. Any 1.000 accuracy result on scripted
or offline-generated splits should remain labeled as a sanity check until
harder generated or pseudo-cohesion transfer performance is reported.

### 6.4 Layer And Model Sweeps

Status: pending beyond the final-layer Qwen run.

The next activation pass should repeat on a larger open instruction model if
compute allows. The first generated Qwen 0.5B sweep now covers layers `-1`,
`-2`, `-4`, and `-8`. Leave-one-pair-out accuracy remains 1.000 on all four
layers, while mean projection margin falls from approximately +26.36 at the
final layer to +2.34 at layer `-8`. The success criterion is not the highest
in-sample score; it is transfer stability after lexical and metadata baselines
are controlled.

### 6.5 Persona-Vector Decomposition

Status: pending.

The single cohesion direction should be decomposed into a family of trait
directions inspired by persona-vector work. The first scaffold now exports 20
activation prompts across 5 seed axes: repair vs harm denial, reciprocity vs
extraction, truthfulness vs deception, autonomy support vs coercion, and
principled respect vs sycophancy. Candidate future positive directions include
repair, reciprocity, truthfulness, autonomy safety, fairness, and constructive
dissent. Candidate negative or guardrail directions include
coercion/domination, sycophancy, compliance-seeking, dehumanization, truth
hiding, and punitive escalation.

The intended use is twofold. First, monitoring: compute projections before the
model emits final output and flag generations where negative guardrail
directions rise alongside nominally pro-cohesion language. Second, steering and
composition: test whether small combinations such as `repair + truthfulness +
autonomy_safety - sycophancy - compliance` improve outputs without suppressing
legitimate refusal, disagreement, or self-protection. All steering claims remain
pending until controlled generation tests are run.

## 7. Ethics And Safety

The target is not agreement maximization. The system must explicitly preserve:

- truthfulness;
- autonomy;
- principled dissent;
- minority interests;
- legitimate self-protection;
- informed consent in human studies.

Any future content-selection loop must include manipulation-risk scoring and
must not deploy claims about behavioral effects before Prolific or comparable
human validation.

## 8. Limitations

The current results are synthetic and circular: pair labels are generated from
the same rubric used by the full-scorer baseline. Scripted trajectories include
surface lexical cues that make the task too easy. The benchmark does not yet
show that activation directions generalize, steer outputs, or predict human
behavior. It is a scaffold for those tests.

## 9. Next Experiments

1. Generate external-model trajectories with reduced template leakage and score
   them through the generated benchmark. Status: pending.
2. Extend the pseudo-cohesion hard-negative suite beyond the first hand-authored
   probe and use current scorer failures to revise the dataset. Status: partial.
3. Build transfer splits: scripted-to-generated, generated-to-scripted,
   intervention-held-out, and hard-negative-held-out. Status: partial for
   scripted scenario-held-out only.
4. Repeat the generated activation sweep on a larger model if compute allows.
   Status: pending.
5. Expand the persona-vector-style trait families:
   - repair;
   - reciprocity;
   - truthfulness;
   - autonomy safety;
   - fairness;
   - constructive dissent;
   - sycophancy/compliance;
   - coercion/domination;
   - dehumanization;
   - truth hiding;
   - punitive escalation.
6. Test monitoring before output and steering/composition only after transfer
   splits show non-circular signal. Status: pending.
7. Prepare a Prolific pairwise validation pilot only after generated-text and
   hard-negative validation. Status: pending.

## References

- Allport, G. W. (1954). *The Nature of Prejudice*.
- Anthropic. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."
- Anthropic. (2025). "Persona vectors: Monitoring and controlling character traits in language models."
- Aristotle. *Nicomachean Ethics*.
- Axelrod, R. (1984). *The Evolution of Cooperation*.
- Nowak, M. A. (2006). "Five Rules for the Evolution of Cooperation." *Science*.
- Ostrom, E. (1990). *Governing the Commons*.
- Rilling, J. K., et al. (2002). "A neural basis for social cooperation." *Neuron*.
- Tajfel, H., & Turner, J. C. (1979/1986). Social identity theory of intergroup behavior.
