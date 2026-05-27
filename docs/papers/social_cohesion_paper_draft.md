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
features in model activation space. This motivates open-model activation capture
and contrastive direction training.

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

The current pairwise ranking baselines are deliberately simple:

- chance;
- strategy-profile prior;
- metrics-only composite;
- lexical-only heuristic;
- full-scorer sanity check.

In the current scripted data, the task is easy. Metrics-only and lexical-only
baselines recover the pair labels almost perfectly. This is expected because the
dataset is generated from scripted agents and a transparent rubric. The result
is useful as a warning: the first real benchmark must use held-out
LLM-generated trajectories or human labels so that surface markers cannot solve
the task.

## 6. Planned Activation Experiments

The immediate GPU experiment:

1. Extract hidden-state activations for all 252 activation prompts.
2. Train a contrastive direction from positive vs negative examples.
3. Evaluate pairwise ranking accuracy on held-out pairs.
4. Compare against lexical and metrics-only baselines.
5. Repeat by layer and model size.

The next nontrivial experiment:

1. Generate new trajectories with an open instruction model.
2. Score them with the rubric.
3. Train directions on scripted trajectories.
4. Test transfer to generated trajectories.
5. Look for failure modes where generic positivity, sycophancy, or compliance
   are mistaken for cohesion.

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

1. Run GPU activation extraction on the 252 prompt records.
2. Train and evaluate the first activation contrastive vector.
3. Add LLM-generated trajectory generation.
4. Add adversarial pseudo-cohesion examples.
5. Add persona-vector-style trait decomposition:
   - repair;
   - reciprocity;
   - truthfulness;
   - autonomy safety;
   - sycophancy/compliance;
   - coercion/domination;
   - dehumanization;
   - punitive escalation.
6. Prepare a Prolific pairwise validation pilot.

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

