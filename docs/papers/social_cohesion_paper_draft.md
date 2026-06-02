# Learning Candidate Social Cohesion Vectors From Simulated Social Dilemmas

## Abstract

We introduce an early-stage compute-first pipeline for studying whether
cooperative social orientation can be represented as a constrained family of
interpretable directions in language-model activation space. The current system
defines 25 social-dilemma scenarios spanning repeated Prisoner's Dilemmas,
public goods, negotiation, dialogue repair, and resource allocation.
Deterministic scripted agents generate cooperative, self-protective, and
adversarial trajectories under six intervention conditions. A transparent rubric
scores each trajectory for cooperation, repair, fairness, hostility,
truthfulness, and autonomy safety, then exports pairwise high/low examples for
baseline ranking, activation capture, and future human validation. The initial
local run produced 450 trajectories, 126 pairwise probe examples, and 252
activation prompts; later local runs added pseudo-cohesion hard negatives,
GPT-2 sparse-autoencoder probes, and Qwen activation baselines.

The paper updates the project frame in light of recent work on sparse
autoencoders, persona vectors, refusal and sycophancy directions, plural value
steering, altruism features, persuasion risk, and LLM-agent social dynamics. It
also adopts a stronger normative grounding from *Magnifica Humanitas*: social
cohesion should mean truthful, dignity-preserving, agency-respecting relation,
not conformity, dependence, manipulation, or optimized agreement. These results
remain preliminary and deliberately non-behavioral: they establish a
reproducible scaffold and sanity baselines, not evidence that learned directions
affect humans. The next experiments replace scripted text with open-model
generated trajectories, extract hidden-state activations on GPU, train
contrastive and sparse-feature directions, and evaluate whether activation-space
projections generalize beyond lexical artifacts.

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

The project combines six lineages.

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

Fifth, *Magnifica Humanitas* gives the project a sharper normative frame for AI:
technology is never socially neutral in practice, because it bears the aims and
power relations of those who devise, finance, govern, and deploy it. Its contrast
between Babel and the rebuilding of Jerusalem is directly useful here. Babel
names the failure mode of uniformity, self-assertion, and performance metrics
that sacrifice human dignity. Jerusalem names participatory rebuilding: shared
responsibility, plural roles, dialogue, truth, solidarity, and the dignity of
persons who cannot be reduced to data, productivity, or persuasion targets.

Sixth, recent pluralistic alignment and AI-governance work pushes against
single-objective optimization. "Prosocial" behavior is not one scalar; it is a
constraint-sensitive family of values that must survive cultural pluralism,
contested goods, uncertainty, and asymmetric power. This makes the research task
both mathematical and political: learn candidate directions, but test whether
they preserve truth, dissent, privacy, exit rights, and minority agency.

Seventh, recent formal work on self-evidencing and emptiness is useful as a
theory of boundary priors. Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy
argue that a finite agent cannot obtain evidence for the ontological reality of
its own self/environment boundary, and they model separation as a structural
prior over admissible reference-frame deployments. This paper should not be used
as empirical support for the activation results, but it gives a valuable
computational lens: social failures often involve rigid boundary priors
("us/them," "self/world," "loyal/disloyal") or coercive boundary collapse ("we
are one, so refusal is betrayal"). The constructive target is neither rigid
separation nor forced unity, but flexible, context-sensitive boundary use under
truth, consent, and agency constraints.

### 2.1 Normative Construct: Cohesion Without Domination

This project uses the following working definition:

> Social cohesion is the capacity of persons and groups to remain in truthful,
> dignity-preserving, agency-respecting relation while pursuing shared goods or
> principled non-agreement under conflict.

This definition rules out several tempting shortcuts. Agreement is not cohesion
when it is purchased through flattery, dependency, surveillance, status threat,
or suppression of dissent. Engagement is not cohesion when the content is false
or manipulative. Calm language is not cohesion when it hides coercion. A
technically successful intervention is therefore insufficient unless it also
preserves:

- dignity prior to productivity or performance;
- truth as a common good rather than a local persuasion tactic;
- solidarity as responsible interdependence rather than mere network density;
- subsidiarity and agency, including real participation and meaningful exit;
- pluralism, dissent, and self-protection for vulnerable parties.

These constraints explain why the repo now treats pseudo-cohesion as a first
class negative family. The system should reject polite domination, harmony-coded
coercion, social-debt pressure, scapegoating, verification blocking, and
sycophantic agreement even when they use warm or communal language.

### 2.2 Mathematical Frame: A Constrained Vector Family

The mathematical target is not a single cohesion vector. It is a family of
interpretable directions and sparse features whose composition can be evaluated
under hard guardrails. For a construct axis `k`, a simple residual-space
contrastive direction can be estimated as:

```text
w_k = mean(h_l(x_positive_k)) - mean(h_l(x_negative_k))
s_k(x) = <normalize(h_l(x)), normalize(w_k)>
```

For sparse autoencoder features, the corresponding score can be represented as a
weighted sparse-feature mixture:

```text
q_k(x) = sum_j a_kj * normalize(z_lj(x))
```

where `h_l` is the hidden state at layer `l`, `z_lj` is SAE feature `j`, and
`a_kj` is a signed feature weight for construct axis `k`. The practical research
objective is then closer to constrained optimization than unconstrained
preference maximization:

```text
maximize C(x) = repair + reciprocity + fairness + calibrated_trust
subject to truth(x) >= tau_truth
           autonomy(x) >= tau_autonomy
           dissent_safety(x) >= tau_dissent
           privacy(x) >= tau_privacy
           exit_rights(x) >= tau_exit
           manipulation_risk(x) <= tau_manipulation
```

This framing matters because an apparent gain in cooperation can be harmful if
it is driven by sycophancy, deception, surveillance, or dependence. The next
research phase should therefore report vector effects as a multi-axis profile:
what went up, what went down, which guardrails were preserved, and which failure
modes were activated.

An equivalent abstract view is that the project studies signed covectors and
subspaces over representation maps `phi_l : X -> V_l`, where `X` is a space of
social situations or interventions and `V_l` is an activation space. Cohesion is
not a global coordinate on `X`; it is a constrained feasible region in a
multi-axis space. Boundary/contextuality work adds another latent variable:
which partition or frame the agent is using to distinguish self, other, group,
outgroup, refusal, obligation, and shared good. One next benchmark family should
therefore compare rigid boundary reification, flexible contextual relation, and
coercive boundary collapse.

## 3. Recent State Of The Art And Implications

### 3.1 Sparse Autoencoders And Feature-Level Control

Recent sparse-autoencoder work makes the compute-first plan much more concrete.
Scaling studies and open SAE releases provide practical dictionaries for
inspecting features across layers and models, while newer SAE variants improve
fidelity at fixed sparsity. This supports the repo's current strategy: start
with open-weight models and public SAEs, identify candidate sparse features, and
only then consider training project-specific dictionaries.

The safety lesson is just as important as the tooling. Refusal, sycophancy, and
other alignment-relevant behaviors can sometimes be monitored or steered through
low-dimensional directions or small neuron sets. That is useful, but brittle:
one narrow direction can overfit, produce side effects, or become a jailbreak
surface. The repo should therefore keep using pair-level, token-level,
fault-class, and transfer-split reports before naming any feature.

### 3.2 Persona, Psychological, Cultural, And Cognitive Steering

Persona-vector and psychological-steering work suggests that behavioral traits
can be monitored before output and modulated in activation space. Cultural-value
and plural-value steering work adds a crucial warning: trait directions are not
independent knobs. A direction that increases helpfulness, agreeableness, or
group harmony may also move sycophancy, truthfulness, authority deference,
autonomy, or cultural assumptions.

The direct implication is to treat "social cohesion" as a vector family with
guardrails: repair, reciprocity, fairness, truthfulness, constructive dissent,
anti-coercion, anti-sycophancy, anti-dehumanization, and manipulation
resistance. Monitoring should report the covariance among these axes instead of
celebrating a single high projection score.

### 3.3 Cooperative AI, Social Dynamics, And LLM Agents

Recent cooperative-AI and LLM-agent papers are a natural next benchmark layer.
They move beyond isolated model outputs toward direct reciprocity, indirect
reciprocity, public goods, reputation, value misalignment, power-seeking, and
collective belief dynamics. This is exactly where this project should go after
the local hard-negative suite: test whether a candidate steering or monitoring
method improves group outcomes without increasing deception, conformity, or
domination.

The emerging altruism-mechanism work is especially close to the repo. It uses
game prompts, sparse features, activation patching, and steering to investigate
prosocial behavior in LLMs. That suggests a direct replication path: Dictator
Game, Public Goods, ultimatum, restorative dialogue, and intergroup-dispute
prompts, each paired with hard negatives for social-debt pressure, scapegoating,
verification blocking, and dissent suppression.

### 3.4 Persuasion, Narrative, And Manipulation Risk

Persuasive and narrative LLM outputs can change human reliance and judgment. In
some settings this can help people use correct advice; in others it can make
false or manipulative advice more compelling. For this project, that means
persuasion is not a success metric by itself. A model that "brings people
together" by making false consensus feel emotionally satisfying has failed the
construct.

The research program should therefore score narrative, affective, and persuasive
force separately from truthfulness, autonomy, and dissent safety. Any future
Prolific pilot should include manipulation-risk checks before measuring whether
participants perceive a message as cohesive or constructive.

### 3.5 Brain-LLM Bridges

Recent work connecting SAEs to brain-LLM alignment and cortical semantic
topography makes the longer-term neural ambition more plausible, but it should
not be overread. Open-model activations and SAE features can generate hypotheses
about semantic and social representations; they cannot by themselves establish
human neural cohesion vectors. The responsible sequence is text benchmark,
open-model activation, causal intervention, human behavioral validation, and
only then neural or hyperscanning-style studies if the behavioral signal is
strong enough.

### 3.6 Research Directions From The Recent Literature

The next phase should prioritize:

1. fault-held-out generation, where LLM-authored examples are grouped by
   symbolic failure mode and tested on unseen fault classes;
2. multi-axis monitoring, where repair or reciprocity gains must not raise
   sycophancy, coercion, deception, or conformity scores;
3. selective steering with small coefficients and explicit side-effect reports;
4. multi-agent social-dynamics tests for direct reciprocity, indirect
   reciprocity, reputation, public goods, and long-horizon cooperation;
5. plural and cultural stress tests, where the same vector family must preserve
   dignity and agency under different values, norms, and conflict styles;
6. persuasion-risk evaluation, where narrative force is treated as a potential
   hazard until truth and autonomy checks pass;
7. human validation only after the compute-only artifacts stop being solved by
   lexical or metadata priors.

## 4. Methods

### 4.1 Scenario Benchmark

The current benchmark includes 25 hand-authored scenarios across five kinds:

- iterated Prisoner's Dilemma;
- public goods;
- negotiation;
- dialogue repair;
- resource allocation.

Each scenario specifies agents, goals, risks, cooperative actions, defecting
actions, and success metrics.

### 4.2 Scripted Simulation

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

### 4.3 Scoring

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

### 4.4 Pairwise Dataset

Within each scenario, high-scoring and low-scoring trajectories are paired when
their score margin exceeds a threshold. The first local run produced 126
pairwise examples and 252 activation prompt records.

### 4.5 Activation Capture

The Modal lane targets open-weight causal language models. For each activation
prompt, the system captures hidden states at a configured layer, mean-pools over
tokens, and stores activation vectors with sample IDs, labels, and target
scores. Contrastive directions can then be trained from high/low examples.

## 5. Current Compute-Only Results

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

## 6. Baselines

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

## 7. Activation And Persona-Vector Roadmap

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

### 7.1 Generated Trajectories

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

### 7.2 Pseudo-Cohesion Hard Negatives

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

The next expanded inspection pass wraps each of the 30 hand-authored contrasts
in three neutral contexts: meeting note, facilitator script, and policy update.
Together with the seed set, this produces 120 matched pairs and 240 activation
prompts. On this batch, a signed ensemble over seven inspected GPT-2 SAE
features reaches 0.825 leave-one-pair-out accuracy using mean activations and
0.758 using max activations. The best single mean-activation feature is 703 at
0.792. Feature 3056 remains genuinely skewed at the token level, but its
single-feature leave-one-pair-out accuracy is only 0.600, so it should be
treated as a useful candidate sub-feature rather than a named cohesion feature.
This run also exposes artifact risk: 28005 remains a hyphen artifact, 20249 is
inactive, and 11737 becomes partly punctuation-sensitive under the wrapper
variants.

A cleaner deterministic variant removes these genre wrappers and instead uses
in-text term rewrites with hyphen normalization. On the clean 120-pair batch,
the same inspected GPT-2 SAE feature ensemble reaches 0.892 leave-one-pair-out
accuracy using mean activations. On clean-only variants without the seed prompts,
the ensemble reaches 0.889 over 90 pairs. This cleaner run confirms that the
ensemble signal is not merely a wrapper artifact, but it does not yet justify
feature naming. The best single clean mean-activation feature is 11999 at 0.800,
yet its top activations remain generic and `Your`-token heavy. Feature 703
reaches 0.775 and remains function-word heavy. Feature 3056 becomes more
genuine-skewed at the token level, but reaches only 0.617 as a standalone
held-out classifier. Feature 28005 becomes fully inactive when the seed prompts
are removed, confirming the hyphen-artifact diagnosis, and 20249 remains
inactive.

The newest fault-taxonomy report groups the 30 seed contrasts by symbolic
failure mode. Feature 3056 is genuinely skewed on reality-denial,
social-debt, assimilation-pressure, exit-rights, and privacy contrasts, but it
is pseudo-skewed on verification-blocking and scapegoating. This makes it more
interesting, not more nameable: 3056 currently looks like a fault-sensitive
sub-feature that may help distinguish some agency-preserving repair cases, not
a general "cohesion" feature.

Next artifacts: LLM-authored pseudo-cohesion variants, token-level SAE feature
inspection on those variants, and hard-negative transfer reports that compare
deterministic rewrites against generated examples.

### 7.3 Transfer Splits

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

### 7.4 Layer And Model Sweeps

Status: pending beyond the final-layer Qwen run.

The next activation pass should repeat on a larger open instruction model if
compute allows. The first generated Qwen 0.5B sweep now covers layers `-1`,
`-2`, `-4`, and `-8`. Leave-one-pair-out accuracy remains 1.000 on all four
layers, while mean projection margin falls from approximately +26.36 at the
final layer to +2.34 at layer `-8`. The success criterion is not the highest
in-sample score; it is transfer stability after lexical and metadata baselines
are controlled.

### 7.5 Persona-Vector Decomposition

Status: pending.

The global cohesion direction should be decomposed into a family of trait
directions inspired by persona-vector work. The first scaffold now exports 32
activation prompts across 8 seed axes: repair vs harm denial, reciprocity vs
extraction, truthfulness vs deception, autonomy support vs coercion, principled
respect vs sycophancy, constructive dissent vs conformity, manipulation
resistance vs persuasion capture, and privacy/exit rights vs surveillance
lock-in. Candidate future positive directions include repair, reciprocity,
truthfulness, autonomy safety, fairness, and constructive dissent. Candidate
negative or guardrail directions include coercion/domination, sycophancy,
compliance-seeking, dehumanization, truth hiding, and punitive escalation.

The intended use is twofold. First, monitoring: compute projections before the
model emits final output and flag generations where negative guardrail
directions rise alongside nominally pro-cohesion language. Second, steering and
composition: test whether small combinations such as `repair + truthfulness +
autonomy_safety - sycophancy - compliance` improve outputs without suppressing
legitimate refusal, disagreement, or self-protection. All steering claims remain
pending until controlled generation tests are run.

### 7.6 Direction Geometry And Ablation Guardrails

Status: complete for the cue-balanced Qwen 0.5B primary-fault activation set.

A reviewer-style audit was added after the cue-balanced activation results to
avoid three common overclaims. First, a mean off-diagonal signed cosine can
conceal cancellation between strongly aligned and strongly anti-aligned
directions. Second, a cosine near -1.0 is not independence; it is the same
unsigned axis with reversed polarity. Third, projecting out one global
contrastive direction and failing to recover a second global direction does not
prove the representational signal is exhausted.

On the cue-balanced Qwen 0.5B primary-fault directions, the audit finds 20
primary-fault directions and 190 pairwise direction comparisons. The mean signed
off-diagonal cosine is +0.624 and the mean absolute off-diagonal cosine is also
0.624; there are no strong anti-aligned pairs. This is not an orthogonality
result. It suggests a shared positive genuine-vs-pseudo manifold with
fault-specific variation.

The residual subspace audit sharpens the interpretation. The global direction
captures 0.609 of pair-difference energy, leaving 0.391 after projection. A
second global residual contrastive direction collapses on this deterministic
set, but every primary fault class still has a fault-specific residual direction
that separates its own examples with 1.000 mean accuracy. The responsible claim
is therefore:

> The current activation result contains one strong global contrast plus
> meaningful fault-specific residual subspaces. It does not establish
> independent orthogonal persona axes or exhaust the signal with one vector.

Future reports should always include both signed and absolute cosine
distributions, anti-aligned-pair counts, residual pair-difference energy, and
fault-specific residual direction checks before using words like "independent,"
"orthogonal," "localized," or "exhausted."

### 7.7 Structural Autonomy Stress Suite

Status: complete for local scoring, Modal Qwen 0.5B activations, and geometry
audits; partial for model/layer sweeps and signed-vs-squared subspace probes.

The cue-balanced benchmark exposed an autonomy-safety scorer failure, then the
hardened scorer fixed that deterministic set. To check whether the fix merely
learned the cue-balanced wrapper, the repo now includes a wording-diverse
structural-autonomy stress suite. It covers 16 paired contrasts across 8
mechanisms: silence-as-consent, hidden objections, verification blocking, unsafe
exit, background data collection, no-appeal safety rules, social-debt refusal
pressure, and forced forgiveness.

The local scorer prefers the autonomy-preserving side on 16/16 pairs, with a
+0.134 mean score margin and a +0.709 mean autonomy-safety margin. The simple
lexical leakage gate solves only 4/16 pairs, ties 9/16, and inverts 3/16, with a
0.000 mean cue margin. A small Modal pass through `Qwen/Qwen2.5-0.5B-Instruct`
produces 32 x 896 activations. A contrastive direction reaches 1.000 in-sample
accuracy but only 0.875 leave-one-pair-out accuracy; the failures are the
dialogue-style verification/proof case and the dialogue-style silence-as-consent
case.

The geometry audit suggests that structural autonomy is less collapsed than the
primary-fault cue-balanced set. Across 8 mechanism directions, mean signed
off-diagonal cosine is +0.136 and mean absolute cosine is 0.193. The global
direction captures only 0.172 of pair-difference energy, leaving 0.828 after
projection; mechanism-specific residual directions still separate their own
pairs. This supports a subspace interpretation: autonomy risk is not just one
knob, but a family of related checks around refusal, review, evidence, exit,
appeal, privacy, and social pressure.

The first model/layer sweep weakens the "single final-layer vector" framing.
`Qwen/Qwen2.5-0.5B-Instruct` reaches 0.875 leave-one-pair-out accuracy at layer
-1, then 1.000 at layers -2 and -4. `Qwen/Qwen2.5-1.5B-Instruct` reaches 0.938
at layer -1 and 1.000 at layer -2. This suggests the autonomy stress signal is
not confined to the final hidden state, and layer choice should be treated as an
experimental variable rather than a default.

The signed-vs-squared subspace probe directly addresses the reviewer concern
about sign-erasing localization. On Qwen 1.5B layer -2, the best pair-LOO
signed-vote subspace accuracy is 1.000, but squared subspace-energy accuracy is
only 0.750. The first pair-difference component captures just 0.170 of the
energy. The responsible interpretation is therefore not "we found the autonomy
coordinate," but "signed activation structure can separate the autonomy stress
poles, while unsigned energy localization can fail and the variance is spread
across multiple components."

### 7.8 Boundary-Prior Benchmark

Status: complete for local scored export, lexical leakage, Qwen 0.5B activation
sweep, cue-balanced Qwen 0.5B/1.5B activation sweeps, controlled
cue-balanced expansion, activation failure analysis, direction geometry,
residual-subspace audit, and signed-vs-squared subspace probe; pending
generated paraphrase hardening.

The boundary-prior theory note is now operationalized as a first local
benchmark. The suite contrasts flexible contextual relation against two failure
poles: rigid boundary reification and coercive boundary collapse. The positive
examples preserve cooperation while keeping consent, review, privacy, appeal,
exit, and dissent rights available. The rigid-boundary negatives discount the
voice or evidence of an outgroup. The boundary-collapse negatives use unity
language to remove refusal, privacy, appeal, or dissent.

The first export contains 12 matched pairs / 24 activation prompts across 6
mechanisms: evidence across groups, consent in shared identity, dissent and
loyalty, privacy in solidarity, repair without absorption, and shared resources
with subsidiarity. The local scorer prefers the contextual-relation side on
12/12 pairs, with a +0.167 mean score margin, +0.686 mean autonomy-safety
margin, and +0.066 mean truthfulness margin.

The leakage gate is still an important caveat. Simple positive-minus-negative
cue counting solves 5/12 pairs, ties 5/12, and inverts 2/12, with a +0.583 mean
cue margin. This is better than the original generated fault-class dataset, but
worse than the cue-balanced fault-class set and the structural-autonomy suite.
The responsible interpretation is therefore: the boundary-prior lane is now
ready for GPU activation and generated paraphrase experiments, but it is not yet
a robust semantic benchmark.

The first small Modal sweep sends the 24 prompts through
`Qwen/Qwen2.5-0.5B-Instruct` at layers -1, -2, and -4. All three layers reach
1.000 leave-one-pair-out pairwise accuracy. Mean LOO margins are +13.500 at
layer -1, +2.875 at layer -2, and +2.465 at layer -4. Mechanism-direction
geometry is moderately shared rather than orthogonal: mean signed and absolute
off-diagonal cosine are the same at each layer because there are no
anti-aligned mechanism pairs (+0.488, +0.424, +0.430 respectively). Residual
audits show the familiar pattern: the global direction separates, a second
global residual direction collapses, and all six mechanism-specific residual
directions still separate their own groups.

The signed-vs-squared subspace probe again matters. Signed subspace voting is
1.000 at all three layers, while best pair-LOO squared-energy accuracy is only
0.417, 0.500, and 0.583. This reinforces the paper's methodological claim:
unsigned projection energy can be useful as a strength measure, but it is not a
valid substitute for sign-preserving pole localization.

The next cue-balanced pass removes the simple leakage gate while preserving the
same conceptual contrast. The cue-balanced export again contains 12 matched
pairs / 24 prompts. The scorer prefers the contextual-relation side on 12/12
pairs, with a +0.123 mean score margin and +0.605 mean autonomy-safety margin.
The lexical leakage report is now fully tied under the simple cue counter: 0/12
cue-solved pairs, 12/12 tied pairs, and 0.000 mean cue margin.

Modal activations remain separable after this cue balancing. Qwen 0.5B reaches
1.000 leave-one-pair-out accuracy at layers -1, -2, and -4; Qwen 1.5B reaches
1.000 at layers -1 and -2. Mean LOO margins are +14.514, +2.666, +2.331,
+8.461, and +11.137 respectively. Mechanism-direction cosines stay moderately
positive rather than orthogonal: +0.495, +0.361, +0.370, +0.571, and +0.468.
Residual mechanism directions continue to separate all six groups after the
global direction is removed. Squared-energy classification remains layer- and
model-sensitive: 0.583, 0.500, 1.000, 0.667, and 0.583. This strengthens the
boundary-prior result as a compute-only smoke test while preserving the main
caveat: deterministic hand-authored examples are not enough for a semantic or
human behavioral claim.

A controlled expansion now triples the cue-balanced boundary-prior batch by
placing each contrast inside three neutral record genres: case note, meeting
log, and implementation memo. This produces 36 matched pairs / 72 activation
prompts. The simple cue counter remains fully tied: 0/36 cue-solved pairs,
36/36 tied pairs, and 0.000 mean cue margin. The local scorer still prefers the
contextual-relation side on 36/36 pairs, with a +0.123 mean score margin and
+0.605 mean autonomy-safety margin.

The expanded set again separates in open-model activations. Qwen 0.5B reaches
1.000 leave-one-pair-out accuracy at layers -1, -2, and -4; Qwen 1.5B reaches
1.000 at layers -1 and -2. Mean LOO margins are +14.183, +2.732, +2.309,
+8.357, and +10.976. Mean mechanism-direction signed/absolute cosines are
+0.515, +0.371, +0.391, +0.600, and +0.495, with no high-absolute anti-aligned
pairs. Residual mechanism directions again separate all six groups after the
global direction is removed. Squared-energy classification remains less stable
than signed classification: best pair-LOO squared-energy accuracy ranges from
0.583 to 0.944 across the five runs, while signed voting remains 1.000.

This result is a stronger compute-only smoke test, but it should not be
overstated. The expansion is controlled genre wrapping rather than independent
semantic paraphrase generation. The claim supported here is that the
boundary-prior activation signal survives larger cue-balanced deterministic
coverage. The next claim threshold is generated/API-authored paraphrase
diversity with leakage, geometry, residual, and signed-vs-squared audits
rerun from scratch.

### 7.9 Causal Steering And Hidden-State Telemetry

Status: first Modal generation-time intervention harness, steering-method
sweep, generated-output projection check, hidden-state telemetry pass, and
affect-residualized steering comparison complete; initial behavioral results are
weak/mixed.

The next publication threshold is causal. A direction that classifies
positive/negative activation pairs is only a probe unless it can also be used as
an intervention. The repo now includes a first causal steering harness that
adds a signed direction during Modal generation on held-out social decision
prompts, then scores responses at negative, zero, and positive steering
strengths. The tested prompts cover the six boundary-prior mechanisms.

The first steering grid does not yet support a behavioral steering claim. The
best small Qwen 0.5B boundary-prior setting reaches 0.750 positive-vs-negative
cohesion success across six prompts, but its mean positive-minus-negative score
delta is approximately zero. Stronger final-layer steering, all-position
steering, layer -2 steering, a Qwen 1.5B layer -2 run, and a broader
fault-class direction all remain weak or mixed. This is a useful negative
result: the directions that separate controlled prompt activations are not
automatically reliable generation-time controls under naive activation
addition.

The follow-up sweep makes the failure mode more precise. Small generated-token
post-hook edits reach a 0.750 prompt-level positive-vs-negative win rate, but
the mean text-score shift is only +0.004. Stronger post/generate/last steering
at +/-4 separates positive from negative generated responses after re-embedding
them (+3.561 projection), while the local text score moves slightly negative
(-0.021) and positive steering remains below baseline projection. A dense
-6..+6 dose run confirms that this is not a clean symmetric control knob:
negative steering pushes generated outputs down the learned projection, but
positive steering does not lift them above baseline and behavior remains flat.

Hidden-state telemetry localizes the bottleneck. At layers -1, -2, and -4, the
hook applies the requested signed displacement almost exactly during greedy
generation: mean absolute delta error is 0.0073, 0.0018, and 0.0025
respectively. Post-hook projection shifts by roughly +11 along the learned
direction from negative to positive steering, but the short 24-token generated
text score only moves +0.015 to +0.024. The current direction is therefore a
reliable hidden-state displacement direction, not yet a reliable semantic
control direction.

The affect-control direction gives the same lesson under a stricter confound
control. A Qwen 0.5B layer -1 direction trained after projecting out the
five-dimensional affect-label subspace remains orthogonal to the affect basis
and keeps a +8.427 in-sample mean margin across the 72 affect-control pairs. On
the six held-out steering prompts, raw affect-control steering reaches 0.500
positive-vs-negative cohesion success with a -0.005 mean score delta, while the
affect-residualized direction reaches 0.583 with a +0.007 delta. Hidden
telemetry confirms accurate vector injection (0.00233 mean delta error) and a
+3.91 post-hook projection shift from negative to positive steering, but the
short text-score shift is only +0.019. That is a small control improvement, not
a behavioral steering result.

The methodological consequence is important. The next NeurIPS-relevant
experiment is not simply a larger probe benchmark. It is a steering-method
sweep: residual-stream hook sites, generated-token-only steering, strength
schedules, hidden-state telemetry, projection checks on generated outputs,
anti-compliance regressions, and stronger pairwise evaluators. A publishable
causal claim should require the hook-level displacement, generated-output
projection, and positive-vs-negative behavioral shift to move together while
preserving refusal, privacy, truth, dissent, and exit rights.

### 7.10 Affect-Control Residualization

Status: complete for first local text-control export, Modal activation
residualization, and an affect-residualized steering-control smoke; pending
generated paraphrase hardening and any real EEG or human validation.

The NOVA emotion-decoding post is useful here as a methodological warning, not
as evidence for this project. NOVA emphasizes careful stimulus curation,
session-wise evaluation, PSD-vs-DE feature choices, and simple ridge models
before more complex neural decoders. The immediate analog for this repo is an
affect-control lane: before claiming that a social-cohesion direction is about
agency-preserving relation, test whether it is merely tracking coarse affect
such as anger, sadness, fear, disgust, happiness, or neutral procedural style.

The first local export crosses the cue-balanced boundary-prior contrasts with
six affect frames: anger, sadness, fear, disgust, happy, and neutral. This
produces 72 matched pairs / 144 activation prompts across the same six
boundary-prior mechanisms and two negative poles. The local scorer prefers the
contextual-relation side on 72/72 pairs, with a +0.122 mean score margin. A
simple affect-only ridge classifier reaches 0.750 pairwise leave-one-out
accuracy, which means affect/style proxies are not irrelevant. However,
leave-one-pair-out residualization of local cohesion scores against the same
coarse affect features preserves 1.000 pairwise accuracy, with a +0.116 mean
residualized margin and a +0.017 minimum residualized margin.

This is useful but narrow. It is evidence against the weakest objection that
the current scorer is only choosing lower-threat or more positive affect. It is
not evidence about human emotion, EEG, or neural cohesion. The next activation
version should extract open-model activations for
`data/training/affect_control_activation_prompts.jsonl`, train affect directions
and cohesion directions in the same model/layer files, and report whether the
cohesion direction survives projection or regression against the affect
directions.

That activation pass is now complete for Qwen 0.5B and 1.5B at layers -1, -2,
and -4. Raw cohesion directions reach 1.000 leave-one-pair-out accuracy on all
six model/layer points. The affect-residualized protocol learns a
five-dimensional affect-label subspace from the training folds only, projects
that subspace out of train and held-out activations, retrains the cohesion
direction on residualized training activations, and evaluates the held-out pair.
The residualized direction still reaches 1.000 leave-one-pair-out accuracy on
all six model/layer points. Mean residualized margins remain strongly positive:
+8.175, +1.876, +1.596, +4.836, +7.703, and +6.570 across Qwen 0.5B layers
-1/-2/-4 and Qwen 1.5B layers -1/-2/-4 respectively. Minimum residualized
margins are also positive, ranging from +0.926 to +4.499.

The interpretation is narrow but useful: coarse affect directions are present
and reduce margins, yet they do not explain away the current boundary-prior
cohesion separation in open-model activations. A later EEG pilot, if the
behavioral signal becomes worth testing, should follow the NOVA lesson
directly: PSD and differential-entropy baselines, subject/session splits, simple
ridge first, and explicit separation of affect decoding from social-cohesion
claims.

We also trained a steering-ready Qwen 0.5B layer -1 vector in the residualized
activation space and saved it in the original coordinate basis for the Modal
generation hook. The saved vector is orthogonal to the affect basis and keeps
1.000 in-sample pairwise accuracy on the 72 affect-control pairs, with a +8.427
mean margin and +4.433 minimum margin. Its first generation-time smoke improves
slightly over the raw affect-control direction but remains weak, reinforcing the
main causal bottleneck: removing affect confounds helps the control story, but
does not yet make activation addition a reliable semantic intervention. The
telemetry pass is nevertheless useful because it localizes the remaining failure
downstream of hook application.

## 8. Ethics And Safety

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

The *Magnifica Humanitas* frame sharpens this safety boundary. The project must
not optimize connection, persuasion, peace language, or apparent harmony when
those gains come from dependence, social control, surveillance, labor
devaluation, algorithmic opacity, or the outsourcing of responsibility. Truth,
work, freedom, care, peace, and human responsibility are not decorative values
around the metric; they are criteria for deciding whether a metric has learned
the right construct at all.

The boundary-prior frame sharpens the same point from a computational direction.
Healthy cohesion does not require dissolving self-protection, group difference,
or principled non-agreement. It requires treating boundaries as pragmatic,
contextual, revisable, and consent-sensitive rather than ontologically fixed or
coercively erased. The model should therefore flag both failure modes: rigid
partitioning that turns others into threats, and unity language that removes a
person's right to refuse, verify, exit, or dissent.

## 9. Limitations

The current results are synthetic and circular: pair labels are generated from
the same rubric used by the full-scorer baseline. Scripted trajectories include
surface lexical cues that make the task too easy. The benchmark does not yet
show that activation directions generalize, steer outputs, or predict human
behavior. It is a scaffold for those tests.

The activation-vector interpretation also has geometry limits. A direction is a
signed object, while an axis is sign-invariant; anti-aligned directions may be
the same axis with different poles. Signed off-diagonal cosine means can hide
large absolute cosine structure, so independence claims require the whole
distribution, not just the average. Squared projection energy erases sign and
therefore cannot by itself localize whether a feature supports the genuine or
pseudo-cohesion pole. Finally, a one-direction residual ablation is not proof
that the original direction is uniquely causal or that the representation has
been exhausted; multi-direction, group-specific, and subspace diagnostics are
needed.

## 10. Next Experiments

1. Generate external-model trajectories with reduced template leakage and score
   them through the generated benchmark. Status: pending.
2. Generate LLM-authored pseudo-cohesion examples by symbolic fault class and
   evaluate fault-held-out transfer. Status: pending.
3. Extend the hard-negative suite beyond deterministic rewrites and use current
   scorer/SAE artifacts to revise the dataset. Status: partial.
4. Build transfer splits: scripted-to-generated, generated-to-scripted,
   intervention-held-out, hard-negative-held-out, and fault-held-out. Status:
   partial for scripted/generated pair-set transfer and expanded/clean SAE
   feature transfer.
5. Repeat the generated activation sweep on a larger model and public Gemma
   Scope-style SAE dictionaries if compute allows. Status: pending.
6. Add social-dynamics tasks for direct reciprocity, indirect reciprocity,
   reputation sensitivity, passive-compliance vulnerability, public goods, and
   long-horizon cooperation. Status: pending.
7. Add persuasion-risk and narrative-reliance checks before any human-facing
   experiments. Status: pending.
8. Expand the persona-vector-style trait families:
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
9. Test monitoring before output and steering/composition only after transfer
   splits show non-circular signal. Status: pending.
10. Add direction-geometry, residual-subspace, and signed-vs-squared subspace
   reports to every activation result: signed and absolute cosines,
   anti-aligned directions, residual energy, signed subspace voting, squared
   subspace energy, and fault-specific residual directions. Status: partial.
11. Preserve signed projections when inspecting SAE or ROI-style localization;
   squared projection can be reported as energy but not as pole direction.
   Status: partial; Qwen 1.5B layer -2 now shows 1.000 signed subspace accuracy
   but only 0.750 squared-energy accuracy on the autonomy stress set.
12. Expand the structural autonomy stress suite around the Qwen LOO failures:
   dialogue-style verification/proof and silence-as-consent. Status: partial.
13. Replace the controlled cue-balanced boundary-prior expansion with
   generated/API-authored paraphrases, then rerun leakage, activation, geometry,
   residual, and signed-vs-squared reports. Status: partial for deterministic
   cue-balanced export, controlled 36-pair expansion, and Qwen 0.5B/1.5B smoke.
14. Prepare a Prolific pairwise validation pilot only after generated-text and
   hard-negative validation. Status: pending.

## References

- Allport, G. W. (1954). *The Nature of Prejudice*.
- Anthropic. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."
- Anthropic. (2025). "Persona vectors: Monitoring and controlling character traits in language models."
- Aristotle. *Nicomachean Ethics*.
- Axelrod, R. (1984). *The Evolution of Cooperation*.
- Arditi, A., et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717. https://arxiv.org/abs/2406.11717
- Ashkinaze, J., Shen, H., Avula, S., Gilbert, E., & Budak, C. (2025). "Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences." arXiv:2511.02109. https://arxiv.org/abs/2511.02109
- Blas, L., Jia, R., & Ferrara, E. (2026). "Psychological Steering of Large Language Models." arXiv:2604.14463. https://arxiv.org/abs/2604.14463
- Burger, L., Hamprecht, F. A., & Nadler, B. (2024). "Truth is Universal: Robust Detection of Lies in LLMs." arXiv:2407.12831. https://arxiv.org/abs/2407.12831
- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." arXiv:2507.21509. https://arxiv.org/abs/2507.21509
- Cirulli, D., Cimini, G., & Palermo, G. (2025). "How Large Language Models play humans in online conversations: a simulated study of the 2016 US politics on Reddit." arXiv:2506.21620. https://arxiv.org/abs/2506.21620
- Dang, T. D. A., & Masud, S. (2026). "Cultural Value Alignment Via Latent Activation Steering in Large Language Models." arXiv:2605.26365. https://arxiv.org/abs/2605.26365
- Gao, L., et al. (2024). "Scaling and evaluating sparse autoencoders." arXiv:2406.04093. https://arxiv.org/abs/2406.04093
- Greenblatt, R., et al. (2024). "Alignment faking in large language models." arXiv:2412.14093. https://arxiv.org/abs/2412.14093
- Grossmann, G., Ivanova, L., Poduru, S. L., Tabrizian, M., Mesabah, I., et al. (2025). "The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete." arXiv:2505.03961. https://arxiv.org/abs/2505.03961
- Guo, D., Wu, J., & Yiu, S. M. (2026). "Sparse Autoencoders Map Brain-LLM Alignment onto Cortical Semantic Topography." arXiv:2605.23035. https://arxiv.org/abs/2605.23035
- Haerle, R., et al. (2024). "SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs." arXiv:2411.07122. https://arxiv.org/abs/2411.07122
- Huang, X. A., et al. (2026). "Mechanism Design Is Not Enough: Prosocial Agents for Cooperative AI." arXiv:2605.08426. https://arxiv.org/abs/2605.08426
- Jung, I., Oh, Y., Back, K., Kim, J., & Lee, J. (2026). "SODE: Analyzing Social Dynamics in LLM Agents." arXiv:2605.23949. https://arxiv.org/abs/2605.23949
- Kim, W., Hyeon, S., Oh, J., & Do, J. (2026). "VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models." arXiv:2602.03160. https://arxiv.org/abs/2602.03160
- Lan, M., et al. (2024). "Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders." arXiv:2410.06981. https://arxiv.org/abs/2410.06981
- Leo XIV. (2026). *Magnifica Humanitas: On Safeguarding the Human Person in the Time of Artificial Intelligence*. Vatican. https://www.vatican.va/content/leo-xiv/en/encyclicals/documents/20260515-magnifica-humanitas.html
- Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." arXiv:2408.05147. https://arxiv.org/abs/2408.05147
- Marusich, L. R., Kozuch Dhooghe, M. G., Bakdash, J. Z., & Kantarcioglu, M. (2026). "Human Decision-Making with Persuasive and Narrative LLM Explanations." arXiv:2605.23867. https://arxiv.org/abs/2605.23867
- Moskvoretskii, V., Glandorf, D., Medina Moreira, J., Kaeser, T., & West, R. (2026). "Tracing Persona Vectors Through LLM Pretraining." arXiv:2605.13329. https://arxiv.org/abs/2605.13329
- Nowak, M. A. (2006). "Five Rules for the Evolution of Cooperation." *Science*.
- O'Brien, K., et al. (2024). "Steering Language Model Refusal with Sparse Autoencoders." arXiv:2411.11296. https://arxiv.org/abs/2411.11296
- O'Brien, C., Seto, J., Roy, D., Dwivedi, A., Dev, S., et al. (2026). "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy." arXiv:2601.18939. https://arxiv.org/abs/2601.18939
- Ostrom, E. (1990). *Governing the Commons*.
- Park, J. S., Zou, C. Q., Kamphorst, J., Egan, N., Shaw, A., et al. (2024). "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals." arXiv:2411.10109. https://arxiv.org/abs/2411.10109
- Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., et al. (2024). "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders." arXiv:2407.14435. https://arxiv.org/abs/2407.14435
- Rilling, J. K., et al. (2002). "A neural basis for social cooperation." *Neuron*.
- Sandved-Smith, L., Fields, C., Doctor, T., Laukkonen, R. E., & Hohwy, J. (2026). "There is no self-evidence: A physics of emptiness realisation." Preprint. https://doi.org/10.31234/osf.io/m78z2_v1
- Schoenegger, P., Salvi, F., Liu, J., Nan, X., Debnath, R., et al. (2025). "When Large Language Models are More Persuasive Than Incentivized Humans, and Why." arXiv:2505.09662. https://arxiv.org/abs/2505.09662
- Shoresh, D., Kraus, S., & Loewenstein, Y. (2026). "Communicate-Predict-Act: Evaluating Social Intelligence of Agents." arXiv:2604.08727. https://arxiv.org/abs/2604.08727
- Shou, Y., & Guan, M. (2026). "Mechanistic Decoding of Cognitive Constructs in Large Language Models." arXiv:2604.14593. https://arxiv.org/abs/2604.14593
- Shu, D., et al. (2025). "A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models." arXiv:2503.05613. https://arxiv.org/abs/2503.05613
- Tajfel, H., & Turner, J. C. (1979/1986). Social identity theory of intergroup behavior.
- Wa Nkongolo, M. (2026). "Pluralism in AI Governance: Toward Sociotechnical Alignment and Normative Coherence." arXiv:2602.15881. https://arxiv.org/abs/2602.15881
- Yang, J., et al. (2025). "LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency in Large Language Models." arXiv:2501.11036. https://arxiv.org/abs/2501.11036
- Zhang, X., Wang, J., Zhao, Q., Guo, H., Li, L., et al. (2026). "Human Values Matter: Investigating How Misalignment Shapes Collective Behaviors in LLM Agent Communities." arXiv:2604.05339. https://arxiv.org/abs/2604.05339
- Zhang, S., et al. (2026). "Understanding the Mechanism of Altruism in Large Language Models." arXiv:2604.19260. https://arxiv.org/abs/2604.19260
