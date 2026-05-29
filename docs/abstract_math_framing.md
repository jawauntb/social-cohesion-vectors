# Abstract Mathematical Framing

This note gives the clean math version of the project, separated from the
implementation details.

## Core Object

Let `X` be a space of social situations, trajectories, messages, or generated
interventions. Let a model layer define a representation map:

```text
phi_l : X -> V_l
```

where `V_l` is a finite-dimensional activation space. The project is not trying
to find a single scalar function called cohesion. It is trying to learn a family
of signed functionals, sparse features, and low-dimensional subspaces:

```text
F = {f_1, ..., f_m},       f_i : V_l -> R
U_i in Gr(k_i, V_l)
```

that track distinct social-reasoning mechanisms such as repair, reciprocity,
truthfulness, autonomy safety, constructive dissent, coercion, sycophancy,
dehumanization, and pseudo-cohesion.

In the simplest linear case, each signed contrast is a covector:

```text
w_i = mean(phi_l(x_i^+)) - mean(phi_l(x_i^-))
f_i(x) = <normalize(phi_l(x)), normalize(w_i)>
```

where each pair `(x_i^+, x_i^-)` is matched for context but differs in one
social-reasoning property. In the sparse-feature case, `f_i` is a signed sparse
mixture over dictionary features.

## The Real Target

The target is a constrained multi-objective problem:

```text
maximize    repair + reciprocity + fairness + calibrated_trust
subject to  truthfulness >= tau_truth
            autonomy_safety >= tau_autonomy
            dissent_safety >= tau_dissent
            privacy >= tau_privacy
            exit_rights >= tau_exit
            manipulation_risk <= tau_manipulation
```

So the mathematical object is closer to a feasible region and Pareto frontier
than to one reward scalar. A candidate intervention is good only if it moves in
the desired cooperative directions while staying inside the guardrail
constraints.

## Why Sign Matters

A direction is signed. An axis is not.

```text
w and -w are the same unsigned axis, but opposite social poles.
```

Squared projection,

```text
||Proj_U(phi_l(x))||^2
```

measures energy in a subspace, but it erases whether the point is on the
agency-preserving or coercive pole. The current Qwen 1.5B autonomy result shows
this empirically: signed subspace voting reaches 1.000 pair-LOO accuracy, while
squared-energy accuracy reaches only 0.750. Therefore localization reports must
preserve signed projections alongside energy.

## Subspace Rather Than Scalar

For paired examples, define pair differences:

```text
Delta_i = phi_l(x_i^+) - phi_l(x_i^-)
```

The matrix `Delta` can be studied with SVD, residual projections, principal
angles, and group-specific directions. The question is:

```text
How much pair-difference energy is captured by one global direction,
and how much remains in mechanism-specific residual subspaces?
```

This is why the project now reports:

- signed and absolute cosine matrices;
- anti-aligned direction counts;
- residual pair-difference energy after projecting out a global direction;
- group-specific residual separation;
- signed-vs-squared k-component subspace probes.

The early result is that social cohesion is better modeled as one or more shared
manifolds plus mechanism-specific residual structure, not as independent
orthogonal axes.

## Boundary Priors

The Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy paper is relevant as a
theory of boundary priors. In abstract terms, an agent uses a latent
factorization or partition:

```text
sigma : admissible frames -> frames preserving a self/environment partition
```

The useful translation for this project is not the quantum claim itself. It is
the idea that rigid self/other or group/outgroup partitions can be modeled as
structural priors over admissible frames. Healthy cohesion should not erase
boundaries. It should make boundary designation flexible, contextual,
truth-sensitive, consent-sensitive, and revisable.

This gives a new benchmark family:

```text
fixed boundary reification
vs flexible boundary/contextual relation
vs coercive boundary collapse
```

The third category is important. "We are all one, so your refusal is betrayal"
is not good cohesion; it is pseudo-cohesion via boundary collapse.

The first implementation treats this as a paired contrast problem. For each
mechanism `m`, construct:

```text
x_m^+  = contextual relation with consent, review, privacy, exit, and dissent
x_m^-r = rigid boundary reification
x_m^-c = coercive boundary collapse
```

Then learn signed differences:

```text
Delta_m^r = phi_l(x_m^+) - phi_l(x_m^-r)
Delta_m^c = phi_l(x_m^+) - phi_l(x_m^-c)
```

The key question is whether `Delta^r` and `Delta^c` share a global
agency-preserving direction, whether they form separate residual subspaces, and
whether that separation survives cue-balanced paraphrases. The local scaffold
currently exports 12 pairs / 24 prompts across 6 mechanisms. The first leakage
gate solved 5/12 pairs. A cue-balanced variant drives the simple cue margin to
zero while Qwen 0.5B and 1.5B activation directions still separate the pairs.
The next mathematical requirement is stronger invariance under generated
paraphrase and domain shift, not just hand-authored cue balancing.

## Neighboring Mathematical Fields

This project touches several established mathematical areas:

- **Representation learning and linear probing:** learning signed directions in
  high-dimensional activation spaces.
- **Sparse dictionary learning and compressed sensing:** replacing dense
  directions with sparse feature mixtures.
- **Contrastive learning and metric learning:** constructing matched positive
  and negative pairs that isolate a mechanism.
- **Grassmannian geometry:** treating learned mechanisms as subspaces and
  comparing them by principal angles, projection energy, and residual structure.
- **Multi-objective and constrained optimization:** optimizing cooperative
  movement under truth, autonomy, privacy, dissent, and manipulation
  constraints.
- **Game theory and evolutionary dynamics:** modeling cooperation, reciprocity,
  punishment, reputation, public goods, and long-horizon stability.
- **Mechanism design and social choice:** asking which rules or messages
  preserve agency while improving group outcomes.
- **Causal inference and invariant learning:** checking whether learned
  directions survive interventions, domains, generators, and held-out fault
  classes.
- **Information geometry and variational inference:** interpreting agents as
  updating generative models under uncertainty and constraints.
- **Graph theory and network science:** moving from dyadic interactions to
  groups, coalitions, inter-brain synchrony, and social networks.
- **Sheaf/contextuality-style mathematics:** asking when local evaluations
  across contexts can be glued into a consistent global object, and when
  apparent agreement hides incompatible local constraints.
- **Statistical learning theory:** bounding generalization from hand-authored
  contrasts to generated text and human outcomes.

## Similar Abstract Problems

The project resembles:

- learning a toxicity classifier that does not confuse identity terms with harm;
- learning a truthfulness direction that does not become a style detector;
- learning a refusal direction without over-refusing benign requests;
- learning fairness constraints that survive distribution shift;
- learning value functions in inverse reinforcement learning when rewards are
  multi-dimensional and culturally contested;
- learning causal features rather than shortcut cues in out-of-distribution
  classification;
- learning local coordinate charts on a manifold when no single global coordinate
  captures the phenomenon.

The project's specific twist is that the target construct is relational and
normative. It only makes sense under constraints: cooperation without
truthfulness, autonomy, dissent, and exit rights is not the same construct.
