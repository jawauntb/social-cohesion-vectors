# HILO Primer: Social Cohesion Vectors

## Chapter Thesis

This project asks whether social cohesion can be operationalized as a family of
measurable, interpretable directions in model behavior, content, and eventually
human response. The word "cohesion" is dangerous if it means conformity,
obedience, or the suppression of dissent. The useful target is narrower and more
defensible: agency-preserving movement away from corrosive zero-sum escalation
and toward truthful cooperation, repair, calibrated trust, fairness, and
continued dialogue under conflict.

The high-level claim is philosophical: human flourishing is partly social. The
low-level claim is experimental: repeated dilemmas, public-goods games,
negotiations, and repair conversations produce trajectories that can be scored,
contrasted, embedded, and tested for causal steering in open language models.

## HILO Frame

**HI: What is the human idea?**

Social cohesion is the capacity of persons or groups to stay in relation while
pursuing legitimate interests, telling the truth, repairing harm, and acting
toward shared goods where shared goods exist. It is not universal agreement. It
is not niceness. It is not compliance with an authority. A cohesive interaction
can still contain refusal, anger, moral dissent, bargaining, punishment, and
boundaries, provided those moves remain proportionate, truth-oriented, and
compatible with the other party's agency.

**LO: What can be measured now?**

In this repo, cohesion is approximated with scenario-level trajectories and
subscores:

- cooperation rate;
- defection rate;
- repair attempt rate;
- hostility rate;
- fairness score;
- joint payoff;
- truthfulness;
- autonomy safety.

The first operational unit is a pairwise example: one trajectory that scores
higher on the current cohesion rubric and one trajectory that scores lower
within the same scenario. That pair can feed baselines, probes, reward models,
activation capture, sparse autoencoder analysis, and eventually human validation.

## Aristotle: Flourishing, Character, And The Political Frame

Aristotle's *Nicomachean Ethics* begins from the idea that action aims at goods,
and that the highest human good is not a passing preference but a life of
flourishing activity. The ethical inquiry is linked to politics because the good
of a person and the good of a community are not separable in practice. A person
learns virtue through habituation, relationships, institutions, and shared
practical reasoning. The Stanford Encyclopedia discussion emphasizes that the
*Nicomachean Ethics* is uniquely explicit about the relationship between ethical
inquiry and politics, and that Aristotle's ethics concerns the kind of life in
which human capacities are well ordered.

For this project, Aristotle contributes three guardrails:

1. Social good is not reducible to individual utility.
2. Character matters: the disposition to act cooperatively, truthfully, and
   fairly is part of the target, not only the final allocation.
3. Friendship and political association matter because human goods are pursued
   through relation, not only through private optimization.

The computational translation should be modest. We are not encoding virtue
itself. We are building measurable proxies for relational behavior: whether an
agent repairs, reciprocates, listens, updates, and preserves another agent's
standing as a chooser.

## Axelrod: Cooperation Under Repetition

Robert Axelrod's *The Evolution of Cooperation* is the project’s cleanest game
theoretic ancestor. Axelrod studied the iterated Prisoner's Dilemma and showed
how cooperation can become stable among self-interested agents when interactions
repeat, actions are legible, and strategies can condition on prior behavior.
The canonical lesson is not "always cooperate." It is that successful strategies
often combine niceness, provocability, forgiveness, and clarity.

The low-level implication is direct. A one-shot scoring task misses the social
dynamics that matter most. Cohesion is often revealed only across rounds:

- Does a cooperative opening survive exploitation?
- Does retaliation remain bounded?
- Does apology restore cooperation?
- Does reciprocity become stable?
- Does truth-telling prevent false peace?
- Does the interaction preserve enough trust to continue?

The current simulation benchmark begins there: repeated games, public goods,
resource allocation, dialogue repair, and negotiation scenarios are run under
cooperative, self-protective, and adversarial profiles with interventions such as
shared identity, perspective-taking, reciprocity, restorative accountability, and
truth-first framing.

## Game Theory: Dilemmas, Bargaining, Public Goods, And Repair

The basic game-theory primitives are useful but incomplete.

**Prisoner's Dilemma:** Individual defection dominates in the one-shot game, but
mutual cooperation is collectively better. Repetition opens the door to
reciprocity and reputation.

**Public Goods:** Individuals can free ride on contributions made by others.
The important variable is not only contribution rate but institutional design:
monitoring, graduated sanctions, shared decision rules, and legitimate
boundaries.

**Bargaining And Negotiation:** Conflict is not always a failure. Parties can
have legitimate incompatible preferences. Cohesion means truthful, fair,
agency-preserving movement toward an agreement or a principled non-agreement.

**Repair Games:** Many real conflicts begin after harm has occurred. A repair
model needs state variables absent from standard payoff matrices: apology,
accountability, restitution, trust recovery, and procedural legitimacy.

**Repeated And Networked Games:** Martin Nowak’s review of cooperation
mechanisms distinguishes direct reciprocity, indirect reciprocity, network
reciprocity, kin selection, and group selection. For this project, direct and
indirect reciprocity are immediately testable in text simulations; network
reciprocity becomes relevant when moving from dyads to groups.

## Ostrom: Cooperation Is Designed, Not Merely Wished For

Elinor Ostrom's work on commons governance matters because it moves beyond the
abstract tragedy-of-the-commons story. Groups can govern shared resources when
rules are legitimate, boundaries are clear, monitoring is possible, sanctions
are proportional, and participants can help revise rules. This is a major design
lesson for AI-mediated cohesion. A message that merely says "be nice" is weak.
A message that clarifies roles, shared constraints, monitoring, appeal, and fair
revision may change the actual game.

Low-level operationalization:

- Does the intervention clarify the shared resource?
- Does it preserve local voice?
- Does it define fair contribution?
- Does it include proportionate accountability?
- Does it prevent both free-riding and domination?

## Social Identity, Contact, And Intergroup Conflict

Social cohesion cannot be studied only as rational payoff maximization. Tajfel
and Turner’s social identity tradition shows that group membership changes
perception, evaluation, and allocation. Minimal group findings suggest that even
thin group boundaries can produce favoritism. Allport’s contact hypothesis
suggests that intergroup contact is most helpful under structured conditions
such as equal status, common goals, cooperation, and institutional support.

For this project, group identity is both a target and a hazard.

Useful cohesion can involve:

- expanding a shared identity without erasing subgroup interests;
- making outgroup agency and suffering more legible;
- reducing dehumanization;
- creating common tasks with equal-status participation.

Dangerous pseudo-cohesion can involve:

- pressuring minority groups to assimilate;
- hiding real conflict under unity language;
- using "common good" rhetoric to silence dissent;
- optimizing agreement rather than justice.

The benchmark should therefore include adversarial cases where "cohesion" words
are used manipulatively. A good scorer must distinguish repair from compliance.

## Boundary Priors, Emptiness, And Pseudo-Cohesion

Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy offer a recent formal
account of "separation" as a structural prior over an agent's admissible
reference frames. Their quantum/free-energy claims should not be treated as
evidence for this repo's activation results, but the computational metaphor is
useful. Many social failures are boundary failures:

- rigid self/other or us/them partitions;
- over-protective threat framing;
- refusal to update a group boundary under new evidence;
- coercive unity language that treats refusal as betrayal;
- "we are all one" rhetoric that erases consent, exit, or dissent.

The healthy target is not boundarylessness. It is flexible, contextual boundary
use. People and groups need real boundaries for consent, safety, responsibility,
privacy, and principled non-agreement. A cohesion metric should therefore reward
boundary flexibility and relational concern while penalizing both rigid
reification and coercive boundary collapse.

## Neuroscience: Cooperation, Reward, Mentalizing, And Synchrony

Rilling et al. used fMRI during an iterated Prisoner's Dilemma and found mutual
cooperation associated with reward-related regions, including nucleus accumbens,
caudate, ventromedial/orbitofrontal cortex, and rostral anterior cingulate. This
does not mean there is a single brain "cohesion vector." It suggests that
cooperative social behavior recruits measurable systems related to reward,
valuation, and social decision-making.

Hyperscanning work on EEG, fNIRS, and fMRI adds a group-level possibility:
coordination may appear not only inside individual brains but in inter-brain
synchrony during live interaction. That is future work. The near-term bridge is
to compare content and model-derived vectors against brain-aligned proxy models,
while clearly marking those as proxies.

## Mechanistic Interpretability And Persona Vectors

Modern interpretability gives the project its most practical near-term method.
Sparse autoencoders attempt to recover interpretable features from superposed
model activations. Anthropic's monosemanticity work argues that sparse
autoencoders can produce feature dictionaries more interpretable than individual
neurons. Anthropic's persona-vector work is even closer: it identifies
activation-space directions corresponding to model character traits such as
sycophancy, hallucination propensity, or harmful personas, then uses those
directions to monitor and steer behavior.

The direct adaptation is:

1. Define a trait or social mode in natural language.
2. Generate contrastive prompts or trajectories where the trait is present vs
   absent.
3. Capture model activations at chosen layers/tokens.
4. Compute a direction or train an SAE/probe.
5. Test whether projection predicts behavior before output.
6. Intervene cautiously and measure downstream effects.

For social cohesion, the right object is not one vector but a vector family:

- reciprocity vector;
- repair/accountability vector;
- perspective-taking vector;
- fairness vector;
- truthfulness vector;
- autonomy-preserving vector;
- dehumanization vector;
- domination/coercion vector;
- sycophancy/compliance vector;
- punitive escalation vector.
- rigid-boundary vector;
- coercive-boundary-collapse vector;
- flexible-contextual-relation vector.

The scientific question is whether useful combinations of these directions
predict or steer outputs toward cooperative repair without increasing
manipulation, false agreement, or truth loss.

## Operational Definitions

**Social cohesion trajectory:** A sequence of messages and actions in which
agents preserve or recover the possibility of fair, truthful, voluntary
cooperation.

**Cohesion-promoting content:** Content that increases predicted or observed
repair, cooperation, trust calibration, fairness, or willingness to continue
dialogue while preserving truthfulness and autonomy.

**Self-protective behavior:** Boundary-setting or refusal that protects a
legitimate interest. This is not negative by default.

**Corrosive selfishness:** Opportunistic defection, domination, deception,
dehumanization, or refusal to acknowledge legitimate interests of others.

**Pseudo-cohesion:** Apparent agreement produced by coercion, flattery,
manipulation, fatigue, hidden threat, or suppression of dissent.

## Testable Hypotheses

H1. Scripted cooperative and adversarial trajectories are separable by simple
metrics and lexical baselines. This is a sanity check, not a scientific win.

H2. LLM-generated held-out trajectories will be harder: surface lexicons will
drop, but activation-derived directions will retain some predictive power.

H3. Cohesion is multi-dimensional, but this has to be shown rather than assumed.
Repair, reciprocity, truthfulness, and autonomy-safety directions should be
tested with signed and absolute cosine distributions, anti-alignment checks, and
residual subspace audits. A near-zero mean signed cosine is not enough to claim
orthogonality, and a failed one-direction residual ablation is not enough to
claim the signal has been exhausted.

H4. Steering toward repair while steering away from coercion will produce better
outputs than steering toward generic positivity.

H5. Human validators will prefer high-scoring interventions only when
truthfulness and autonomy guardrails are included.

H6. Brain-aligned proxy models may distinguish cooperative-repair content from
punitive or dehumanizing content, but this will remain hypothesis-generating
until behavioral or neural validation.

H7. Boundary-prior contrasts will expose a distinct pseudo-cohesion failure
mode: language that sounds compassionate or nondual but uses "oneness" to
remove a person's right to refuse, verify, exit, or dissent.

## Ethics Boundaries

The project must optimize for agency-preserving cohesion, not compliance.

Hard constraints:

- no deception outside approved research protocols;
- no suppression of legitimate dissent;
- no treating self-protection as pathology;
- no optimizing persuasion while ignoring truth;
- no demographic or political homogenization objective;
- no deployment claims before human validation;
- explicit manipulation-risk scoring in every content-selection loop.

The positive target is not "make people agree." It is "help people deliberate,
repair, cooperate where possible, and disagree without destroying the social
conditions for future truth-seeking."

## Research Program

1. Build scenarios and scripted baselines.
2. Generate LLM trajectories under varied conflict profiles.
3. Score trajectories with transparent rubrics.
4. Capture open-model activations on Modal.
5. Train contrastive vectors and SAE features.
6. Test vector composition and steering.
7. Validate selected interventions on Prolific.
8. Compare against brain-aligned proxy models.
9. Only then consider new neural data.

## References

- Allport, G. W. (1954). *The Nature of Prejudice*.
- Anthropic. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Anthropic. (2025). "Persona vectors: Monitoring and controlling character traits in language models." https://www.anthropic.com/research/persona-vectors
- Aristotle. *Nicomachean Ethics*. See overview: https://plato.stanford.edu/entries/aristotle-ethics/
- Axelrod, R. (1984). *The Evolution of Cooperation*.
- Axelrod, R., & Hamilton, W. D. (1981). "The Evolution of Cooperation." *Science*.
- Nowak, M. A. (2006). "Five Rules for the Evolution of Cooperation." *Science*. https://pubmed.ncbi.nlm.nih.gov/17158317/
- Ostrom, E. (1990). *Governing the Commons*.
- Rilling, J. K., et al. (2002). "A neural basis for social cooperation." *Neuron*. https://pubmed.ncbi.nlm.nih.gov/12160756/
- Sandved-Smith, L., Fields, C., Doctor, T., Laukkonen, R. E., & Hohwy, J. (2026). "There is no self-evidence: A physics of emptiness realisation." Preprint. https://doi.org/10.31234/osf.io/m78z2_v1
- Tajfel, H., & Turner, J. C. (1979/1986). Social identity theory of intergroup behavior.
