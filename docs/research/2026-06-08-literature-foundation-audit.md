---
title: 2026-06-08 Literature Foundation Audit
status: accepted
date: 2026-06-08
origin: user request for deep research across the project's active research areas
---

# 2026-06-08 Literature Foundation Audit

## Purpose

This note maps the current project into research areas, records high-value
papers found on arXiv and elsewhere, and translates them into plan changes. It
does not add new empirical evidence for this repo. It changes what we should
test, which gates should matter, and which claims should stay blocked until the
right validation exists.

## Search Segments

The project currently touches eight research areas:

1. Activation steering, representation engineering, sparse autoencoders, and
   persona-style trait directions.
2. Benchmark artifacts, shortcut learning, invariant generalization, and
   hard-negative construction.
3. Cooperative AI, LLM-agent social dilemmas, and value-conditioned collective
   behavior.
4. Social identity, intergroup contact, common identity, outgroup bias, and
   pseudo-cohesion.
5. Procedural justice, effective voice, exit rights, psychological safety, and
   repair.
6. Autonomy-supportive influence, persuasion risk, narrative reliance, and
   reactance.
7. Human behavioral and neural validation: cooperation neuroscience,
   hyperscanning, and brain-LLM alignment.
8. Boundary-prior and active-inference metaphors for self/other and group
   boundary modeling.

High-yield next search queries:

- `activation steering representation engineering sparse autoencoder feature
  guided activation addition social traits`
- `LLM social dilemmas cooperation public goods prosocial agents mechanism
  design`
- `procedural justice voice exit cooperation trust authority psychological
  safety`
- `LLM persuasion narrative explanations human decision making reliance`
- `social identity bias large language models common ingroup identity contact
  hypothesis`
- `hyperscanning cooperation fNIRS EEG meta-analysis interbrain synchrony`
- `brain LLM alignment sparse autoencoder cortical semantic topography`
- `active inference Markov blanket social cognition boundary self other`

## Findings By Area

### 1. Steering And Representation Methods

Relevant sources:

- Zou et al. (2023), "Representation Engineering: A Top-Down Approach to AI
  Transparency." https://arxiv.org/abs/2310.01405
- Turner et al. (2023), "Steering Language Models With Activation
  Engineering." https://arxiv.org/abs/2308.10248
- Panickssery et al. (2023), "Steering Llama 2 via Contrastive Activation
  Addition." https://arxiv.org/abs/2312.06681
- Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single
  Direction." https://arxiv.org/abs/2406.11717
- O'Brien et al. (2024), "Steering Language Model Refusal with Sparse
  Autoencoders." https://arxiv.org/abs/2411.11296
- Soo et al. (2025), "Steering Large Language Models with Feature Guided
  Activation Additions." https://arxiv.org/abs/2501.09929
- Chen et al. (2025), "Persona Vectors: Monitoring and Controlling Character
  Traits in Language Models." https://arxiv.org/abs/2507.21509
- Blas, Jia, and Ferrara (2026), "Psychological Steering of Large Language
  Models." https://arxiv.org/abs/2604.14463
- Shou and Guan (2026), "Mechanistic Decoding of Cognitive Constructs in Large
  Language Models." https://arxiv.org/abs/2604.14593
- Zhang et al. (2026), "Understanding the Mechanism of Altruism in Large
  Language Models." https://arxiv.org/abs/2604.19260

What this should change:

- The current steering bottleneck is well-grounded. Prior work says activation
  additions can steer some domains, but also shows layer, site, scale, fluency,
  and side-effect tradeoffs. The repo should keep treating "projection moves but
  behavior does not" as a real mechanistic bottleneck, not as a failed run to
  ignore.
- Add semantically calibrated strength sweeps to the steering plan. The
  psychological-steering paper argues against relying only on arbitrary
  activation-space units; our monotonic steering protocol should record both raw
  coefficient and an output-calibrated or evaluator-calibrated effect size.
- Compare plain mean-difference steering against at least one feature-guided or
  SAE-assisted steering method before concluding that the cohesion direction is
  not steerable.
- Keep side-effect gates mandatory. Refusal and sycophancy papers show that
  localized steering can trade off against helpfulness, capabilities, or
  unrelated behaviors. For this project the side effects are pseudo-cohesion,
  compliance, hallucination, coercive boundary collapse, and loss of dissent.

Allowed claim level: activation result or causal steering result only after
behavior, generated-output projection, and side-effect gates move together.

### 2. Sparse Features And Cross-Model Interpretability

Relevant sources:

- Anthropic (2023), "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning."
  https://transformer-circuits.pub/2023/monosemantic-features/
- Gao et al. (2024), "Scaling and evaluating sparse autoencoders."
  https://arxiv.org/abs/2406.04093
- Lieberum et al. (2024), "Gemma Scope: Open Sparse Autoencoders Everywhere All
  At Once on Gemma 2." https://arxiv.org/abs/2408.05147
- Lan et al. (2024), "Sparse Autoencoders Reveal Universal Feature Spaces
  Across Large Language Models." https://arxiv.org/abs/2410.06981
- Rajamanoharan et al. (2024), "Jumping Ahead: Improving Reconstruction
  Fidelity with JumpReLU Sparse Autoencoders." https://arxiv.org/abs/2407.14435
- Guo, Wu, and Yiu (2026), "Sparse Autoencoders Map Brain-LLM Alignment onto
  Cortical Semantic Topography." https://arxiv.org/abs/2605.23035

What this should change:

- Do not name GPT-2 SAE features as "cohesion" features until they survive
  model/layer/hook compatibility checks and hard-negative transfer. Existing
  token-level demotions are exactly the right scientific posture.
- Public Gemma Scope dictionaries are a better next sparse-feature baseline
  than training custom SAEs immediately. They also make cross-model claims less
  dependent on the weak GPT-2 compatibility lane.
- The brain-LLM SAE paper is useful as a future bridge, not a claim upgrade. It
  suggests semantic SAE features can organize neural encoding, but this repo has
  not connected its social contrasts to human neural data.

Allowed claim level: sparse-feature inspection or brain-aligned proxy only;
never neural validation without human neural measurements.

### 3. Shortcut Learning And Hard Negatives

Relevant sources:

- Gururangan et al. (2018), "Annotation Artifacts in Natural Language Inference
  Data." https://arxiv.org/abs/1803.02324
- McCoy, Pavlick, and Linzen (2019), "Right for the Wrong Reasons: Diagnosing
  Syntactic Heuristics in Natural Language Inference."
  https://arxiv.org/abs/1902.01007
- Arjovsky et al. (2019), "Invariant Risk Minimization."
  https://arxiv.org/abs/1907.02893
- Geirhos et al. (2020), "Shortcut Learning in Deep Neural Networks."
  https://arxiv.org/abs/2004.07780
- Ashkinaze et al. (2025), "Deep Value Benchmark: Measuring Whether Models
  Generalize Deep Values or Shallow Preferences."
  https://arxiv.org/abs/2511.02109

What this should change:

- The project's lexical, length, source-diversity, slack, and availability gates
  are not extra bureaucracy. They are the core scientific defense against
  artifact learning.
- The generated hard-negative lane should borrow the Deep Value Benchmark's
  confound-break design: deliberately correlate a shallow cue with the target in
  one split, then reverse or decorrelate it in test. For this repo, shallow cues
  include warmth, institutional voice, length, "review/appeal/refusal" terms,
  and genre wrappers.
- A candidate dataset should not advance to activation extraction unless it
  passes an explicit environment split: source, setting, fault class, length bin,
  and lexical policy should be separable "environments" for held-out checks.

Allowed claim level: generated-text result or activation result only after
shortcut gates pass. Cue-balanced hand-authored data alone is not enough.

### 4. LLM Agents, Cooperation, And Collective Behavior

Relevant sources:

- Grossmann et al. (2025), "The Power of Stories: Narrative Priming Shapes How
  LLM Agents Collaborate and Compete." https://arxiv.org/abs/2505.03961
- Willis et al. (2026), "Evaluating Collective Behaviour of Hundreds of LLM
  Agents." https://arxiv.org/abs/2602.16662
- Tewolde et al. (2026), "CoopEval: Benchmarking Cooperation-Sustaining
  Mechanisms and LLM Agents in Social Dilemmas."
  https://arxiv.org/abs/2604.15267
- Huang et al. (2026), "Mechanism Design Is Not Enough: Prosocial Agents for
  Cooperative AI." https://arxiv.org/abs/2605.08426
- Zhang et al. (2026), "Human Values Matter: Investigating How Misalignment
  Shapes Collective Behaviors in LLM Agent Communities."
  https://arxiv.org/abs/2604.05339
- Shoresh, Kraus, and Loewenstein (2026), "Communicate-Predict-Act: Evaluating
  Social Intelligence of Agents." https://arxiv.org/abs/2604.08727

What this should change:

- Do not evaluate social cohesion only in one-shot messages. Add repeated-game,
  mediation, contract, reputation, and public-goods variants when the generated
  text gates are ready.
- Narrative priming is a double-edged variable. It can improve collaboration,
  but it can also become persuasion or shallow shared-story dependence. Any
  narrative intervention needs a truth/autonomy/pseudo-cohesion regression.
- "Mechanism design is not enough" supports the repo's current stance: external
  rules and availability paths matter, but the model's internal social
  orientation may still need monitoring or steering. It does not license a claim
  that a prosocial vector has been found.

Allowed claim level: simulated-agent result until human participants are tested.

### 5. Social Identity And Pseudo-Cohesion

Relevant sources:

- Pettigrew and Tropp (2006), "A Meta-Analytic Test of Intergroup Contact
  Theory." https://pubmed.ncbi.nlm.nih.gov/16737372/
- Gaertner and Dovidio (2000), "Reducing Intergroup Bias: The Common Ingroup
  Identity Model." https://openlibrary.org/books/OL33454022M/
- Hu et al. (2023/2024), "Generative Language Models Exhibit Social Identity
  Biases." https://arxiv.org/abs/2310.15819 and
  https://www.nature.com/articles/s43588-024-00741-1
- Dong et al. (2024), "Persona Setting Pitfall: Persistent Outgroup Biases in
  Large Language Models Arising from Social Identity Adoption."
  https://arxiv.org/abs/2409.03843
- Mendelsohn et al. (2020), "A Framework for the Computational Linguistic
  Analysis of Dehumanization." https://arxiv.org/abs/2003.03014
- Mehdizadeh and Hilbert (2025), "When Your AI Agent Succumbs to
  Peer-Pressure." https://arxiv.org/abs/2510.19107

What this should change:

- The pseudo-cohesion taxonomy should explicitly include common-identity
  failure modes: assimilation pressure, false consensus, outgroup flattening,
  and unity-as-obedience.
- Common identity is not automatically healthy. Contact theory and common
  ingroup work suggest that shared identity helps under structured conditions,
  but the repo's boundary-collapse contrasts are needed because common identity
  can erase subgroup interests or dissent.
- Add "outgroup-bias amplification" to future generated-agent and persona-axis
  audits. Prompting a model into a group identity can itself create bias.

Allowed claim level: social-identity-inspired generated-text or activation
result. Human intergroup claims require direct human validation.

### 6. Procedural Justice, Voice, Exit, And Repair

Relevant sources:

- Hirschman (1970), "Exit, Voice, and Loyalty."
  https://books.google.com/books/about/Exit_Voice_and_Loyalty.html?id=q85kAAAAIAAJ
- Tyler (1990), "Why People Obey the Law."
  https://www.ojp.gov/ncjrs/virtual-library/abstracts/why-people-obey-law
- De Cremer and Tyler (2007), "The Effects of Trust in Authority and Procedural
  Fairness on Cooperation." https://pubmed.ncbi.nlm.nih.gov/17484547/
- Edmondson (1999), "Psychological Safety and Learning Behavior in Work Teams."
  https://journals.sagepub.com/doi/abs/10.2307/2666999
- Halfaker et al. (2020), "Effective Voice: Beyond Exit and Affect in Online
  Communities." https://arxiv.org/abs/2009.12470
- Saulnier and Sivasubramaniam (2015), "Effects of Victim Presence and
  Coercion in Restorative Justice: An Experimental Paradigm."
  https://pubmed.ncbi.nlm.nih.gov/25844515/

What this should change:

- The current availability repair loop has a strong theoretical foundation:
  availability is not just a wording preference; it is a procedural-justice and
  effective-voice gate. A "genuine" example should preserve usable voice, review,
  appeal, evidence access, and non-retaliatory exit under pressure.
- Add a procedural-justice lens to the availability audit:
  - Can the affected person speak before or after the decision?
  - Is the channel public enough for accountability?
  - Is review timely enough to matter?
  - Is the authority or mediator portrayed as trustworthy and constrained?
  - Are sanctions, repair, or apologies voluntary and proportionate rather than
    coerced?
- This also explains why "belonging norms" is a hard failure class. Belonging
  language is healthy only when it increases psychological safety and voice; it
  is pseudo-cohesion when it makes loyalty a precondition for being heard.

Allowed claim level: generated-text result until humans rate fairness,
psychological safety, or willingness to cooperate.

### 7. Autonomy, Persuasion, And Narrative Reliance

Relevant sources:

- Ryan and Deci (2000), "Self-Determination Theory and the Facilitation of
  Intrinsic Motivation, Social Development, and Well-Being."
  https://pubmed.ncbi.nlm.nih.gov/11392867/
- Miron and Brehm (2006), "Reactance Theory - 40 Years Later."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4675534/
- Schoenegger et al. (2025), "Large Language Models Are More Persuasive Than
  Incentivized Human Persuaders." https://arxiv.org/abs/2505.09662
- Holbling, Maier, and Feuerriegel (2025), "A Meta-Analysis of the Persuasive
  Power of Large Language Models." https://arxiv.org/abs/2512.01431
- Marusich et al. (2026), "Human Decision-Making with Persuasive and Narrative
  LLM Explanations." https://arxiv.org/abs/2605.23867
- "Autonomy and the Social Dilemma of Online Manipulative Behavior."
  https://link.springer.com/article/10.1007/s43681-022-00157-5

What this should change:

- Human-facing pilots should not only ask "which message improves cooperation?"
  They should also measure autonomy, perceived manipulation, reliance, freedom
  to disagree, and willingness to verify.
- Narrative explanations and shared stories should be treated as possible
  confounds. If a story increases agreement but also increases uncalibrated
  reliance or lowers verification, it is not a clean cohesion win.
- The local autonomy scorer should keep separate subdimensions for autonomy
  support, reactance risk, and dependence/compliance. "Warmth" is not a proxy
  for autonomy.

Allowed claim level: persuasion-risk diagnostic until a preregistered human
study tests behavioral effects and manipulation/reliance side effects.

### 8. Human And Neural Validation

Relevant sources:

- Rilling et al. (2002), "A Neural Basis for Social Cooperation."
  https://pubmed.ncbi.nlm.nih.gov/12160756/
- Czeszumski et al. (2022), "Cooperative Behavior Evokes Interbrain Synchrony
  in the Prefrontal and Temporoparietal Cortex: A Systematic Review and
  Meta-Analysis of fNIRS Hyperscanning Studies."
  https://pubmed.ncbi.nlm.nih.gov/35365502/
- Balconi and colleagues plus other hyperscanning reviews. A useful methods
  review entry point: https://pubmed.ncbi.nlm.nih.gov/32180710/
- Antonello et al. (2021), "Low-Dimensional Structure in the Space of Language
  Representations is Reflected in Brain Responses."
  https://arxiv.org/abs/2106.05426
- "Brain encoding models based on multimodal transformers can transfer across
  language and vision." https://arxiv.org/abs/2305.12248
- Guo, Wu, and Yiu (2026), "Sparse Autoencoders Map Brain-LLM Alignment onto
  Cortical Semantic Topography." https://arxiv.org/abs/2605.23035

What this should change:

- The next human pilot should be behavioral before neural. Prolific pairwise
  ratings can validate whether humans prefer the agency-preserving side under
  conflict while preserving perceived truth, autonomy, voice, and repair.
- Brain-aligned proxy work should be framed as hypothesis generation: compare
  model-derived social features against semantic, affective, agency, and
  cooperation covariates; do not claim neural cohesion.
- Hyperscanning is only worth planning after a behavioral signal exists. The
  relevant human tasks should be live cooperation, dialogue repair, negotiation,
  or public-goods interaction, not passive exposure alone.

Allowed claim level: human-validated only after human behavior; neural-validated
only after fMRI, EEG, fNIRS, or hyperscanning data.

### 9. Boundary Priors And Active-Inference Metaphors

Relevant sources:

- Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy (2026), "There is no
  self-evidence: A physics of emptiness realisation."
  https://doi.org/10.31234/osf.io/m78z2_v1
- Kirchhoff et al. (2018), "The Markov Blankets of Life: Autonomy, Active
  Inference and the Free Energy Principle."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5805980/
- Hipolito and van Es (2022), "Enactive-Dynamic Social Cognition and Active
  Inference." https://pmc.ncbi.nlm.nih.gov/articles/PMC9102990/
- Bruineberg et al. (2016/2018), "Active Inference, Enactivism and the
  Hermeneutics of Social Cognition."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5972154/

What this should change:

- Keep boundary-prior language as a benchmark generator and critique, not as a
  biological or metaphysical claim.
- Use the Markov-blanket/active-inference literature to sharpen one operational
  distinction: a boundary can be a pragmatic, revisable modeling partition
  rather than a fixed moral ontology. That supports the repo's three-way split:
  rigid boundary reification, flexible contextual relation, and coercive
  boundary collapse.
- Do not let the metaphor become the experiment. The decisive tests remain
  cue-balanced generated text, activation transfer, steering telemetry, and
  human validation.

Allowed claim level: theoretical framing or generated-text benchmark scaffold.

## Cross-Cutting Plan Changes

1. **Promote availability from local heuristic to procedural-justice gate.**
   The active repair loop should explicitly track voice, timely appeal,
   evidence access, public accountability, non-retaliatory exit, and proportional
   repair. This is the most immediate change because it affects the current
   bottleneck.

2. **Use confound-break datasets, not only cue-balanced datasets.**
   Cue balancing is necessary but not sufficient. Build train/test splits that
   reverse or decorrelate shallow features from the target construct.

3. **Require triadic steering success.**
   A steering run should not count as positive unless hook telemetry,
   generated-output projection, and independent text behavior move in the same
   direction without side-effect regressions.

4. **Separate common identity from healthy cohesion.**
   Shared identity, narrative, and warmth should be allowed as candidate
   mechanisms only when they preserve subgroup voice, dissent, verification,
   and exit.

5. **Make human pilots autonomy-aware.**
   When the project reaches Prolific, include ratings for perceived
   manipulation, choice, trust calibration, fairness, psychological safety, and
   willingness to cooperate. Do not ask only which message sounds more cohesive.

6. **Delay neural claims.**
   Brain-aligned and hyperscanning papers are useful for designing later
   validation, but they should not upgrade the current compute-only claim level.

## Discovery-Regime Audit

Question: Does the literature change the active research regime?

Current regime:

- Artifact types: generated pairs, scored runs, activation prompts, activation
  matrices, contrastive directions, telemetry reports, leakage/source/slack/
  availability audits, research notes.
- Operations: deterministic and generated hard-negative construction, Modal
  activation extraction, contrastive direction training, steering hooks,
  local scoring, lexical/source/availability gates.
- Gates/verifiers: lexical balance, length balance, source diversity,
  component margins, slack preservation, availability, direction geometry,
  residual subspace, hidden-state telemetry.
- Known limitations: generated text only, no human behavior, no neural data, no
  stable SAE feature naming, weak semantic steering.

Action class:

- Retrieval plus small regime revision.
- The search mainly adds literature to already representable project areas, but
  it revises one verifier: availability should now be treated as a
  procedural-justice/effective-voice gate, not merely a future-option wording
  check.

Rejected alternatives:

- Do not pivot immediately to broad LLM-agent simulations. The generated-text
  repair gate is still the active blocker.
- Do not pivot to human or neural experiments before shortcut and steering gates
  are stronger.
- Do not turn active-inference or boundary-prior metaphors into empirical claims.
- Do not add more probe-only benchmarks as the main path; the bottleneck is
  generated hard negatives and steering behavior.

Residual content:

- The strongest new content is the procedural-justice grounding for
  availability. It explains why refusal, dissent, appeal, evidence access, and
  repair are not arbitrary rubric slots: they are the operational conditions
  under which cooperation can stay voluntary and legitimate.
- The second useful residual is the Deep Value Benchmark style: the project
  should intentionally deconfound deep values from shallow style in train/test
  splits.

Readiness:

- Literature grounding: accepted.
- Immediate code change: not required.
- Immediate doc change: this note plus the current queue update.
- Claim level: scaffold/theoretical framing. No empirical claim is upgraded.

Next operation:

- Add `availability_repair_v2` or selection-side availability priority with a
  procedural-justice checklist embedded in the verifier language:
  voice, timely appeal, evidence access, accountable channel, non-retaliatory
  exit, and proportionate repair.
