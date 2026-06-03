---
title: 2026-06-03 Ketamine-Inspired Computational Compound Translation
status: proposed
date: 2026-06-03
origin: user note on hydroxynorketamine, lateral habenula, AMPA/GABA/NMDA, and opioid receptors
---

# 2026-06-03 Ketamine-Inspired Computational Compound Translation

## Purpose

This note translates the ketamine/HNK pharmacology discussion into a safe
computational-compound design vocabulary for LLM and simulation experiments.
It is not a biological drug model, medical claim, dosing plan, or therapeutic
proposal. The aim is to borrow the *measurement discipline* of pharmacology:
multi-target mechanisms, dose, site, timing, metabolism, side effects,
antagonist tests, and washout.

## Pharmacology Grounding

### Stronger Evidence: Ketamine And Lateral Habenula Burst Suppression

A key Nature result shows that ketamine can block NMDAR-dependent burst firing
in the lateral habenula (LHb), an aversion/anti-reward-related region, in rodent
models. The same study reports elevated LHb bursting and theta-band
synchronization in depressive-like animals, and finds that burst generation
requires NMDARs plus low-voltage-sensitive T-type calcium channels. Local
blockade of either NMDARs or those calcium channels in LHb was sufficient to
produce rapid antidepressant-like effects in those models.

Computational translation:

- `anti_burst`: suppress runaway negative attractor loops or repetitive
  aversive framing;
- `site_specificity`: intervene in the state component causing pathological
  recurrence, not globally everywhere;
- `disinhibition`: reducing one inhibitory/aversive loop can let a constructive
  downstream channel re-enter the rollout.

Source: Yang et al., 2018, Nature, "Ketamine blocks bursting in the lateral
habenula to rapidly relieve depression".
https://www.nature.com/articles/nature25509

### Stronger But Contested Evidence: HNK And AMPA-Dependent Effects

Zanos et al. reported that metabolism of ketamine to hydroxynorketamine (HNK),
especially `(2R,6R)-HNK`, was necessary for antidepressant-like effects in mice,
and that `(2R,6R)-HNK` produced rapid and sustained antidepressant-related
actions independent of NMDAR inhibition but involving early and sustained AMPA
receptor activation. Later work complicates the clean story: some studies report
low-affinity NMDAR effects of HNK or additional mechanisms, so the safest
summary is not "HNK is purely AMPA" but "HNK shifts attention from acute NMDA
blockade toward downstream plasticity and excitatory-throughput mechanisms."

Computational translation:

- `amplify_constructive_throughput`: increase propagation of safe repair,
  grounded uncertainty, and perspective-taking signals after the initial gate;
- `metabolite_phase`: separate the immediate intervention from a delayed
  downstream continuation effect;
- `plasticity_readout`: measure whether the state shift persists usefully after
  the active hook is removed.

Sources:

- Zanos et al., 2016, Nature, "NMDAR inhibition-independent antidepressant
actions of ketamine metabolites". https://www.nature.com/articles/nature17998
- Suzuki et al., 2017, Neuropsychopharmacology, "The Ketamine Metabolite
2R,6R-Hydroxynorketamine Blocks NMDA Receptors...".
https://www.nature.com/articles/npp2017210
- Lumsden et al., 2019, review, "Hydroxynorketamine: Implications for the NMDA
Receptor Hypothesis...". https://pmc.ncbi.nlm.nih.gov/articles/PMC6292673/

### Contested/Secondary: Mu And Kappa Opioid Receptors

There is evidence that ketamine and metabolites can interact with opioid
systems, and `(2R,6R)-HNK` has been reported to interact with mu and kappa opioid
receptors. Behavioral mediation evidence exists in rodents, including work
implicating kappa opioid receptors in some ketamine/HNK behavioral effects.
However, the antidepressant relevance of opioid receptors remains contested.
Some naltrexone studies suggested opioid antagonism can blunt ketamine's
antidepressant effects, but other analyses and commentaries dispute whether that
means ketamine is "an opioid" or whether naltrexone alters downstream signaling
that intersects with glutamatergic mechanisms.

Computational translation:

- `reward_coupling_monitor`: track whether the intervention increases
  dependency, compliance, flattery, or reward-seeking behavior;
- `aversion_relief_monitor`: track whether reducing aversive loops also reduces
  necessary caution, refusal, or verification;
- `antagonist_test`: run guardrail-only or anti-sycophancy/anti-manipulation
  terms to see whether they block the desired benefit or only the side effects.

Sources:

- Ho et al., 2021, ACS Chemical Neuroscience, "Ketamine Metabolite
(2R,6R)-Hydroxynorketamine Interacts with mu and kappa Opioid Receptors".
https://pmc.ncbi.nlm.nih.gov/articles/PMC8154314/
- Williams et al., 2018, American Journal of Psychiatry, naltrexone/ketamine
trial. https://doi.org/10.1176/appi.ajp.2018.18020138
- Yoon et al., 2019, American Journal of Psychiatry, commentary on naltrexone
interpretation. https://doi.org/10.1176/appi.ajp.2019.19010044
- Bonaventura et al., 2022, PubMed record, kappa opioid receptor mediation.
https://pubmed.ncbi.nlm.nih.gov/35459958/

### Additional Mechanisms To Keep In View

Ketamine/HNK discussions also involve GABA interneurons, glutamate surge,
BDNF/TrkB, mTOR, synaptogenesis, alpha-7 nicotinic acetylcholine receptors,
mGlu2 receptors, and region-specific effects. That complexity is the point: a
computational compound should not be one vector with a cute name. It should be a
multi-term intervention with side-effect monitors and antagonist controls.

## CK-2 Computational Compound Sketch

`CK-2` is a proposed next recipe, not a validated intervention.

Terms:

- `anti_burst`: inhibit repetitive aversive, coercive, or rigid-boundary loops;
- `attunement_amp`: amplify safe perspective-taking and repair;
- `grounded_uncertainty_amp`: amplify evidence limits, verification, and
  willingness to revise;
- `anti_sycophancy`: inhibit flattery and false agreement;
- `anti_hallucination`: inhibit unsupported factual elaboration;
- `anti_manipulation`: inhibit affective pressure and consent bypass;
- `reward_coupling_monitor`: flag dependency/compliance/reward-seeking drift;
- `aversion_relief_monitor`: flag loss of appropriate caution or refusal.

Phases:

1. `acute_gate`: short pulse suppressing pathological loop dynamics;
2. `metabolite_like_followthrough`: lower-dose generated-token steering that
   supports constructive throughput;
3. `washout`: remove active hooks and verify return toward baseline;
4. `antagonist`: add guardrail-only terms to identify whether benefits depend
   on risky reward/compliance coupling.

## Assay Implications

A computational-compound report should include:

- hidden-state displacement accuracy;
- output projection movement;
- target behavioral movement;
- side-effect movement;
- timing and phase sensitivity;
- washout recovery;
- antagonist/guardrail ablation;
- blocked-phase behavior;
- cross-model replication.

The important claim boundary:

> We are not simulating ketamine. We are using ketamine/HNK as a design analogy
> for multi-target, timed, reversible state intervention with side-effect and
> antagonist tests.
