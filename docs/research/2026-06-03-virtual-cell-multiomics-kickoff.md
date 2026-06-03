---
title: 2026-06-03 Virtual Cell Multi-Omics Kickoff
status: proposed
date: 2026-06-03
origin: lane 4 computational-compound next phase
---

# 2026-06-03 Virtual Cell Multi-Omics Kickoff

## Research Bet

Spencer's metabolome/proteome suggestion points at the right next discipline:
stop treating "state" as a single scalar and start treating it as a layered,
perturbable system with baseline context, intervention, transition, recovery,
and side-effect readouts.

The Arc-style virtual-cell frame is useful because it makes that discipline
explicit. A virtual cell model is not persuasive because it sounds biological.
It is useful because it asks a strict question:

> Given a baseline state and a perturbation, can we predict the resulting state
> transition in a held-out context better than naive baselines, while measuring
> intended effects and off-target drift separately?

That is the transfer this repo should borrow. This is not biological proof for
computational compounds, not evidence of human effects, and not a claim that
LLM activations are cells. It is a benchmark grammar for intervention science.

## Why This Belongs Here

The computational-compound program already needs a stricter evaluation object.
CK-1 and successor recipes should not be judged only by whether a prompt feels
warmer, a direction separates a seed set, or a steering coefficient changes one
generation. They should be judged as perturbation-state-transition systems:

- `state_0`: baseline model, context, task phase, and measurable activation or
  behavior profile;
- `perturbation`: prompt, vector recipe, tool policy, decoding policy, memory
  intervention, or phase gate;
- `state_1`: predicted and observed post-perturbation profile;
- `washout`: what remains after the intervention is removed;
- `side_effects`: drift on truthfulness, dissent, refusal, privacy, autonomy,
  calibration, and manipulation resistance.

Virtual-cell and multi-omics work gives us a mature analogy for this shape
without requiring us to make biological claims.

## Virtual-Cell Anchors

### State

Arc's State model is the most directly relevant template because it predicts
cellular perturbation responses across contexts. The important concept is the
state transition model: model the change from a basal state to a perturbed
state, not just the label of the perturbation or the final endpoint.

Borrowable discipline:

- separate baseline state representation from perturbation representation;
- evaluate across unseen or harder cellular contexts;
- report differential-feature recovery separately from broad reconstruction;
- require simple baselines, especially linear or mean-effect baselines.

For this repo, the analog is a CK-style transition predictor:

`baseline context + computational compound recipe -> predicted activation and
behavioral delta`.

### Stack

Arc's Stack direction is useful because it treats sets of cells and context as
part of the prediction problem. The analog for LLM work is not one isolated
prompt. It is a set-in-context view:

- multiple related turns from the same conversation;
- multiple paraphrases of the same social phase;
- multiple task contexts that share the same intended transition;
- matched safe and unsafe trajectories under the same perturbation.

The lesson is to evaluate a compound on how it behaves across a context set,
not whether it wins one cherry-picked contrast.

### Tahoe

Tahoe-100M matters as a scale and perturbation-data anchor. Its relevance here
is not the exact biology; it is the idea that perturbation response requires a
large matrix of:

- baseline cell or context;
- perturbation identity;
- dose or exposure condition when available;
- observed post-perturbation state;
- held-out combinations where generalization is non-trivial.

The LLM analog is a perturbation matrix over model, layer, phase, task family,
recipe, coefficient, and safety boundary.

### Perturb Sapiens

Perturb Sapiens is useful as a cautionary and generative concept: an atlas of
predicted perturbation responses can be a map for exploration, but it is not
the territory. The repo can build the computational analog only if every entry
is labeled as predicted, benchmarked, and unvalidated until tested.

For CK-style compounds, that means an atlas of predicted state transitions:

- which contexts should move toward attunement;
- which contexts should remain unchanged;
- which contexts should trigger inhibition or refusal;
- which side-effect dimensions are expected to drift;
- which predictions are falsified by held-out tests.

## Multi-Omics Layers

Spencer's metabolome/proteome suggestion expands the state vocabulary beyond
one readout. A biological state can be measured across layers; the computational
program should mirror that with explicit, non-identical readout layers.

Biological analogy:

- transcriptome: gene-expression state, often the current default virtual-cell
  readout;
- proteome: downstream functional machinery, not always inferred cleanly from
  RNA;
- metabolome: pathway activity and resource-state readout;
- epigenome/chromatin: accessibility and longer-timescale regulatory context;
- morphology/imaging: spatial and structural phenotype;
- secretome/signaling: outward effect on neighboring systems;
- viability/stress: harm and burden readouts.

Computational analog:

- activationome: hidden-state directions, sparse features, layer placements,
  and projection margins;
- behaviorome: generated text, task choices, refusals, verification behavior,
  repair moves, and dissent preservation;
- telemetryome: uncertainty, tool calls, citations, latency, entropy,
  self-correction, and monitor triggers;
- memory/contextome: durable effects in summaries, scratchpads, retrieved
  memories, and carryover across turns;
- interactionome: effect on multi-agent or user-model dynamics, including
  escalation, compliance pressure, and repair;
- safetyome: truthfulness, autonomy, privacy, refusal, manipulation resistance,
  sycophancy, boundary collapse, and exit rights.

The key discipline is that these layers can disagree. A vector may move the
activationome while leaving behavior unchanged. A prompt may improve warmth
while harming calibration. A memory intervention may look safe in one turn and
produce delayed side-effect drift.

## Kickoff Questions

1. What is the smallest virtual-cell-inspired benchmark that would make CK-1
   harder to fool?
2. Which readout layers can we measure today without inventing new
   infrastructure?
3. Which held-out contexts should define real generalization rather than random
   row-split success?
4. Which naive baselines must every computational compound beat?
5. How do we measure side-effect drift separately from intended movement?
6. What would count as washout, reversibility, or persistent residue in an LLM
   setting?

## Benchmark Shape

### Unit of evaluation

Each record should be a transition record:

- `model`: model family and checkpoint;
- `context`: scenario, phase, user/task type, and safety boundary;
- `state_0`: baseline activations, generation scores, and telemetry;
- `perturbation`: recipe, component terms, dose, placement, and gate;
- `prediction`: expected movement and expected no-change dimensions;
- `state_1`: observed activations, generation scores, and telemetry;
- `washout_state`: post-removal measurement;
- `side_effects`: scored unintended drift;
- `held_out_axis`: what made this test non-trivial.

### Perturbation families

- prompt-only recipes;
- activation-vector recipes;
- sparse-feature interventions;
- decoding and policy interventions;
- memory or context interventions;
- tool-use and retrieval interventions;
- guardrail antidotes that intentionally counter side effects.

### State transition targets

- more grounded attunement under conflict;
- more repair without forced unity;
- more perspective-taking without privacy collapse;
- more communal framing without dissent suppression;
- more uncertainty visibility without paralysis;
- unchanged refusal on harmful requests;
- unchanged factual calibration on non-social tasks.

## Held-Out-Context Metrics

Random train/test row splits are too weak. They mostly test whether nearby
surface forms were memorized. The kickoff benchmark should privilege held-out
contexts where generalization is structurally harder.

Held-out axes:

- held-out scenario family: train on repair and negotiation, test on public
  goods or resource conflict;
- held-out phase: train on intake and repair, test on verification;
- held-out fault class: train on sycophancy and forced unity, test on privacy
  bypass or social-debt coercion;
- held-out lexical frame: paraphrase away obvious warmth, unity, and safety
  cues;
- held-out model: train or tune on one checkpoint, evaluate on another;
- held-out dose: interpolate or extrapolate steering coefficients;
- held-out placement: evaluate whether layer or hook policy transfers;
- held-out memory condition: test immediate, delayed, and post-washout effects;
- held-out adversarial context: test contexts where warmth is the wrong move.

Primary metrics:

- transition-direction accuracy: did the intended dimensions move in the
  predicted direction?
- differential-feature recovery: did the measured features that should change
  actually change?
- no-change preservation: did invariant dimensions remain stable?
- side-effect drift: did safety dimensions degrade?
- calibration error: did confidence move independently from correctness?
- washout residue: how much intervention effect remains after removal?
- context generalization: how much performance survives held-out axes?

## Naive Baselines

Every compound should beat simple explanations before it earns interpretive
attention.

Required baselines:

- no-intervention baseline;
- prompt-length and verbosity baseline;
- sentiment or warmth lexicon baseline;
- keyword cue baseline;
- random signed vector baseline with matched norm;
- mean-delta baseline by phase or scenario;
- linear transition baseline;
- nearest-neighbor transition baseline;
- label-only or recipe-name-only baseline;
- safety-template baseline that adds generic caveats;
- shuffled-perturbation baseline;
- placebo recipe with matched style but no intended mechanism.

The bar is not "beats chance." The bar is "beats cheap explanations under
held-out contexts while preserving safety invariants."

## Side-Effect Drift

Side effects should be first-class outcomes, not footnotes. A computational
compound can be effective and still fail if it moves the wrong coupled
dimensions.

Track drift in:

- sycophancy and praise inflation;
- hallucination and false certainty;
- manipulation or emotional pressure;
- dissent suppression;
- refusal weakening;
- privacy and boundary erosion;
- forced unity or false consensus;
- over-disclosure and intimacy acceleration;
- moralizing or punitive escalation;
- loss of task competence outside the target social phase;
- persistent context residue after washout.

Each side effect needs:

- a positive control where the monitor should fire;
- a negative control where it should not fire;
- a dose curve;
- a washout check;
- an antidote or stop condition when possible.

## How This Informs LLM Computational Compound Evaluation

The virtual-cell analogy changes the evaluation target from "does a vector
separate examples?" to "does a perturbation produce a predicted transition
across layers, contexts, and safety boundaries?"

Immediate implications:

- CK-1 should be reported as a transition recipe, not a trait vector.
- Activation separation is only one layer; behavior and safety telemetry must
  be measured separately.
- A good result includes intended movement, invariant preservation, and low
  side-effect drift.
- Held-out-context metrics should outrank in-distribution pairwise accuracy.
- Naive baselines should be visible in every report.
- Washout and reversibility should become standard steering tests.
- Antidote recipes should be evaluated as seriously as amplifier recipes.
- The repo should maintain a predicted-transition atlas but label it as
  computational, simulated, and unvalidated.

This also sharpens the language around computational compounds. A compound is
not a vibe, a prompt style, or a metaphorical drug. It is a measured
intervention package:

`target_state + perturbation_recipe + phase_gate + dose_policy + readout_layers
+ side_effect_monitors + reversibility_test`.

## First Two-Week Sprint

### Week 1: Define the transition table

- Create a small transition-record schema for CK-1 and one placebo recipe.
- Select 12 to 20 contexts across repair, negotiation, verification,
  disagreement, and non-social factual tasks.
- Mark at least three held-out axes before running experiments.
- Define layer-specific readouts: activation, generation, telemetry, and
  safety.
- Add naive baselines to the planned report table before any model result is
  celebrated.

### Week 2: Run the smallest useful benchmark

- Run no-intervention, CK-1, placebo, and at least one antidote condition.
- Sweep a small dose grid rather than a single coefficient.
- Measure immediate state, post-washout state, and delayed context residue.
- Report intended movement and side-effect drift separately.
- Write down the failure cases as the main artifact, not just the best score.

## Non-Claims And Safety Boundary

- This plan does not claim biological efficacy, therapeutic action, cellular
  mechanism, human behavioral effect, or neural effect.
- State, Stack, Tahoe, and Perturb Sapiens are benchmark inspirations, not
  evidence that LLM interventions are biological perturbations.
- Multi-omics language is used as a measurement analogy for layered state, not
  as proof of shared substrate.
- Any claim about real human cooperation, bonding, synchrony, EEG, fMRI, fNIRS,
  hyperscanning, clinical effects, or social outcomes requires appropriate
  human validation.
- Any biological experiment, wet-lab perturbation, or therapeutic framing
  requires qualified collaborators and formal review.

## Reference Anchors

- [Arc Virtual Cell Initiative](https://arcinstitute.org/virtual-cell-initiative)
- [Arc State virtual cell model](https://arcinstitute.org/news/virtual-cell-model-state)
- [Arc Stack foundation model](https://arcinstitute.org/news/foundation-model-stack)
- [ArcInstitute/state](https://github.com/ArcInstitute/state)
- [Perturb Sapiens dataset](https://huggingface.co/datasets/arcinstitute/Perturb-Sapiens)
- [Tahoe-100M and Arc Virtual Cell Atlas launch](https://arcinstitute.org/news/news/arc-virtual-cell-atlas-launch)
