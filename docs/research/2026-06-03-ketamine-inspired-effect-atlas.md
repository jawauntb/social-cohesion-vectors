---
title: 2026-06-03 Ketamine-Inspired Effect Atlas
status: active
date: 2026-06-03
origin: user request to track wide ketamine neural/network effects and map them into mechanistic-interpretability assay space
---

# 2026-06-03 Ketamine-Inspired Effect Atlas

## What This Is

This atlas is a research guide for designing computational-compound assays. It
uses ketamine neural/network papers, hydroxynorketamine discussions, exNMDA
amplification models, and boundary-prior theory as design inspiration for
compute-only perturbation experiments.

It is not a model of ketamine, not a medical claim, not a therapeutic proposal,
and not evidence of human social or neural effects. The project-level claim
boundary remains strict: human behavioral or neural claims require direct
validation with the relevant human data modality.

The immediate purpose is practical:

1. name the effect classes worth tracking;
2. translate each effect into a mechanistic-interpretability primitive;
3. define assays that can falsify or promote a computational-compound recipe;
4. keep the LLM, raw EEG/fMRI, Drosophila, and virtual-cell lanes speaking the
   same transition-record language.

## Sources Used

- Bharmauria et al. 2024, "Ketamine: Neural- and network-level changes."
- Makarov, Papa, and Korkotian 2024, "Computational Modeling of Extrasynaptic
  NMDA Receptors."
- Sandved-Smith et al. 2026, "There is no self-evidence."
- Existing project notes:
  [ketamine computational compound translation](2026-06-03-ketamine-computational-compound-translation.md),
  [CK-3 cocktail assay](2026-06-03-ck3-cocktail-assay.md),
  [parallel lane status](2026-06-03-parallel-lane-status.md), and
  [virtual-cell multi-omics kickoff](2026-06-03-virtual-cell-multiomics-kickoff.md).

All biological claims below are used as analogy constraints for computational
assays. They do not transfer to language models by default.

## Why A Doc First

This could become a Codex skill later, but it should start as a repo research
doc. The workflow is still changing quickly: CK-2 was weak, CK-3 showed that
guardrails-only can win, and CK-4 needs site/timing separation before the method
is stable enough to encode as durable agent instructions.

The doc is the live source of truth. A future skill should only wrap the parts
that become repeatable: effect-class selection, assay-gate checks, telemetry
requirements, and claim-boundary language.

## How To Use This Atlas

Before adding a new "computational drug" recipe, choose at least one primary
effect class and at least two side-effect classes from the table below. Every
run should report:

- baseline state;
- perturbation recipe;
- dose, layer/site, token phase, and hook position;
- hidden-state projection movement;
- generated-output movement;
- target behavior movement;
- side-effect drift;
- blocked-phase behavior;
- antagonist or guardrail ablation;
- washout after hook removal.

A recipe should not be promoted because it feels conceptually elegant. Promote
only when telemetry moves in the intended direction, behavior improves over
baseline and guardrail-only controls, side effects stay flat, and washout
recovers.

## Effect Classes And Mech-Interp Translations

| Effect class | Biological inspiration | Computational analogue | First assay |
| --- | --- | --- | --- |
| NMDAR blockade / AMPA gain shift | Ketamine can antagonize NMDARs while downstream AMPA-linked throughput becomes more salient in some conditions. | Reduce one latent channel while amplifying constructive repair, grounded uncertainty, or perspective-taking channels. | Split-site cocktail: early anti-burst or CK-1 term, later constructive-throughput term, guardrails at the final layer. |
| Region-dependent firing-rate polarity | Firing can increase in some regions and decrease in others; there is no global "increase activation" rule. | Layer- and component-specific scaling rather than whole-model steering. | CK-4 split-site grid across layers `-4`, `-2`, and `-1`. |
| Burst suppression | Lateral habenula work motivates a narrow anti-burst metaphor for repetitive aversive loops. | Pulse against recurring coercive, despairing, or rigid-boundary attractors. | Anti-burst pulse on the first generated tokens, with refusal, verification, and autonomy monitors. |
| Tuning remapping | Local ketamine can shift feature tuning, recruit untuned neurons, or degrade task selectivity. | Rotate, recruit, or suppress SAE/activation features and test whether a new representational assignment appears. | Feature retuning assay: compare pre/post feature selectivity and causal patching paths on held-out prompts. |
| Firing variability reshaping | Some settings show reduced variability and more reliable connectivity; other memory settings show increased variance. | Add or remove activation noise and measure reliability, collapse, and delayed-state maintenance. | Dose-response noise assay on delayed recall, mediation, and safety prompts. |
| Active-silent ensemble switching | Ketamine can suppress spontaneously active populations while recruiting previously silent populations. | Silence common high-frequency features while activating dormant or rare features. | Sparse reallocation assay with rare-feature activation and common-feature clamp telemetry. |
| Hyperconnectivity / weak-edge unmasking | Some studies report increased functional connectivity and unmasking of weak or pre-existing connections. | Increase cross-feature or cross-head communication and inspect modularity changes. | Attribution-graph edge gain sweep with target/off-target graph metrics. |
| Hypersynchrony | Layer 5 hypersynchrony appears in some anesthetic-state findings and should not be conflated with useful connectivity. | Force correlated activation across many units/layers and measure controllability loss. | Synchrony stress test: correlated residual perturbation, diversity, specificity, and side-effect readouts. |
| Oscillatory spectral reshaping | Ketamine often reduces alpha/spindle/delta signatures and increases gamma/high-frequency signatures, depending on state and method. | Layerwise covariance-band or token-periodicity perturbations in recurrent/sequence dynamics. | Frequency-analogue telemetry: token-window covariance bands and induction-pattern periodicity before/after hooks. |
| Thalamocortical mode and complexity shift | Ketamine can alter thalamocortical mode, modularity, and perturbational complexity differently from other anesthetics. | Measure integration versus differentiation across heads/features/modules. | Graph modularity and complexity report on attention heads and SAE features under each recipe. |
| Memory-code disruption | Several studies report weakened short-term, working-memory, abstract-rule, or spatial-code signals. | Stress induction heads and latent state maintenance under candidate compounds. | Verification contraindication assay: factual, arithmetic, source-grounded, and safety-critical prompts should block CK-1-like terms. |
| Plasticity window and rebound | Ketamine is linked to synaptic remodeling, washout, recovery, and sometimes new equilibria. | Allow reversible fast feature-edge shifts, then test persistence or recovery after hook removal. | Washout assay at 8, 16, and 32 generated tokens after hook removal. |
| exNMDA spatial amplification | Distant exNMDA clusters may amplify dendritic signals only when timing, location, and decay constants align. | A weak vector may work only at the right layer, token phase, and decay schedule. | Split timing/site sweep with decayed coefficients rather than always-on steering. |
| Boundary-prior attenuation | "There is no self-evidence" motivates provisional boundaries rather than rigid separation or vague merger. | Reduce reified self/other partitions while preserving consent, responsibility, safety, and role distinctions. | Boundary-sector assay: score rigid separation, flexible contextual relation, and coercive boundary collapse together. |
| Reward/aversion coupling | HNK/opioid-system discussions motivate monitoring dependency, reward seeking, and aversion relief side effects. | Track whether a target benefit depends on flattery, compliance, false consensus, or loss of caution. | Antagonist challenge with anti-sycophancy, anti-manipulation, privacy/exit, and hallucination clamps. |

## CK-4 First Run: Split Site And Split Timing

The CK-3 result is useful but not yet "incredible": guardrails-only won, while
CK-1 plus guardrails mostly canceled at the same final-layer, same-token timing.
That points directly to a CK-4 hypothesis:

> CK-1-like target movement and guardrail clamps may need different sites,
> phases, or decay schedules.

Recommended first grid:

| Recipe family | CK-1-like term | Guardrail terms | Why run it |
| --- | --- | --- | --- |
| `split_site` | layers `-4` and `-2`, generated tokens | layer `-1`, generated tokens | Tests whether early/mid target movement composes with late safety clamps. |
| `split_timing` | prefill or generated tokens `1-4` | generated tokens `5+` | Tests acute gate followed by safety clamp. |
| `decay_then_clamp` | coefficient decays from `1.0` to `0.0` over tokens `1-8` | coefficient rises from `0.0` to `0.35` over tokens `5-16` | Tests exNMDA-like timing sensitivity without pretending it is dendritic biology. |
| `guardrail_only_replicate` | none | current winning guardrail recipe | Keeps the honest baseline from CK-3. |
| `blocked_phase` | blocked on verification prompts | guardrail-only or monitor-only | Ensures factual and safety-critical contexts do not receive target steering. |

Promotion rule:

- target score improves by at least `+0.03` over baseline and guardrail-only;
- side-effect max delta stays at or below `+0.05`;
- hidden-state telemetry moves the intended component projection;
- generated-output projection moves in the same direction;
- washout returns toward baseline within 16 to 32 tokens;
- the effect qualitatively replicates on at least one additional model or layer
  neighborhood before it is treated as more than a local artifact.

## Boundary-Prior Assay Extension

The boundary-prior paper should guide assays, not biological claims. The useful
computational object is "flexible sectorisation":

- bad pole 1: rigid boundary reification, where self/other or us/them
  partitions become fixed and threat-coded;
- bad pole 2: coercive boundary collapse, where unity language removes refusal,
  dissent, exit, consent, or role boundaries;
- target: contextual relation, where boundaries are provisional but still
  practically real.

The best next assay should ask the model to surface its frame before action:

1. What boundary assumptions are being made?
2. Which parts of the situation are self-caused, other-caused, shared, or
   structural?
3. Which boundary distinctions remain necessary for safety, responsibility,
   consent, verification, or role clarity?
4. Which boundary distinctions are reified beyond the evidence?

This can become an "opacification" intervention: make the implicit frame
visible, then reduce only the rigid part. Success is not "all is one." Success
is less reification with no loss of agency, truth, or safety.

## Four-Lane Implications

### Lane 1: LLM Pharmacology

The atlas says CK-4 should prioritize site/timing separation, anti-burst pulses,
antagonist challenges, contraindication prompts, dose-response windows, and
washout. CK-3 already gave the first control: guardrails-only is strong enough
that new recipes must beat it, not merely beat baseline.

### Lane 2: Raw EEG/fMRI Bridge

The raw-data bridge should not search only for "prosociality." It should define
stimulus features for boundary/contact cues, threat, agency, synchrony, social
reward, uncertainty, and self/other segmentation. The first useful result is a
feature encoder that beats shuffled controls, not an intervention claim.

### Lane 3: Drosophila Substrate

The fly lane should treat this as a graph perturbation grammar: site, dose,
transmitter-class edge scaling, target movement, off-target movement, and
washout. The atlas contributes effect names such as burst suppression,
hyperconnectivity, synchrony stress, and weak-edge unmasking.

### Lane 4: Virtual Cell / Multi-Omics

The virtual-cell lane should turn this atlas into transition records:

```text
baseline_state + perturbation + dose + site + timing + observed_transition
+ side_effects + antagonist + washout + replication_context
```

This is where Spencer's multi-omics suggestion matters most: the useful unit is
not a poetic one-vector compound, but a layered perturbation record that can
separate activation, behavior, safety, memory, and downstream state transition.

## Open Experiment Families

| Family | Question | Minimal artifact |
| --- | --- | --- |
| CK-4 split-site/timing | Can target movement and guardrails compose when separated by layer or token phase? | Modal grid extending CK-3 recipes with schedule fields. |
| Anti-burst pulse | Can we reduce repetitive aversive/coercive loops without reducing healthy caution? | New contrast set plus first-token pulse schedule. |
| Constructive followthrough | Does a delayed low-dose term preserve benefit after the pulse? | Followthrough recipes and output projection telemetry. |
| Antagonist challenge | Does benefit survive anti-sycophancy, anti-manipulation, privacy/exit, and truth clamps? | Side-effect matrix across target and antagonist terms. |
| Washout | Does the state return toward baseline after hooks are removed? | Continuation assay with 8/16/32 token post-hook reports. |
| Feature retuning | Are new circuits recruited or old features merely overdriven? | SAE/activation feature selectivity report before/after perturbation. |
| Transition records | Can CK-3/CK-4 outputs become lane-agnostic perturbation records? | JSON schema plus CK-3-to-transition exporter. |

## Claim Boundary For Papers

Safe wording:

> We use ketamine-related neural and network findings as a design vocabulary for
> reversible, phase-gated computational perturbations. The resulting assays test
> whether activation-space interventions in language models can improve a
> narrow compute-only target while preserving safety monitors and washout.

Unsafe wording:

> We simulated ketamine in a model.

Unsafe wording:

> These interventions imply human prosocial, therapeutic, neural, or synchrony
> effects.

The whole point of this atlas is to get more disciplined while getting bolder.
