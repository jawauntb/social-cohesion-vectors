---
title: 2026-06-02 Next Computational-Drug Research Program
status: proposed
date: 2026-06-02
origin: docs synthesis lane F
---

# 2026-06-02 Next Computational-Drug Research Program

## Research Bet

The next program should treat a **computational drug** as a reversible,
dose-controlled, phase-gated state intervention with measured side effects. The
goal is not to claim that language models, flies, cells, or humans share one
mechanism. The goal is to build a careful cross-substrate research vocabulary:
state, dose, gate, substrate, readout, reversibility, and harm boundary.

The near-term output should be a compute-first program that can later support
raw biological data work without overstating what the current repo has shown.

## Program Spine

### CK-1 state-modulator baseline

Use `CK-1 attunement amplifier` as the baseline recipe for the computational
lane. It stays a social-state modulator, not a ketamine analog and not a human
effect claim. The baseline questions are:

- Can a compositional recipe move model activations more reliably than a single
  vector?
- Can phase gates preserve truthfulness, dissent, privacy, refusal, and exit?
- Can telemetry detect warmth-with-side-effects, especially sycophancy,
  hallucination, manipulation, or boundary collapse?

CK-1 should remain the calibration object for every richer direction: if a new
substrate or model cannot improve on CK-1's measurement discipline, it is not
yet ready to carry the "computational drug" label.

Immediate benchmark state:

- `seed`: 4 phase contrasts / 8 activation prompts;
- `cue_balanced`: 4 matched contrasts / 8 activation prompts with tied simple
  lexical cue scores in every safe-vs-pseudo pair;
- `expanded`: seed plus cue-balanced, for 8 contrasts / 16 activation prompts.

Scratch validation on 2026-06-02 found `cue_balanced` lexical cue-solved rate
`0.000` while the deterministic CK-1 scorer still preferred the safe-attunement
pole in all pairs. The next empirical question is whether activation-space
directions survive this harder batch.

First answer: yes, on the scripted prompt batch. Qwen/Qwen2.5-0.5B-Instruct
layer `-1` separates `cue_balanced` CK-1 examples with leave-one-pair-out
accuracy `1.000` and mean projection margin `+5.1063`; the `expanded` batch
also reaches LOO accuracy `1.000` with mean margin `+5.8329`. A small placement
sweep suggests final-layer separation is strongest among layers `-1`, `-2`,
and `-4`.

Second answer: causal steering is mixed. A two-prompt calibration sweep using
the cue-balanced layer `-1` direction favored strong positive steering, with
positive-vs-negative CK-1 success `1.000` and positive-minus-baseline CK-1
delta `+0.053`. The full six-prompt held-out sweep did not replicate that as a
global control result: positive-vs-negative CK-1 success was `0.417`, the mean
positive-minus-negative CK-1 delta was `-0.006`, and the best average score was
at strength `-3`, only `+0.019` above baseline.

Third answer: timing helps. Re-running the same full six-prompt sweep as a
generated-token-only post-hook intervention raises positive-vs-negative CK-1
success to `0.583`, positive-minus-negative CK-1 delta to `+0.015`, and
pseudo-risk delta to `-0.033`, with best strength `+6`. This remains a small
compute-only result, but it supports the central design bet that intervention
timing and phase gating matter.

This is exactly the kind of result the computational-drug frame should expect:
activation separability is not the same as a safe intervention. The next CK-1
work should treat sign, dose, layer, timing, and phase gate as the intervention,
not as afterthought parameters.

### Raw EEG and data direction

The data lane should move from derived labels and feature summaries toward raw,
time-aligned signals: EEG waveforms, stimulus traces, task timing, behavioral
responses, and metadata provenance. The first useful artifact is a registry of
open datasets whose consent, task design, labels, and sampling quality are
compatible with social-state or affect-control questions.

Likely first substrates:

- Alljoined-1.6M: a THINGS-lineage EEG-image corpus with about 1.67M stimulus
  rows, 20 subjects, 16,740 unique images, and consumer-grade 32-channel EEG.
  Treat licensing as non-commercial until verified.
- THINGS-EEG2: a cleaner lab-grade comparator with 10 subjects, 82,160 trials
  per subject, the same 16,740 image conditions, raw EEG, preprocessed features,
  stimuli, and DNN feature maps.
- NSD or BOLD5000: fMRI sanity checks for lower-noise image-to-brain alignment
  before betting too hard on EEG reconstruction.

This is a representation-learning direction before it is a neuroscience claim.
Raw EEG can test whether model-derived social-state axes align with measured
human signals only after preregistered analysis, adequate controls, and
behavioral validation. Until then, it is a substrate for methods development,
quality control, and hypothesis generation.

### Drosophila substrate

Drosophila is the first tractable biological substrate because it has unusually
strong genetic, behavioral, and connectomic resources, including whole-brain
FlyWire-scale connectome work. The fit is not "fly prosociality." The fit is
state modulation in a compact nervous system where perturbation, circuit
structure, behavior, and side effects can eventually be related.

Near-term framing:

- graph toy: use adult FlyWire as a directed weighted connectome substrate for
  path, motif, and perturbation-target ranking;
- drug analog toy: scale or clamp outgoing edges by predicted transmitter class,
  such as acetylcholine, glutamate, GABA, dopamine, serotonin, or octopamine;
- dynamics toy: use a released Brian2 leaky-integrate-and-fire adult fly model
  for optogenetic-like activation and silencing;
- tractable exhaustive toy: use the larval connectome for smaller whole-brain
  perturbation sweeps;
- serious bridge: use connectome-prior causal effectome work only when paired
  with real perturbation or imaging data.

Treat connectome/cell-type structure as a substrate map, not a sufficient causal
model. Reserve any wet-lab direction for collaborators with appropriate animal
protocols and domain expertise.

### Arc virtual-cell direction

The Arc-style virtual-cell lane offers a cellular analog of the CK-1 discipline:
predict how cell states shift under genetic, chemical, cytokine, or
environmental perturbations before treating an intervention as meaningful. This
direction should ask whether state-modulator concepts can be formalized in
cellular terms: perturbation, dose response, target state, off-target drift,
reversibility, and benchmarked prediction error.

The useful transfer is experimental structure:

- baseline state plus perturbation predicts a state transition;
- held-out contexts matter more than random row splits;
- naive baselines are mandatory;
- state discrimination, effect-size calibration, and differential-feature
  recovery are separate metrics.

Arc's State and Stack work make this concrete through transcriptomic
perturbation prediction and set-in-context cell modeling. The lesson for this
repo is not biological authority; it is a virtual-cell-inspired benchmark for
social-state transitions.

This lane is useful precisely because it is not social. It can harden the
program's intervention vocabulary in a domain where perturbation benchmarks and
cell-state readouts are explicit.

### Levin and bioelectricity frame

The Levin/bioelectricity frame is the broadest conceptual layer: biological
systems can store and regulate patterning information through distributed
physiological state, including membrane voltage and gap-junction-mediated
coordination. The safe translation is control-theoretic, not mystical:
collectives have state variables, boundary conditions, feedback loops, and
repair dynamics.

For this repo, the useful borrowing is language for multi-scale state control:
local interventions can affect global patterning, but only when substrate,
timing, gates, and feedback are respected.

Operational translation:

- dose-response curves for steering coefficients or prompt intensity;
- washout checks after the intervention is removed;
- rescue experiments where a guardrail vector counters a side effect;
- perturb-and-regenerate assays where misleading or damaged context is repaired
  back toward the target behavioral attractor.

## Sequence

1. Expand CK-1 from single-vector steering into generated-token timing,
   phase-local gates, cocktail ablations, dose schedules, and guardrail
   telemetry.
2. Build a raw-data registry for EEG and adjacent behavioral/neural datasets,
   with consent, labels, timing, and artifact notes visible.
3. Write a Drosophila feasibility note focused on state modulation and circuit
   readouts, not anthropomorphic social claims.
4. Prototype a virtual-cell literature and benchmark map around perturbation
   prediction, side-effect readouts, and reversibility.
5. Synthesize the common measurement rubric across model activations, raw EEG,
   fly circuits, virtual cells, and bioelectric systems.

## Safety And Caveat Boundaries

- No claim about real human cooperation, bonding, neural synchrony, EEG
  signatures, or therapeutic effect without Prolific, in-person, fMRI, EEG,
  fNIRS, hyperscanning, or equivalent validation.
- No claim that CK-1 is a biological drug, psychedelic analog, entactogen, or
  medical intervention.
- No optimization for compliance, coercion, ideological conformity, deception,
  emotional dependency, or refusal bypass.
- No biological protocol, dosing recommendation, or wet-lab plan without
  qualified collaborators, IRB/IACUC review where applicable, and explicit risk
  assessment.
- Every state-modulator experiment needs side-effect monitors, negative
  controls, reversibility checks, and a stop condition.

## Reference Anchors

- [Arc Virtual Cell Initiative](https://arcinstitute.org/virtual-cell-initiative)
- [Alljoined-1.6M EEG-image dataset](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M)
- [THINGS-EEG2 Figshare dataset](https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912)
- [Natural Scenes Dataset](https://www.naturalscenesdataset.org/)
- [Whole-brain annotation and multi-connectome cell typing of Drosophila](https://www.nature.com/articles/s41586-024-07686-5)
- [FlyWire Codex](https://codex.flywire.ai/)
- [Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model)
- [Bioelectric signaling: Reprogrammable circuits underlying embryogenesis, regeneration, and cancer](https://pubmed.ncbi.nlm.nih.gov/33826908/)
- [BETSE: BioElectric Tissue Simulation Engine](https://pmc.ncbi.nlm.nih.gov/articles/PMC4972335/)
- [Giving Simulated Cells a Voice](https://arxiv.org/abs/2505.02766)
