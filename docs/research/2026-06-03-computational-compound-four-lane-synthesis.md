---
title: 2026-06-03 Computational Compound Four-Lane Synthesis
status: proposed
date: 2026-06-03
origin: user request to kick off four parallel approaches from Spencer/Nick discussion notes
---

# 2026-06-03 Computational Compound Four-Lane Synthesis

## Decision

The next phase should split the computational-compound program into four lanes
that share one measurement vocabulary but do not collapse into one claim.

Shared vocabulary:

```text
state + perturbation + dose + site + timing + gate + side_effects + washout
```

Shared caution:

> None of these lanes proves human prosocial effects, neural synchrony,
> pharmacological action, therapeutic benefit, or biological mechanism. They are
> compute-first hypothesis-generation lanes that become stronger only when their
> own validation standards are met.

## Lane 1: LLM Pharmacology Assay

Primary doc: [2026-06-03 LLM Pharmacology Assay](../plans/2026-06-03-llm-pharmacology-assay.md)

Goal: turn CK-1 from an activation-space candidate into a proper assay with
coefficient sweeps, hook/layer placement, generated-token timing, phase gates,
side-effect monitors, cocktail ablations, and washout checks.

Nearest-term success:

- generated-token-only or decayed timing beats always-on steering;
- `ck1_full` beats positive-only steering on side effects;
- output projection, behavior score, and side-effect monitors move coherently;
- washout returns toward baseline after hook removal.

## Lane 1b: Ketamine/HNK Translation

Primary doc: [2026-06-03 Ketamine-Inspired Computational Compound Translation](2026-06-03-ketamine-computational-compound-translation.md)

Effect atlas: [2026-06-03 Ketamine-Inspired Effect Atlas](2026-06-03-ketamine-inspired-effect-atlas.md)

Goal: keep the pharmacology analogy honest. Ketamine/HNK is useful because it
forces a multi-target model: LHb burst suppression, downstream AMPA/plasticity
signals, contested opioid/reward/aversion coupling, and side-effect/antagonist
checks.

Computational translation:

- `anti_burst`: suppress runaway aversive or coercive loops;
- `constructive_throughput`: amplify safe repair and grounded uncertainty;
- `reward_coupling_monitor`: catch dependency, flattery, compliance, or
  reward-seeking drift;
- `antagonist_test`: use guardrail terms to distinguish benefit from side
  effect.

## Lane 2: Raw EEG/fMRI Bridge

Primary doc: [2026-06-03 Raw EEG/fMRI Bridge Registry](2026-06-03-raw-eeg-bridge-registry.md)

Goal: follow Spencer's advice and skip derived proxies first. Build a direct
stimulus-to-brain bridge from raw/minimally processed EEG/fMRI datasets to
visual, semantic, affective, and weak social-scene features.

Nearest-term success:

- THINGS-EEG2 encoder beats shuffled controls with visual/semantic features;
- social-scene features add only after visual, semantic, and affective controls;
- Alljoined replay quantifies whether scale compensates for noisier consumer EEG;
- no claim is made about real social cohesion or intervention effects.

## Lane 3: Drosophila Substrate

Primary doc: [2026-06-03 Drosophila Substrate Kickoff](2026-06-03-drosophila-substrate-kickoff.md)

Goal: use fly connectomes and toy dynamics as a tractable state-modulation
substrate. This is not fly prosociality and not pharmacology. It is a compact
place to test perturbation logic: graph site, transmitter-class edge scaling,
dose, off-target effects, and washout in simulation.

Nearest-term success:

- tiny FlyWire/Codex graph toy loads cleanly;
- transmitter-class edge scaling produces target/off-target metrics;
- adult LIF perturbation smoke reports activation, silencing, and washout;
- larval connectome sweep spec defines exhaustive perturbation matrices before
  full data work.

## Lane 4: Virtual Cell / Multi-Omics

Primary doc: [2026-06-03 Virtual Cell Multi-Omics Kickoff](2026-06-03-virtual-cell-multiomics-kickoff.md)

Goal: borrow perturbation-state-transition discipline from virtual-cell and
multi-omics work. Treat CK-style compounds as transition predictors with
baseline state, perturbation, observed state, side-effect drift, held-out
contexts, naive baselines, and washout.

Nearest-term success:

- define transition records for CK-1 and a placebo recipe;
- separate activation, behavior, telemetry, memory/context, interaction, and
  safety readout layers;
- require held-out contexts and naive baselines before celebrating any result;
- report side-effect drift as a first-class output.

## Integrated Next Sprint

1. Merge or rebase onto the CK-1 causal steering code from PR #37.
2. Implement the smallest CK-1 pharmacology assay: generated-token timing,
   all-vs-last hook position, low-dose grid, and side-effect report.
3. Create the raw-data access checklist for THINGS-EEG2, Alljoined, NSD,
   BOLD5000, FACED, and IAPS before downloading anything.
4. Create a Drosophila substrate registry and one tiny graph fixture.
5. Define a transition-record schema that can represent both CK-1 LLM results
   and virtual-cell-inspired perturbation maps.

## Claim Boundary

The strongest claim this four-lane phase can make is methodological:

> We now have a concrete research program for computational compounds as
> reversible, dose-controlled, phase-gated state interventions with side-effect
> monitors and washout tests across LLMs, raw human-response datasets, fly
> circuit substrates, and virtual-cell-inspired perturbation benchmarks.

Everything beyond that requires its own evidence.
