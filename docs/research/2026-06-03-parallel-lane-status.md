---
title: 2026-06-03 Parallel Lane Status
status: active
date: 2026-06-03
origin: user asked how the other three parallel directions are going
---

# 2026-06-03 Parallel Lane Status

## Summary

The four-lane program is intact on `main`. CK-3 is the only lane that has now
run a new compute assay. The other three lanes are scoped as kickoff registries
and are ready for their first concrete artifacts.

Shared guide:

- [Ketamine-Inspired Effect Atlas](2026-06-03-ketamine-inspired-effect-atlas.md)
  now names the neural/network-inspired effect classes, their
  mechanistic-interpretability analogues, and the assay gates that should govern
  CK-4 and later transition-record work.
- [CK-4 Parallel Lane Status](2026-06-03-ck4-parallel-lane-status.md)
  records the active execution batch across scheduled cocktails, transition
  records, raw EEG/fMRI provenance, and the toy Drosophila substrate.
- [CK-5 Parallel Execution Status](2026-06-03-ck5-parallel-execution-status.md)
  tracks the current five-task batch: CK-4 Modal run prep, transition-record
  upgrades, Drosophila matrix, raw EEG manifest builder, and optional SAE env
  cleanup.

## Lane 1: LLM Pharmacology / CK Assay

Status: running.

The CK-2 single-vector assay was weak. CK-3 added a real cocktail runner and
ran a five-recipe grid. The best recipe was `guardrails_only`, not CK-1 plus
guardrails, which suggests the guardrail axes are useful but the CK-1 agonist
does not yet compose cleanly at the same layer and timing.

Next artifact: CK-4 site/timing separation, with CK-1 and guardrails applied at
different layers or generation phases.

The effect atlas turns this into a concrete split-site/split-timing grid:
early/mid CK-1-like target movement, later/final-layer guardrail clamps,
contraindication prompts, antagonist challenges, and washout.

Current artifact:

- [CK-4 Scheduled Cocktail Assay](2026-06-03-ck4-scheduled-cocktail-assay.md)
  adds component schedules such as `first-N`, `after-N`, `decay-N`, and
  `ramp-A-B` to the existing CK-3 cocktail runner.

## Lane 2: Raw EEG/fMRI Bridge

Status: designed, not yet run.

The registry recommends skipping Tribe as the first bridge and starting with
raw or minimally processed stimulus-response data. The first concrete run
should be a THINGS-EEG2 stimulus-feature encoder before Alljoined scale-up.

Current assets:

- [Raw EEG/fMRI Bridge Registry](2026-06-03-raw-eeg-bridge-registry.md)
- [Raw EEG/fMRI Dataset Checklist](2026-06-03-raw-eeg-dataset-checklist.md)

Next artifact:

- a dataset-access/provenance checklist for THINGS-EEG2, Alljoined, NSD,
  BOLD5000, FACED, and IAPS;
- then a tiny manifest builder for THINGS-EEG2 image ids, feature paths, and
  EEG epoch metadata.

Claim boundary: no social-cohesion or human-intervention claim. This lane is
representation-learning infrastructure.

## Lane 3: Drosophila Substrate

Status: designed, not yet run.

The Drosophila lane is framed as a tractable perturbation substrate, not fly
prosociality and not pharmacology. The useful object is a graph/dynamics toy
where state, dose, site, side effects, and washout can be measured without
language-model confounds.

Current assets:

- [Drosophila Substrate Kickoff](2026-06-03-drosophila-substrate-kickoff.md)
- [Drosophila Toy Substrate](2026-06-03-drosophila-toy-substrate.md)

Next artifact:

- a tiny graph fixture with nodes, edges, transmitter labels, and target
  readouts;
- a transmitter-class edge-scaling sweep that reports target movement,
  off-target movement, and washout on a synthetic or small curated graph.

Claim boundary: no biological drug effect, no fly social-behavior claim.

## Lane 4: Virtual Cell / Multi-Omics

Status: designed, not yet run.

The virtual-cell lane contributes the evaluation grammar: baseline state,
perturbation, predicted transition, observed transition, side-effect drift,
held-out contexts, naive baselines, and washout. It is not saying LLM
activations are cells; it borrows perturbation-response discipline.

Current assets:

- [Virtual Cell Multi-Omics Kickoff](2026-06-03-virtual-cell-multiomics-kickoff.md)
- [Transition Record Schema](2026-06-03-transition-record-schema.md)

Next artifact:

- a transition-record schema that can represent CK-3 recipes and later raw EEG,
  fly graph, or virtual-cell perturbation records;
- a CK-3-to-transition-record exporter using the assay report as the first
  filled example;
- an atlas-compatible transition field for `effect_class`, so results can be
  compared across LLM, raw EEG/fMRI, fly graph, and virtual-cell-inspired lanes.

Claim boundary: no cellular or biological claim. This is benchmark grammar.
