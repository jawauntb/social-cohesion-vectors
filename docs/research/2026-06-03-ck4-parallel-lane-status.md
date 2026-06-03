---
title: 2026-06-03 CK-4 Parallel Lane Status
status: active
date: 2026-06-03
origin: CK-4 parallel phase execution note
---

# 2026-06-03 CK-4 Parallel Lane Status

## Short Status

CK-4 is now a parallel execution batch, not a single larger assay. The purpose
is to turn the ketamine-inspired effect atlas and CK-3 cocktail result into
four small compute artifacts that share the same perturbation vocabulary while
remaining claim-limited.

The shared claim boundary is unchanged: these artifacts are compute-only. They
do not establish human behavioral effects, neural effects, therapeutic effects,
drug analogs, or biological mechanisms. They are scaffolds for controlled
perturbation records, dataset provenance, and toy-substrate measurement.

## Artifacts Now Being Built

### 1. CK-4 Scheduled Cocktails

The LLM lane is extending CK-3 from same-layer, same-token cocktails into
scheduled recipes. The key change is explicit separation of:

- CK-1-like target movement;
- guardrail clamps;
- layer/site;
- generated-token phase;
- coefficient schedule;
- blocked verification or safety-critical contexts;
- washout after hooks are removed.

The first useful output is a recipe grid and report format that can say whether
target movement, output behavior, side-effect monitors, and washout agree. A
scheduled cocktail is not promoted because it resembles any biological account;
it is promoted only if it beats baseline and guardrail-only controls under the
existing CK assay gates.

### 2. Transition Records

The transition-record lane is turning CK-style runs into lane-agnostic records:

```text
baseline_state + perturbation + dose + site + timing + gate
+ observed_transition + side_effects + antagonist + washout
+ replication_context
```

The immediate artifact is a small schema and at least one filled record from
CK-3 or CK-4. Its job is bookkeeping discipline: preserve what was changed,
where it was changed, what moved, what failed, and what returned toward
baseline. The record should be generic enough to later describe LLM activation
runs, raw EEG/fMRI encoding checks, Drosophila graph toys, and virtual-cell
inspired perturbation benchmarks without pretending those substrates are the
same thing.

### 3. Raw EEG Checklist

The raw EEG/fMRI lane is not running a brain-effect claim. It is preparing an
access and provenance checklist before any data pull or encoder training.

The checklist should cover THINGS-EEG2, Alljoined, THINGS-EEG1, NSD, BOLD5000,
FACED, and IAPS with:

- access terms and license notes;
- raw versus preprocessed availability;
- stimulus identity and stimulus-rights status;
- subject/session/trial metadata availability;
- intended first use;
- release restrictions for derivatives, encoders, or manifests.

The first valid outcome is permission to build a tiny stimulus-feature manifest
for THINGS-EEG2. It is not evidence about prosociality, social cohesion,
intervention effects, or neural synchrony.

### 4. Drosophila Toy Substrate

The Drosophila lane is building a compact graph/dynamics toy substrate. The
purpose is to test perturbation grammar outside text: site, dose,
transmitter-class edge scaling, target movement, off-target movement, and
washout.

The first useful artifact is a tiny synthetic or provenance-safe graph fixture
plus a transmitter-class scaling sweep. Expected outputs are target shift,
off-target shift, instability flags, and return-to-baseline behavior after the
perturbation is removed. This is not fly prosociality, not pharmacology, and not
a biological drug-effect claim.

## Coordination Rule

All four artifacts should report enough fields to be convertible into a
transition record. That is the shared spine of the CK-4 batch. The artifacts can
move in parallel because success is local and falsifiable:

- scheduled cocktails must beat their own LLM controls;
- transition records must preserve complete perturbation provenance;
- the raw EEG checklist must block unlicensed or under-specified data use;
- the Drosophila toy must expose target/off-target/washout behavior in a small
  reproducible substrate.

The strongest near-term claim is methodological: CK-4 is separating timing,
site, provenance, and substrate-specific readouts so that future runs can fail
cleanly instead of collapsing into one broad analogy.
