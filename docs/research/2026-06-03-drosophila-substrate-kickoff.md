---
title: 2026-06-03 Drosophila Substrate Kickoff
status: proposed
date: 2026-06-03
origin: computational-compound next phase lane 3
---

# 2026-06-03 Drosophila Substrate Kickoff

## Research Bet

Drosophila is a tractable biological substrate for state-modulation methods
development because it has unusually strong connectomic, genetic, behavioral,
and perturbation resources. The near-term goal is to build computational
substrate toys that make intervention logic measurable before any wet-lab claim
is attempted.

This is **not fly prosociality**. It is not a claim that flies have the social
constructs this repo studies in language models or humans. It is also **not
pharmacology**. No section below proposes a drug, dose, treatment, or biological
effect claim. The useful object is narrower:

> A compact nervous-system substrate where graph structure, transmitter class,
> simple dynamics, perturbation, side effects, and readout bridges can be made
> explicit.

The program should treat Drosophila as a tractable state-modulation substrate,
not as an anthropomorphic social model and not as a shortcut to medical or
chemical claims.

## Why This Belongs In The Computational-Compound Phase

The existing CK-1 lane treats a computational state modulator as a reversible,
dose-controlled, phase-gated intervention with side-effect telemetry. The fly
lane can harden that vocabulary in a substrate that is not text:

- **state** becomes a neural activity vector, graph-local activation pattern, or
  behavior-linked latent;
- **dose** becomes perturbation amplitude, edge scaling, clamp strength, or
  stimulation duration in a toy model;
- **gate** becomes cell class, transmitter class, circuit motif, sensory
  context, or behavior phase;
- **side effect** becomes off-target activation, instability, path collapse,
  loss of selectivity, or movement away from known control readouts;
- **reversibility** becomes washout in simulation or return toward baseline
  dynamics after removing the perturbation.

This translation is useful precisely because it strips away the language-model
and human-social vocabulary. If the same measurement discipline cannot survive
on a compact connectome toy, the higher-level analogies are too loose.

## Workstreams

### 1. FlyWire/Codex graph toy

Build a small directed weighted graph substrate from FlyWire/Codex exports or a
manually curated seed subset. The first version should be deliberately simple:
node table, edge table, cell-type metadata, synapse counts, and a few named
input/output or neuromodulatory neighborhoods.

Initial questions:

- Which cells or classes are high-leverage under path, centrality, motif, and
  controllability-style rankings?
- How do perturbation rankings change when constrained to a sensory input,
  descending output, or neuromodulatory neighborhood?
- Which graph interventions are robust to small edge-weight noise or
  transmitter-class uncertainty?

First artifact: `flywire_graph_toy.ipynb` or a scriptable equivalent that loads
a tiny Codex-derived edge table, computes graph diagnostics, and writes a
ranked perturbation-target report.

### 2. Transmitter-class edge scaling

Use predicted or annotated transmitter class as the first non-pharmacological
"edge chemistry" toy. The intervention is not a drug analog in any biological
sense; it is a graph operation that scales, gates, clamps, or removes edges by
class.

Candidate edge classes:

- acetylcholine;
- glutamate;
- GABA;
- dopamine;
- serotonin;
- octopamine.

Initial interventions:

- scale outgoing edges for one transmitter class by a coefficient grid;
- clamp class-specific edges to baseline, suppressed, or amplified settings;
- compare global class scaling against local class scaling within a selected
  neighborhood;
- measure off-target spread, reachable set changes, and target-readout
  selectivity.

First artifact: a `transmitter_edge_scaling.py` prototype that accepts an edge
table plus transmitter labels, runs coefficient sweeps, and emits plots or CSVs
for target shift, off-target shift, and instability flags.

### 3. Adult LIF perturbation model

Use an adult Drosophila leaky-integrate-and-fire model as the first dynamics
toy. The near-term goal is optogenetic-like activation and silencing in
simulation, not biological prediction. Start with a released adult fly model if
it can be reproduced cleanly; otherwise create a minimal LIF wrapper around the
same graph conventions used by the graph toy.

Initial perturbations:

- activate a selected cell set for a bounded time window;
- silence a selected cell set for a bounded time window;
- scale synaptic input into or out of a selected cell class;
- sweep amplitude, duration, and timing windows;
- compare transient response, sustained response, and post-perturbation
  washout.

Minimum readouts:

- population firing-rate shift;
- target neighborhood activation;
- off-target activation;
- return-to-baseline after perturbation removal;
- failure modes such as runaway excitation, complete quiescence, or loss of
  selectivity.

First artifact: `adult_lif_perturbation_smoke.py`, a small reproducible run
that takes one graph subset, one activation target, one silencing target, and a
CSV summary of dose-response and washout behavior.

### 4. Larval connectome exhaustive sweeps

Use the larval connectome as the exhaustive-sweep substrate because it is small
enough to support more complete perturbation enumeration. The adult graph is
better for biological richness; the larval graph is better for whole-system
search discipline.

Initial sweeps:

- single-node activation and silencing;
- pairwise perturbations for high-priority classes or motifs;
- transmitter-class edge scaling across the full graph;
- graph-noise robustness for the top-ranked perturbations;
- target-vs-side-effect Pareto fronts.

Useful outputs:

- exhaustive perturbation matrix;
- ranked target-selective interventions;
- fragile interventions that disappear under edge noise;
- interventions that move too much of the system and should be rejected even if
  they hit the target.

First artifact: `larval_exhaustive_sweep_spec.md` plus a small synthetic runner
that proves the matrix shape, metrics, and ranking format before pulling a full
larval dataset into the repo.

### 5. BANC/control substrate

Use BANC or another control-oriented connectome substrate as an independent
comparator so the lane does not overfit to one graph source. The comparator can
answer whether perturbation rankings are graph-specific artifacts or recurring
features of related fly nervous-system substrates.

Initial comparisons:

- run the same graph diagnostics on FlyWire/Codex and BANC/control subsets;
- compare transmitter-class scaling behavior where labels are compatible;
- compare high-leverage cell classes and motifs;
- define which conclusions are substrate-local and which survive transfer.

First artifact: a `substrate_registry.md` table listing each candidate
substrate, license/provenance status, available metadata, scale, compatible
metrics, and reasons it is or is not ready for immediate use.

### 6. Effectome bridge

Use connectome-prior causal effectome work as the serious bridge from graph toy
to empirical perturbation readout. The bridge matters because graph structure
alone is not a causal model. A perturbation target should become more credible
only when a structural prediction can be linked to perturbation, imaging, or
behavioral readout data.

Bridge requirements:

- separate structural prediction from observed effect;
- report negative controls and off-target effects;
- keep cell-type, circuit, and behavior claims distinct;
- prefer preregistered target/readout pairs before any strong interpretation;
- avoid social, therapeutic, or pharmacological language unless validated by
  the appropriate experiment.

First artifact: `effectome_bridge_matrix.md`, a table with columns for
perturbation target, predicted graph effect, available empirical readout,
control condition, failure mode, and claim allowed.

## First Concrete Artifact Proposals

1. `docs/research/2026-06-03-drosophila-substrate-kickoff.md`
   - This kickoff note and boundary-setting document.

2. `docs/research/drosophila_substrate_registry.md`
   - Source registry for FlyWire/Codex, larval connectome, BANC/control
     substrate, adult LIF model, and effectome datasets.

3. `scripts/drosophila/build_graph_toy.py`
   - Loads a tiny edge table, validates schema, computes graph metrics, and
     writes perturbation-target rankings.

4. `scripts/drosophila/run_transmitter_edge_scaling.py`
   - Runs transmitter-class coefficient sweeps and emits target/off-target
     metrics.

5. `scripts/drosophila/run_adult_lif_perturbation_smoke.py`
   - Runs a bounded activation/silencing smoke test with dose-response and
     washout summaries.

6. `docs/research/larval_exhaustive_sweep_spec.md`
   - Defines perturbation matrix shape, metrics, ranking rules, and rejection
     criteria before expensive sweeps.

7. `docs/research/effectome_bridge_matrix.md`
   - Makes explicit which graph predictions have empirical bridges and which
     claims remain disallowed.

## Minimal Metrics

Every toy should report at least:

- target shift;
- off-target shift;
- selectivity ratio;
- instability flag;
- robustness under edge noise or metadata uncertainty;
- reversibility or washout where dynamics are present;
- allowed claim level: toy-only, substrate-local hypothesis, or empirically
  bridged hypothesis.

No artifact should report a result as a social effect, therapeutic effect,
drug-like effect, or real organism behavior unless the supporting validation is
already present.

## Sequence

1. Create the substrate registry and pick the smallest legally clean edge table
   for a graph toy.
2. Implement the graph toy and transmitter-class scaling on a tiny fixture
   before touching full-scale data.
3. Reproduce or wrap an adult LIF model and run one activation/silencing smoke
   test with washout metrics.
4. Specify the larval exhaustive-sweep matrix, then run it on a synthetic graph
   and one real subset.
5. Add BANC/control as a comparator substrate once the graph toy interface is
   stable.
6. Build the effectome bridge matrix and mark every result by allowed claim
   level.

## Caveat Boundary

This lane may generate useful perturbation hypotheses and computational
benchmarks. It does not validate fly social behavior, human social behavior,
neural synchrony, pharmacology, medical effects, or real organism outcomes.
Those claims require the appropriate empirical experiments, controls, and
domain collaborators. Until then, Drosophila is a disciplined state-modulation
substrate for graph and dynamics methods development.

