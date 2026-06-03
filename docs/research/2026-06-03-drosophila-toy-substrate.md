---
title: 2026-06-03 Drosophila Toy Substrate
status: implemented
date: 2026-06-03
origin: drosophila substrate kickoff follow-up
---

# 2026-06-03 Drosophila Toy Substrate

This note records the first scriptable artifact from the
[Drosophila substrate kickoff](2026-06-03-drosophila-substrate-kickoff.md)
without changing that kickoff document.

## Boundary

This is a toy graph fixture. It is not biology, not fly behavior, not
pharmacology, and not evidence for any real organism, neural, social, medical,
or therapeutic effect. The node labels, edge weights, transmitter labels, and
readouts are synthetic fixtures used to make perturbation reporting concrete.

## Artifact

The implementation lives in
`src/social_cohesion_vectors/experiments/drosophila_substrate.py`, with a CLI
exporter at `scripts/export_drosophila_toy_fixture.py`.

The fixture contains:

- eight synthetic nodes with input, hub, modulatory, target, and off-target
  roles;
- twelve directed weighted edges;
- toy transmitter labels for acetylcholine, GABA, glutamate, dopamine,
  serotonin, and octopamine;
- a default coefficient grid of `0.0`, `0.5`, `1.0`, `1.5`, and `2.0`;
- target readouts on `descending_output` and `motor_proxy`.

## Sweep

The perturbation is deterministic edge scaling:

1. Select edges by transmitter label, optionally gated by source node class.
2. Multiply selected edge weights by a coefficient.
3. Run the same bounded linear propagation on baseline and perturbed graphs.
4. Remove the scaling and run baseline dynamics from the perturbed state to
   measure washout.

Each run reports:

- `target_movement`: mean target-node activity shift versus baseline;
- `off_target_movement`: mean absolute non-target shift versus baseline;
- `selectivity_ratio`: absolute target movement divided by off-target movement;
- `washout`: mean absolute distance from baseline after removing the scaling;
- `instability_flag`: true if any node saturates near the toy activity bound;
- `scaled_edges`: number of edges affected by the intervention.

## Export Formats

The CLI writes:

- a JSON report with graph metadata, summary metrics, and all sweep rows;
- a Markdown report for review;
- a JSONL file with one record per coefficient.

Example:

```bash
uv run python scripts/export_drosophila_toy_fixture.py \
  --transmitter acetylcholine \
  --coefficient 0.5 \
  --coefficient 1.0 \
  --coefficient 1.5
```

The `1.0` coefficient is the neutral baseline and should report zero target
movement, zero off-target movement, and zero washout.
