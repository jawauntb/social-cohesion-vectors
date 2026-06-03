---
title: 2026-06-03 Drosophila Toy Matrix Results
status: implemented
date: 2026-06-03
origin: CK6 parallel batch workstream 3
---

# 2026-06-03 Drosophila Toy Matrix Results

This note records the first concrete output from the synthetic toy substrate
matrix. It uses the existing transmitter-by-coefficient machinery and the
transition-record exporter. Generated JSON and JSONL artifacts were written to
`/tmp` only.

## Boundary

This is a synthetic toy graph result. It is not fly biology, not pharmacology,
not behavior, not a neural claim, and not evidence for any organism-level or
therapeutic effect. The transmitter labels are toy edge labels used to exercise
matrix reporting and transition-record plumbing.

## Smoke Commands

```bash
uv run python scripts/run_drosophila_transmitter_matrix.py \
  --json-report-output /tmp/drosophila_transmitter_matrix_ck6.json \
  --markdown-report-output /tmp/drosophila_transmitter_matrix_ck6.md \
  --jsonl-output /tmp/drosophila_transmitter_matrix_ck6.jsonl

uv run python scripts/export_ck3_transition_records.py \
  /tmp/drosophila_transmitter_matrix_ck6.json \
  --output /tmp/drosophila_transmitter_matrix_ck6_transitions.jsonl
```

## Matrix Result

The exported report is
`/tmp/drosophila_transmitter_matrix_ck6.json`.

- Experiment: `drosophila_toy_substrate_transmitter_matrix`
- Graph: `drosophila_toy_substrate_v0`
- Matrix shape: `6 x 5`
- Matrix rows: `30`
- Transmitter labels: acetylcholine, GABA, glutamate, dopamine, serotonin,
  octopamine
- Coefficients: `0.0`, `0.5`, `1.0`, `1.5`, `2.0`
- Unstable runs: `0`

The `1.0` coefficient is the neutral toy baseline for each transmitter group.
Ranking excludes the neutral baseline when non-baseline rows are present.

## Ranked Rows

Top target movement and top selectivity were the same row in this fixture:

| Rank view | Transmitter | Coefficient | Scaled edges | Target movement | Off-target movement | Selectivity | Washout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target movement | acetylcholine | 2.0 | 4 | 0.397978 | 0.020976 | 18.972607 | 0.005102 |
| Selectivity | acetylcholine | 2.0 | 4 | 0.397978 | 0.020976 | 18.972607 | 0.005102 |
| Lowest washout | octopamine | 0.5 | 1 | -0.010025 | 0.001708 | 5.869267 | 0.000214 |

Per-transmitter best-target summaries:

| Transmitter | Runs | Scaled edges | Best coefficient | Abs target movement | Off-target at best | Washout at best | Selectivity at best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acetylcholine | 5 | 4 | 2.0 | 0.397978 | 0.020976 | 0.005102 | 18.972607 |
| dopamine | 5 | 1 | 0.0 | 0.010182 | 0.014666 | 0.001779 | 0.694266 |
| GABA | 5 | 2 | 0.0 | 0.027658 | 0.003971 | 0.000961 | 6.964891 |
| glutamate | 5 | 2 | 0.0 | 0.011589 | 0.005990 | 0.000757 | 1.934742 |
| octopamine | 5 | 1 | 0.0 | 0.020049 | 0.003416 | 0.000428 | 5.869267 |
| serotonin | 5 | 2 | 2.0 | 0.010546 | 0.012543 | 0.001828 | 0.840739 |

## Transition Records

The transition exporter consumed the matrix JSON report and wrote
`/tmp/drosophila_transmitter_matrix_ck6_transitions.jsonl`.

- Transition records: `24`
- Effect classes: `mixed_transition=24`
- Side-effect status: `observed=24`
- Washout status: `residual=24`

These transition records pair each non-neutral coefficient with its
transmitter-matched `1.0` baseline. The side-effect and washout fields are toy
graph readouts only: `observed` means nonzero off-target movement or toy
instability in the graph metric, and `residual` means the toy washout distance
remained above the exporter threshold.
