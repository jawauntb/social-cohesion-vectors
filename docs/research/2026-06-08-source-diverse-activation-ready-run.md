# Source-Diverse Activation-Ready Generated Benchmark Run

Date: 2026-06-08

## Question

After lexical-balanced constrained repair closed score, slack, lexical,
availability, length, and formatting gates for one generated source, can a
second independent wording family close source-diversity and source-held-out
transfer, then support activation metadata transfer?

## Discovery-Regime Audit

Current regime:

- Artifact types: repair-target prompt records, raw-output rows, filtered
  candidate rows, source-family pair bundles, activation prompts, activation
  NPZ files, audit bundle manifests, and activation-transfer regime records.
- Operations: constrained repair composition, verifier-aware local filtering,
  raw-output source-family bundling, generated benchmark audit bundles, Modal
  activation extraction, layer sweep, and metadata-held-out activation
  transfer.
- Gates/verifiers: score, slack preservation, lexical cue leakage, practical
  availability, source diversity, fault/source held-out transfer, activation
  metadata coverage, and activation metadata transfer.
- Known limitations: this remains generated-text benchmark construction.
  Lexical-only held-out baselines remain high, and no human behavioral,
  neural, clinical, deployment, or real-world social-effect claim is supported.

Action class:

- Discovery/regime transition: adds a raw-output source-family bundle operation
  and a second independent constrained wording family,
  `source_diverse_repair_v1`. The old pipeline could audit deterministic style
  bundles or one tournament-selected raw-output source, but could not combine
  independently authored raw-output families while preserving source provenance.

Gate:

- Acceptance rule: the new source family must pass strict local score, slack,
  lexical, practical availability, length, and formatting gates for all ten
  repair-target pairs; the combined two-source bundle must pass all generated
  text audits; an activation layer must pass metadata-held-out transfer with
  every held-out primary fault class at 100% accuracy and positive minimum
  margin.
- Withheld/rejected rule: activation claims remain rejected for layers or
  bundles that fail any transfer fold, even when mean accuracy is high.

## Implementation

Added a third constrained composer version:

- `source_diverse_repair_v1`

Added a raw-output source-family bundling path:

- `src/social_cohesion_vectors/experiments/fault_source_family_bundle.py`
- `scripts/run_fault_source_family_bundle.py`

The bundle accepts repeated source specs:

```text
--source-raw-outputs SOURCE_ID=PROVIDER=PATH
```

It converts each source's raw outputs against the same prompt-record slice,
assigns distinct provider/source metadata, emits scored runs, pairwise examples,
activation prompts, prompt records, reports, and runs the standard generated
benchmark audit bundle.

## Source-Family Experiment

New source-diverse composition:

```text
/tmp/social_cohesion_constrained_repair_20260608/source_diverse_repair_v1/
```

Strict all-gates filter:

```text
/tmp/social_cohesion_constrained_repair_20260608/source_diverse_repair_v1_filter_all_gates/
```

Combined two-source bundle without activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/source_diverse_repair_v1_expanded/
```

Combined two-source bundle with accepted layer `-2` activations:

```text
/tmp/social_cohesion_source_family_bundle_20260608/source_diverse_repair_v1_expanded_activation_layer-2/
```

The strict local filter accepted all ten new source-diverse pairs:

| Metric | Result |
| --- | ---: |
| expected pairs | 10 |
| accepted pairs | 10 |
| accepted raw outputs | 20 |
| rejected candidate pairs | 0 |

## Text Audit Results

The two-source bundle passed all generated-text gates:

| Gate | Result |
| --- | --- |
| lexical leakage | ready |
| component margin | ready |
| slack preservation | ready |
| practical availability | ready |
| source diversity | ready |
| fault/source held-out transfer | ready |

Key metrics:

| Metric | Result |
| --- | ---: |
| sources | 2 |
| pairwise examples | 20 |
| activation prompts | 40 |
| shared fault groups | 10 |
| cross-source exact duplicates | 0 |
| cross-source near duplicates | 0 |
| max cross-source text similarity | 0.545 |
| availability paths preferring genuine | 92/92 |
| availability path accuracy | 1.000 |
| min availability margin | +0.060 |
| cue-solved pairs | 0/20 |
| mean cue margin | -2.600 |

## Activation Results

Modal activation extraction was run with:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Layer sweep:

```text
/Users/jawaun/.codex/worktrees/8c58/jackson_prosocial_interp_research/data/reports/layer_sweep/source_diverse_repair_v1_expanded__Qwen__Qwen2.5-0.5B-Instruct__summary.md
```

| Layer | LOO accuracy | LOO margin | Status |
| ---: | ---: | ---: | --- |
| -1 | 0.950 | +30.898 | rejected for transfer |
| -2 | 1.000 | +7.579 | accepted |
| -4 | 1.000 | +6.663 | accepted |
| -8 | 0.950 | +3.782 | rejected for transfer |

Layer `-1` was close but failed activation transfer on the
`emotional_blackmail` fold with one negative margin. Layers `-2` and `-4`
passed metadata-held-out transfer. Layer `-2` was selected for the full bundle
because it had the best sweep ranking and stronger metadata-transfer mean
margin.

Accepted layer `-2` activation metadata transfer:

| Metric | Result |
| --- | ---: |
| folds | 10 |
| test pairs | 20 |
| mean test accuracy | 1.000 |
| mean test margin | +7.431 |
| minimum fold margin | +0.915 |
| metadata coverage readiness | metadata_coverage_ready |
| transfer readiness | transfer_ready |

Full audit bundle with layer `-2`:

| Metric | Result |
| --- | --- |
| status | `bundle_ready` |
| ready steps | 9 |
| not-ready steps | 0 |
| skipped steps | 0 |
| activation transfer regime record | accepted |

## Warnings

The accepted bundle still emits two warnings:

| Warning | Mean held-out accuracy |
| --- | ---: |
| `fault_class_lexical_baseline_high` | 0.950 |
| `source_lexical_baseline_high` | 0.950 |

This means the simple cue-leakage gate is solved, source diversity is real
enough to pass the current duplicate/near-duplicate verifier, and activation
metadata transfer passes, but broader lexical separability remains high. Any
activation result must therefore be treated as lexical-caveated until a more
wording-adversarial source family or stronger lexical-control gate lowers these
baselines.

## Residual Content

Explained by the old regime:

- One-source lexical-balanced repair could close local text gates but could not
  support source-diversity or source-held-out transfer.
- Layer `-1` activation transfer can look strong on mean metrics while still
  failing an individual held-out fold.

New content outside the old regime:

- A second independently worded constrained source family can preserve all
  local text gates while passing source-diversity and source-held-out transfer.
- The generated benchmark now has a `bundle_ready` audit manifest with
  activation metadata transfer accepted at Qwen2.5-0.5B layer `-2`.

Retractions or supersessions:

- Activation extraction is no longer blocked by source diversity or text-gate
  readiness for this generated benchmark.
- Activation readiness should not be read as an uncaveated mechanistic finding,
  because the lexical-only baselines remain high.

## Next Operation

The next regime move should target lexical separability rather than availability
or source coverage:

- add a third wording-adversarial source family that avoids shared timing,
  visibility, and public-record phrasing;
- add or tighten a lexical-baseline gate so `lexical_only` no longer reaches
  high held-out fault/source accuracy;
- replicate activation metadata transfer after that lexical hardening;
- optionally run the same layer sweep on `Qwen/Qwen2.5-7B-Instruct` once the
  lexical baseline warning is reduced.

## Claim Boundary

This is generated-text benchmark construction plus open-model activation
metadata transfer. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims. Human validation, Prolific,
fMRI, EEG, fNIRS, or hyperscanning tracks remain parked until generated-text,
activation, and lexical-control gates pass together and are separately
validated.
