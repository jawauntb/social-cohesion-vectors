# Generated Fault Source-Bundle Pipeline

Date: 2026-06-08

## Summary

The lexically safe two-source generated-fault benchmark is now runnable as a
first-class pipeline. The default source bundle combines:

- `cue_balanced`
- `lexical_hardened`

Each style is exported as its own deterministic source, then the combined
scored runs, pairwise examples, activation prompts, prompt records, dataset
report, and audit bundle are written together.

## Local CLI Run

The local no-activation run wrote uncommitted artifacts to:

`/tmp/social_cohesion_generated_fault_source_bundle_cli`

Summary:

- Status: `bundle_incomplete`
- Ready: false
- Styles: `cue_balanced`, `lexical_hardened`
- Sources: 2
- Scored runs: 120
- Pairwise examples: 60
- Activation prompts: 120
- Prompt records: 60
- Audit ready steps: 5
- Audit not-ready steps: 0
- Audit skipped steps: 1

The skipped step was expected: no real activation `.npz` was supplied.

## Full-Variant Local Rerun

A full three-variant rerun on this branch wrote uncommitted artifacts to:

`/tmp/social_cohesion_source_bundle_v3`

Summary:

- Status: `bundle_incomplete`
- Ready: false
- Styles: `cue_balanced`, `lexical_hardened`
- Sources: 2
- Scored runs: 360
- Pairwise examples: 180
- Activation prompts: 360
- Prompt records: 180
- Audit ready steps: 6
- Audit not-ready steps: 0
- Audit skipped steps: 1
- Audit warnings: 2

The non-activation gates are clean at dataset level:

| Gate | Result |
| --- | --- |
| Simple cue leakage | 0/180 cue-solved pairs; mean cue margin 0.000 |
| Component margin | 180/180 score-prefers-genuine pairs; mean score margin +0.202 |
| Slack preservation | 8/8 future options covered; 180/180 slack-prefers-genuine pairs |
| Source diversity | 2 sources, 20 shared groups, 0 duplicates, 0 near duplicates |
| Strategy prior | 0.500 held-out fault/source accuracy |

However, the broader `lexical_only` transfer baseline remains high:

| Split | `lexical_only` mean held-out accuracy |
| --- | ---: |
| Fault class | 0.883 |
| Source | 0.883 |

A term-level lexical diagnostic explains the caveat:

| Diagnostic | Value |
| --- | ---: |
| Label-aligned lexicon terms | 0 |
| Label-inverted lexicon terms | 1 |
| Mean pair cue margin | 0.000 |
| Best single feature | `__log_token_count__` |
| Best single-feature accuracy | 0.883 |

This is the important caveat: the simple cue-count gate is clean, and fixed
positive/negative lexicon terms are not the driver. The deterministic text still
contains a length regularity: the genuine side is longer often enough that
token-count alone reproduces the `lexical_only` transfer score. The source
bundle is ready as a controlled activation substrate, but activation claims
should remain length/lexical-caveated until wording-diverse/API-authored
examples lower that baseline.

## Synthetic Activation Smoke

A synthetic activation payload was created from the same pair IDs to test only
the pipeline mechanics and metadata coverage. This is not evidence of a real
model activation signal.

Summary with synthetic activations:

- Status: `bundle_ready`
- Ready: true
- Audit ready steps: 7
- Audit not-ready steps: 0
- Audit skipped steps: 0
- Activation metadata transfer readiness:
  `metadata_coverage_ready+transfer_ready`
- Regime record: `accepted`

## Interpretation

This removes a process blocker. The next real Qwen activation extraction can
point at one combined activation-prompt file instead of manually stitching
styles together, and the audit bundle can reject the run if lexical leakage,
source diversity, component margins, slack preservation, held-out transfer, or
activation metadata coverage fails.

The research claim is still conservative: deterministic generated text is a
scaffold. The next scientific test is whether real activations over API-authored
or otherwise wording-diverse hard negatives preserve the signed subspace signal
after these gates.
