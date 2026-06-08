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
