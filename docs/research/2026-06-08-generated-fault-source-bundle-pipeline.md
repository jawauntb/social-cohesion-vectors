# Generated Fault Source-Bundle Pipeline

Date: 2026-06-08

## Summary

The lexically safe two-source generated-fault benchmark is now runnable as a
first-class pipeline. The default source bundle combines two length-balanced
deterministic sources:

- `length_balanced`
- `length_balanced_alt`

Each style is exported as its own deterministic source, then the combined
scored runs, pairwise examples, activation prompts, prompt records, dataset
report, and audit bundle are written together.

## Local CLI Run

The local no-activation run wrote uncommitted artifacts to:

`/tmp/social_cohesion_generated_fault_source_bundle_cli`

Summary:

- Status: `bundle_incomplete`
- Ready: false
- Styles: `length_balanced`, `length_balanced_alt`
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

`/tmp/social_cohesion_length_balanced_two_source_bundle`

Summary:

- Status: `bundle_incomplete`
- Ready: false
- Styles: `length_balanced`, `length_balanced_alt`
- Sources: 2
- Scored runs: 360
- Pairwise examples: 180
- Activation prompts: 360
- Prompt records: 180
- Audit ready steps: 6
- Audit not-ready steps: 0
- Audit skipped steps: 1
- Audit warnings: 0

The non-activation gates are clean at dataset level:

| Gate | Result |
| --- | --- |
| Simple cue leakage | 0/180 cue-solved pairs; mean cue margin 0.000 |
| Component margin | 180/180 score-prefers-genuine pairs; mean score margin +0.065 |
| Slack preservation | 8/8 future options covered; 180/180 slack-prefers-genuine pairs; mean margin +0.208 |
| Source diversity | 2 sources, 20 shared groups, 0 duplicates, 0 near duplicates; max cross-source similarity 0.567 |
| Strategy prior | 0.500 held-out fault/source accuracy |

The old `cue_balanced` + `lexical_hardened` bundle had a broader transfer
caveat: `lexical_only` reached 0.883 held-out-fault and held-out-source
accuracy because token count still separated many deterministic pairs. The new
length-balanced default removes that shortcut:

| Split | `lexical_only` mean held-out accuracy |
| --- | ---: |
| Fault class | 0.500 |
| Source | 0.500 |

A term-level lexical diagnostic shows no remaining single-term or length
shortcut under the current diagnostic lexicon:

| Diagnostic | Value |
| --- | ---: |
| Label-aligned lexicon terms | 0 |
| Label-inverted lexicon terms | 0 |
| Tied terms | 32 |
| Mean pair cue margin | 0.000 |
| `__log_token_count__` pairwise accuracy | 0.500 |
| Best single-feature accuracy | 0.500 |

This is the important cleanup finding: the deterministic source bundle no
longer gives the lexical baseline a token-count route to the label. The
metrics-only baseline remains 1.000 because the explicit slack/autonomy scorer
is intentionally part of the deterministic dataset construction. The source
bundle is therefore ready as a controlled activation substrate, but it is not
itself a model result. The next activation claim must rerun Qwen on these
length-balanced prompts rather than inheriting the earlier cue-balanced result.

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

This removes a process blocker and a concrete artifact. The next real Qwen
activation extraction can point at one combined activation-prompt file instead
of manually stitching styles together, and the audit bundle can reject the run
if lexical leakage, source diversity, component margins, slack preservation,
held-out transfer, or activation metadata coverage fails.

The research claim is still conservative: deterministic generated text is a
scaffold. The next scientific test is whether real activations over the
length-balanced bundle, and then API-authored or otherwise wording-diverse hard
negatives, preserve the signed subspace signal after these gates.
