# Lexical-Hardened Two-Source Bundle

Date: 2026-06-08

## Summary

The generated-fault benchmark now has a second deterministic source style that
is designed to be lexically quiet while still preserving the slack-preservation
contrast. This matters because the prior two-source scaffold proved that source
labels and fault coverage can pass while lexical leakage still blocks activation
readiness.

Local comparison over one generated-fault variant:

| Condition | Pairs | Sources | Lexical cue-solved rate | Source-diversity ready | Bundle status |
| --- | ---: | ---: | ---: | --- | --- |
| template + cue-balanced | 60 | 2 | 0.366667 | true | `not_ready_for_activation_claims` |
| cue-balanced + lexical-hardened | 60 | 2 | 0.000000 | true | `bundle_incomplete` |

For `cue-balanced + lexical-hardened`, the bundle had 5 ready steps, 0
not-ready steps, and 1 skipped step. The skipped step was activation metadata
transfer, because no activation `.npz` file was available in this worktree.

## Interpretation

The useful finding is not that deterministic generated text is scientifically
enough. It is not. The useful finding is that the audit stack now distinguishes
three different failure modes:

- single-source coverage failure;
- metadata-only clone failure;
- source-diverse but lexically leaky benchmark failure.

The new deterministic scaffold gets past all non-activation gates without
falling into those three traps. That makes it a better substrate for the next
activation extraction or API-authored paraphrase run than the old
`template + cue_balanced` mix.

## Gate Results

`template + cue-balanced`:

- Manifest status: `not_ready_for_activation_claims`
- Ready steps: 4
- Not-ready steps: 1
- Skipped steps: 1
- Lexical leakage: 22 of 60 pairs cue-solved, cue-solved rate 0.366667
- Source diversity: ready, 2 sources, 20 shared fault groups, 0 duplicate pairs

`cue-balanced + lexical-hardened`:

- Manifest status: `bundle_incomplete`
- Ready steps: 5
- Not-ready steps: 0
- Skipped steps: 1
- Lexical leakage: 0 of 60 pairs cue-solved, cue-solved rate 0.000000
- Source diversity: ready, 2 sources, 20 shared fault groups, 0 duplicate pairs

## Local Run

The local experiment wrote uncommitted audit artifacts to:

`/tmp/social_cohesion_lexical_hardened_source_bundle`

The committed artifacts are the lexical-hardened style, tests, and this note.
Raw generated data remains out of git.

