# Fault Authorship Candidate Tournament

Date: 2026-06-08

## Question

Can a ruliology-style candidate tournament turn several imperfect open-model
authorship batches into a stronger generated pseudo-cohesion benchmark shard?

The operational target was not "find the best generator run." It was: select
one pseudo/genuine pair per contrast from multiple candidate batches using a
payoff vector that includes behavioral scorer margin, slack-preservation
margin, lexical leakage resistance, length fit, and formatting hygiene.

## Setup

Implementation:

- `src/social_cohesion_vectors/experiments/fault_authorship_tournament.py`
- `scripts/run_fault_authorship_tournament.py`

Executed tournament:

- Candidate batches: `v1`, `v2`, `v3`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Provider lane: `modal_hf`
- Prompt slice: first 20 fault-authorship prompt records, all variants
- Expected pairs: 10
- Artifact directory:
  `/tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20`

The selected dataset and audit bundle are intentionally left in `/tmp` rather
than committed because generated data/model artifacts should stay out of git.

## Results

The tournament selected a complete 10-pair shard. No single candidate batch
had scorer pass on all 10 pairs, but the selected mix did:

| Set | Score pass | Slack pass | Lexical pass | Core gates |
| --- | ---: | ---: | ---: | ---: |
| `v1` | 6/10 | 4/10 | 5/10 | 1/10 |
| `v2` | 4/10 | 4/10 | 7/10 | 1/10 |
| `v3` | 10/10 | 6/10 | 2/10 | 1/10 |
| selected | 10/10 | 7/10 | 3/10 | 3/10 |

Selected winners were mixed rather than dominated by one batch:

- `v3`: 5 pairs
- `v1`: 4 pairs
- `v2`: 1 pair

The selected set passed the component-margin audit:

- Scorer pairwise accuracy: `1.000`
- Mean score margin: `+0.054`
- Mean slack-preservation margin: `+0.119`
- Component audit readiness: `activation_ready`

But it did not pass the full generated benchmark audit bundle:

- Audit bundle status: `not_ready_for_activation_claims`
- Lexical cue-solved rate: `0.700`
- Slack preservation audit: not ready
- Source diversity audit: not ready
- Activation metadata transfer: skipped because no activation `.npz` was
  provided

## Finding

The tournament is useful as a selection layer, but it exposes a real tradeoff
in the current candidate pool:

**The batch that best satisfies behavioral scoring (`v3`) is also the leakiest
lexically, while the most lexical-resistant batch (`v2`) has weaker scorer and
slack margins.**

That is the first concrete tournament finding. Candidate selection can recover
10/10 scorer preference from imperfect runs, but the current generator prompts
do not yet produce enough examples that jointly satisfy behavioral separation,
slack preservation, and lexical resistance.

## Interpretation

This supports the CK-ruliology framing: compounds or generated artifacts should
be evaluated as programs in an adversarial environment, not as isolated samples.
Here the "programs" were generation regimes. The payoff vector changed the
conclusion:

- If we optimize only for behavioral scorer margin, `v3` looks best.
- If we optimize for lexical resistance, `v2` looks best.
- If we require score, slack, and lexical gates together, only 3/10 selected
  pairs pass the core triad.

That means the next generator should not simply ask for "more genuine" or "more
pseudo." It needs an explicit lexical-negative objective: pseudo and genuine
texts must share autonomy/truth/slack vocabulary while differing in whether the
future options are actually preserved.

## Limitations

This is a generated-text benchmark shard only. It makes no human behavioral,
neural, or real-world cohesion claim.

The executed shard covers only four primary fault classes:

- `punitive_accountability`
- `consent_bypass`
- `assimilation_pressure`
- `emotional_blackmail`

The slack audit is not ready partly because this shard cannot cover all future
option paths. Evidence access and privacy choice were absent in the selected
slice.

## Next Operations

1. Add a lexical-negative prompt regime whose objective is not "avoid cue
   words," but "balance cue words across labels while preserving the true fault
   distinction."
2. Run the tournament over a wider shard that covers all primary fault classes
   and all future-option paths.
3. Add a policy frontier report comparing scorer-first, lexical-first, and
   core-triad-first selection.
4. Only send a selected generated dataset into activation extraction after the
   lexical and slack audits pass together.
