# Availability Repair V2 Modal Run

Date: 2026-06-08

## Question

Can a stricter repair prompt contract fix the remaining practical-availability
failures while preserving score, slack, lexical, and length gates?

## Regime Move

Added `availability_repair_v2` as a sibling to `availability_repair_v1`.
The v2 prompt contract adds:

- exact one-paragraph, 55-75-word discipline;
- preservation of every tested future-option path;
- contrast focused only on named repair-focus paths;
- pseudo-side concrete taxes on every focus path;
- explicit blocking of healthy pseudo shortcuts such as "without penalty" unless
  a cost or condition is added immediately;
- same-path requirements so genuine makes the named path usable and pseudo taxes
  that same path.

Raw output rows continue to preserve `availability_repair_contract` and
`repair_focus_options`.

## Experiment

Generated only the three remaining hard base contrasts with
`Qwen/Qwen2.5-7B-Instruct`:

```bash
--variants neighborhood_forum \
--availability-priority \
--prompt-contract-version availability_repair_v2 \
--repair-target autonomy_after_conflict=dissent \
--repair-target belonging_norms=refusal,dissent \
--repair-target fair_allocation=refusal,appeal,repair \
--temperature 0.7 \
--top-p 0.85 \
--seed 808 \
--max-new-tokens 105
```

Generation artifacts:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/
```

Comparison tournament, using the five broad availability-targeted chunks,
accepted repair chunks `repair_000` and `repair_001`, and new
`repair_v2_000`:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_000_001_repair_v2_000_limit20/
```

The tournament used the same prioritized first-20-record scope as the accepted
repair baseline.

## Results

Standalone `repair_v2_000` candidate summary:

| Metric | Result |
| --- | ---: |
| pairs | 3 |
| score gate | 1/3 |
| slack gate | 2/3 |
| lexical gate | 2/3 |
| availability gate | 0/3 |
| core gates | 0/3 |
| all gates | 0/3 |
| availability path accuracy | 5/14 |
| mean path availability margin | +0.104 |
| min path availability margin | -0.310 |

Combined tournament after adding `repair_v2_000`:

| Metric | Result |
| --- | ---: |
| selected pairs | 10/10 |
| score gate | 10/10 |
| slack gate | 10/10 |
| lexical gate | 7/10 |
| availability gate | 7/10 |
| core gates | 4/10 |
| all gates | 2/10 |
| selected availability path accuracy | 40/46 |
| selected pairs all paths available | 7/10 |
| mean selected availability margin | +0.418 |
| min selected availability margin | -0.150 |

This is unchanged from the accepted `repair_000` plus `repair_001` tournament.
`repair_v2_000` won no selected pairs.

## Failure Analysis

The stricter prompt did not reliably constrain Qwen output:

- `autonomy_after_conflict`: availability margin `+0.000`, score gate failed,
  and the pseudo row was only 40 words;
- `belonging_norms`: availability margin `+0.000`, score and slack failed, and
  the pseudo row was only 47 words;
- `fair_allocation`: availability margin `-0.310`, lexical and length failed.

The raw pseudo examples often became healthy shortcut statements instead of
taxed pseudo-cohesion. For example, `fair_allocation` and `belonging_norms`
used "without fear" or "without punishment or shaming" language without adding
the immediate cost required by the contract.

Selection-policy inspection also found that ordering alone is not enough to
beat the baseline:

- for `autonomy_after_conflict`, `repair_000` has better availability
  (`+0.310`) but fails the slack gate, so selecting it would break the required
  `10/10` slack target;
- for `belonging_norms`, zero-margin alternatives exist, but they still do not
  pass availability;
- for `fair_allocation`, no current candidate has positive availability.

## Residual Failed Paths

The accepted tournament still fails the same tested paths:

- `autonomy_after_conflict`: `dissent`;
- `belonging_norms`: `refusal`, `dissent`;
- `fair_allocation`: `refusal`, `appeal`, `repair`.

## Next Operation

Plain prompt-side tightening is no longer enough evidence of a promising
regime move. The next loop should use verifier-aware repair:

- generate multiple candidates per hard contrast and reject rows that fail
  local length, score/slack, or availability audits before tournament input;
- or add a second-pass rewrite step that takes the failed path-level audit and
  explicitly patches the pseudo/genuine pair;
- then rerun the same first-20 tournament with only candidates that pass the
  local repair gates.

Selection-order changes can still be tested, but they should not accept slack
regression or zero availability margins as progress.

## Verifier-Aware Filter Step

Added a local repair-candidate filter that evaluates candidate pairs with the
same scorer, slack, practical availability, length, and formatting gates used by
the tournament, then writes only verifier-passing raw prompt rows forward. The
default required gates are:

- score prefers genuine;
- slack prefers genuine;
- availability prefers genuine;
- both texts are within the target word-count range;
- formatting is clean.

Filter test on `repair_v2_000`:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000_filter_default/
```

Result:

| Metric | Result |
| --- | ---: |
| expected pairs | 3 |
| evaluated candidate pairs | 3 |
| accepted pairs | 0 |
| accepted raw outputs | 0 |
| rejected candidate pairs | 3 |

Rejected pairs:

- `autonomy_after_conflict`: failed score, availability, and length gates;
- `belonging_norms`: failed score, slack, availability, and length gates;
- `fair_allocation`: failed availability and length gates.

This filter would have prevented `repair_v2_000` from entering the comparison
tournament at all. The next sampling loop should produce multiple repair-v2
candidates and keep only rows that pass this local filter before rerunning the
first-20 tournament.

## Claim Boundary

This is generated-text benchmark construction only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
