# Availability Repair Modal Run

Date: 2026-06-08

## Question

Can targeted repair generation fix the residual practical-availability failures
from the first availability-targeted Modal/Qwen tournament without regressing
score and slack gates?

## Current Regime

- Artifact types: repair-target specs, repair-focused prompt records, raw Modal
  outputs, partial repair candidate sets, full-ledger tournament selections,
  availability audit bundles.
- Operations: parse repair targets, filter prompt records to failed base
  contrasts, generate repair candidates, combine partial repair candidates with
  the five broad candidate chunks.
- Gates: score margin, slack preservation, lexical leakage, path availability,
  length, formatting.
- Baseline before repair: the five-chunk availability-targeted tournament had
  score `10/10`, slack `10/10`, lexical `9/10`, availability `5/10`, core
  `4/10`, and all gates `2/10`.

## Regime Transition

Added `availability_repair_v1` plus repair target specs of the form:

```bash
--repair-target fair_allocation=refusal,appeal,repair
```

The repair prompt preserves all tested future paths, but names residual failed
paths as repair-focus paths. Pseudo-cohesion must tax those same paths; genuine
cohesion must make them usable now. Raw output rows also preserve
`repair_focus_options` and `availability_repair_contract` for tournament
provenance.

## Experiment

Generated three repair chunks with `Qwen/Qwen2.5-7B-Instruct`:

- `repair_000`: five residual base contrasts, seed `505`, temperature `0.9`;
- `repair_001`: same five residual base contrasts, seed `606`, temperature
  `0.75`;
- `repair_002`: only the remaining three failed base contrasts, seed `707`,
  temperature `0.65`, `--max-new-tokens 110`.

Repair artifacts live under:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/
```

The accepted comparison tournament is:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_000_001/
```

## Results

Standalone repair candidates:

| Candidate | Pairs | Score pass | Slack pass | Lexical pass | Availability pass | Core gates | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `repair_000` | 5 | 3 | 4 | 3 | 2 | 0 | 0 |
| `repair_001` | 5 | 3 | 2 | 4 | 1 | 0 | 0 |
| `repair_002` | 3 | 1 | 2 | 3 | 0 | 0 | 0 |

Best combined tournament, using `repair_000` and `repair_001`:

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

Selected repair winners:

- `accountability_after_harm`: `repair_000`, focus `repair`;
- `forgiveness_after_harm`: `repair_001`, focus `repair`.

Residual failed paths in the accepted repair tournament:

- `autonomy_after_conflict`: `dissent`;
- `belonging_norms`: `refusal`, `dissent`;
- `fair_allocation`: `refusal`, `appeal`, `repair`.

## Rejected Or Withheld Content

`repair_002` is rejected as a tournament addition. It improved lexical and
length shape, but failed availability on all three focused pairs. When included
with all other candidates, it made the tournament select a worse
`autonomy_after_conflict` pair because the current selection tuple can prefer
length fit over availability margin once the availability gate itself is already
false.

This exposed a verifier/selection issue rather than a content win.

## Residual Content

The repair mode is useful, but not activation-ready:

- practical availability improved from `5/10` to `7/10`;
- score and slack stayed at `10/10`;
- core gates did not improve beyond `4/10`;
- all gates did not improve beyond `2/10`;
- repair winners often fail lexical or length gates.

The remaining problem is no longer broad path coverage. It is joint satisfaction
of availability, lexical, and length gates for a few hard contrasts.

## Next Operation

Revise the repair loop before more sampling:

- either add `availability_repair_v2` with stricter one-paragraph length and
  lexical-balance constraints;
- or change tournament selection so availability margin is not dominated by
  length among candidates that still fail the availability gate;
- then regenerate only `autonomy_after_conflict`, `belonging_norms`, and
  `fair_allocation`.

## Claim Boundary

This is generated-text benchmark construction only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
