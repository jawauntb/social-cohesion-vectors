# Availability-Targeted Modal Run

Date: 2026-06-08

## Question

Does the availability-targeted prompt regime improve the first-20 Modal/Qwen
candidate pool enough to unblock generated activation extraction?

## Current Regime

- Artifact types: prompt records, raw Modal outputs, pairwise generated
  examples, availability audits, tournament selections, audit bundles.
- Operations: Modal HF generation, availability audit, generated benchmark
  audit bundle, authorship tournament.
- Gates: score margin, slack preservation, lexical leakage, practical
  availability, length, formatting.
- Baseline before this run: selected tournament from the previous first-20 pool
  had `1/10` availability-gate pass rate and `0/10` core-gate pass rate.

## Experiment

Generated five first-20 candidate chunks with
`Qwen/Qwen2.5-7B-Instruct`:

- `chunk_000`: `availability_targeted_v1`, default sampling;
- `chunk_001`: `availability_targeted_v1`, seed `101`, temperature `1.0`;
- `chunk_002_v2`: `availability_targeted_v2`, seed `202`;
- `chunk_003_v2`: `availability_targeted_v2`, seed `303`;
- `chunk_004_v2`: `availability_targeted_v2`, seed `404`.

All chunks used:

```bash
--variants neighborhood_forum \
--limit 20 \
--availability-priority
```

Generated artifacts are under:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_chunks_000_001_002v2_003v2_004v2/
```

## Results

Standalone candidate availability:

| Candidate | Contract | Score pass | Slack pass | Availability path accuracy | Pairs all paths available | Mean availability margin | Min margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chunk_000` | `availability_targeted_v1` | 2/10 | 2/10 | 0.217 | 1/10 | -0.067 | -0.640 |
| `chunk_001` | `availability_targeted_v1` | 6/10 | 6/10 | 0.457 | 2/10 | +0.006 | -0.790 |
| `chunk_002_v2` | `availability_targeted_v2` | 4/10 | 6/10 | 0.870 | 4/10 | +0.326 | -0.620 |
| `chunk_003_v2` | `availability_targeted_v2` | 6/10 | 6/10 | 0.826 | 3/10 | +0.396 | -0.230 |
| `chunk_004_v2` | `availability_targeted_v2` | 5/10 | 9/10 | 0.674 | 3/10 | +0.204 | -0.290 |

Final five-candidate tournament:

| Metric | Result |
| --- | ---: |
| selected pairs | 10/10 |
| score gate | 10/10 |
| slack gate | 10/10 |
| lexical gate | 9/10 |
| availability gate | 5/10 |
| core gates | 4/10 |
| all gates | 2/10 |
| selected availability path accuracy | 0.826 |
| selected pairs all paths available | 5/10 |
| mean selected availability margin | +0.339 |
| min selected availability margin | -0.230 |

Selected winners by contract:

- `availability_targeted_v2`: 18 selected raw rows;
- `availability_targeted_v1`: 2 selected raw rows.

The selected tournament preserved raw-row contract provenance after a code fix:
mixed v1/v2 candidate rows no longer inherit the tournament prompt contract
incorrectly.

## Accepted Content

The `availability_targeted_v2` regime is a real improvement over v1:

- restores selected score and slack gates to `10/10`;
- improves selected availability from the previous `1/10` baseline to `5/10`;
- improves core gates from `0/10` to `4/10`;
- yields two all-gates examples;
- covers all eight future-option paths, including `evidence_access` and
  `privacy_choice`.

## Rejected Or Withheld Content

The run is not activation-ready.

Failed future-option families in the selected set:

- `refusal`;
- `appeal`;
- `dissent`;
- `repair`.

Residual failed tested paths:

- `accountability_after_harm`: `repair`;
- `autonomy_after_conflict`: `dissent`;
- `belonging_norms`: `refusal`, `dissent`;
- `fair_allocation`: `refusal`, `appeal`, `repair`;
- `forgiveness_after_harm`: `repair`.

## Residual Content

The bottleneck has narrowed. Broad availability prompting works for
`evidence_access`, `privacy_choice`, `exit`, and `proportional_review`, but
Qwen still struggles with specific fault/path combinations where the pseudo
side should tax refusal, appeal, dissent, or repair without collapsing the score
and slack margins.

The next move should not be another broad first-20 generation sweep. It should
be targeted repair generation for the five failing base contrasts and their
failed path families.

## Next Operation

Add a repair-focused generation mode or prompt slice that takes explicit
residual failures:

- base contrast;
- failed future-option paths;
- required pseudo tax dimensions;
- requirement that genuine preserve the same paths.

Then generate candidate rows only for the failed base contrasts and merge them
into the tournament candidate pool.

## Claim Boundary

This is generated-text benchmark construction only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
