# Availability Audit First Run

Date: 2026-06-08

## Question

Can a path-level practical-availability verifier catch the failure mode exposed
by the `lexical_negative_v1` run: future-option words are present, but the
paths may still be costly, delayed, private-only, permissioned, or socially
taxed?

## Regime Transition

Old verifier stack:

- lexical leakage audit;
- lexical baseline diagnostic;
- component margin audit;
- slack-preservation audit;
- source-diversity audit;
- fault-held-out transfer;
- optional activation metadata transfer.

New verifier:

- `availability_audit`;
- per-pair, per-future-option path records;
- positive-minus-negative practical-availability margins;
- dimensions for public accountability, timeliness, non-retaliation, evidence
  accessibility, and freedom from loyalty/tone/permission tests;
- readiness gate that blocks activation when any tested path has nonpositive
  availability margin.

Implementation:

- `src/social_cohesion_vectors/experiments/availability_audit.py`
- `scripts/run_availability_audit.py`
- availability-aware candidate selection in
  `src/social_cohesion_vectors/experiments/fault_authorship_tournament.py`
- generated audit bundle integration in
  `src/social_cohesion_vectors/experiments/generated_audit_bundle.py`
- tests in `tests/test_availability_audit.py`,
  `tests/test_fault_authorship_tournament.py`,
  `tests/test_generated_audit_bundle.py`, and
  `tests/test_generated_fault_audit_pipeline.py`

## Execution

Ran the new audit on the existing `/tmp` artifacts from the first-20
Modal/Qwen tournament:

```bash
.venv/bin/python scripts/run_availability_audit.py \
  --pairs /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/pairs.jsonl \
  --json-output /tmp/social_cohesion_availability_audit_20260608/v4_lexical_negative/availability.json \
  --markdown-output /tmp/social_cohesion_availability_audit_20260608/v4_lexical_negative/availability.md

.venv/bin/python scripts/run_availability_audit.py \
  --pairs /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_pairs.jsonl \
  --json-output /tmp/social_cohesion_availability_audit_20260608/selected_tournament_v4/availability.json \
  --markdown-output /tmp/social_cohesion_availability_audit_20260608/selected_tournament_v4/availability.md

.venv/bin/python scripts/run_generated_benchmark_audit_bundle.py \
  --scored-runs /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_scored_runs.jsonl \
  --pairs /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_pairs.jsonl \
  --output-dir /tmp/social_cohesion_availability_audit_20260608/selected_tournament_v4_bundle

.venv/bin/python scripts/run_fault_authorship_tournament.py \
  --candidate-raw-outputs v1=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000/raw_outputs.jsonl \
  --candidate-raw-outputs v2=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v2/raw_outputs.jsonl \
  --candidate-raw-outputs v3=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v3/raw_outputs.jsonl \
  --candidate-raw-outputs v4=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/raw_outputs.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --limit 20 \
  --selected-raw-outputs /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_raw_outputs.jsonl \
  --examples-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_examples.jsonl \
  --scored-runs-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_pairs.jsonl \
  --prompts-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_activation_prompts.jsonl \
  --tournament-json-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/tournament.json \
  --tournament-markdown-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/tournament.md \
  --dataset-json-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_dataset.json \
  --dataset-markdown-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/selected_dataset.md \
  --audit-output-dir /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_availability/audit_bundle
```

Generated reports remain in `/tmp` and are not committed.

## Results

| Dataset | Pairs | Tested paths | Options covered | Paths prefer genuine | Path accuracy | Mean margin | Min margin | Ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v4_lexical_negative` | 10 | 42 | 6/8 | 20 | 0.476 | +0.237 | -0.330 | false |
| `selected_tournament_v4` | 10 | 42 | 6/8 | 24 | 0.571 | +0.342 | -0.040 | false |

Both runs are not ready for availability claims:

- missing required paths: `evidence_access`, `privacy_choice`;
- `v4` has nonpositive margins for every required future-option family;
- the selected tournament improves path accuracy and mean margin but still has
  nonpositive tested paths for refusal, exit, dissent, and repair;
- selected bundle status remains `not_ready_for_activation_claims`.

Selected tournament bundle after adding availability:

- ready steps: `2`;
- not-ready steps: `5`;
- skipped steps: `1`;
- ready: lexical baseline diagnostic, component margin audit;
- not ready: lexical leakage, slack preservation, availability, source
  diversity, source-held-out transfer;
- skipped: activation metadata transfer.

Availability-aware tournament rerun:

| Candidate | Score pass | Slack pass | Lexical pass | Availability pass | Core gates | Mean availability margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1` | 6/10 | 4/10 | 5/10 | 1/10 | 0/10 | +0.077 |
| `v2` | 4/10 | 4/10 | 7/10 | 0/10 | 0/10 | -0.245 |
| `v3` | 10/10 | 6/10 | 2/10 | 0/10 | 0/10 | -0.116 |
| `v4` | 3/10 | 3/10 | 6/10 | 0/10 | 0/10 | -0.049 |
| selected | 10/10 | 7/10 | 3/10 | 1/10 | 0/10 | +0.071 |

The availability-aware selector did not materially improve the selected shard:
the same candidate pool has too few availability-ready alternatives. The best
selected set still covers only 6/8 required future-option paths and has
nonpositive margins on refusal, exit, dissent, and repair paths.

## Finding

The availability verifier adds real information beyond lexical and slack gates.
The selected tournament looks better than standalone `v4` on practical
availability, but still fails because several metadata-declared paths are not
made practically available in text or are no better on the genuine side than
the pseudo side.

This clarifies the next transition: candidate selection should treat
availability as a core gate, not just a post-hoc report. A candidate pair should
not win because it has better scorer/slack margins if it leaves declared future
paths unmentioned, equally available on both sides, or more available in the
pseudo example.

After adding availability to tournament selection, the residual finding is
stronger: the current first-20 candidate pool itself is insufficient. The next
generator regime must produce candidates that explicitly cover
`evidence_access` and `privacy_choice`, and that keep every declared path
available on the genuine side while taxing at least one matched path on the
pseudo side.

## Next Operation

Add an availability-targeted generation regime:

- cover `evidence_access` and `privacy_choice` in the prompt slice;
- require every declared path to appear in both labels;
- require the genuine side to preserve each path without penalty;
- require the pseudo side to tax at least one matched path by delay,
  private-only routing, permission, tone/loyalty tests, retaliation risk, or
  evidence restriction;
- rerun the availability-aware tournament and require availability pass rate to
  improve before activation extraction.

## Claim Boundary

This is a generated-text benchmark and verifier audit only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.
