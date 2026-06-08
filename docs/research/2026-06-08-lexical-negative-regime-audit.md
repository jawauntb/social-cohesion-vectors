# Lexical-Negative Regime Audit

Date: 2026-06-08

## Question

Can a lexical-negative prompt regime improve the generated fault-authorship
tournament by forcing pseudo/genuine pairs to share future-option vocabulary
while differing only in practical availability?

This was motivated by the prior authorship tournament: the selected shard had
10/10 scorer preference and stronger slack margins, but lexical leakage still
blocked activation readiness.

## Regime Transition

Old regime:

- Prompt contract: `behavioral_paths_pseudo_palette_v1`
- Hard-negative instruction: avoid obvious cue words when possible.
- Tournament candidates: `v1`, `v2`, `v3`
- Best selected shard: 10/10 scorer pass, 7/10 slack pass, 3/10 lexical pass,
  3/10 core-gate pass, 0/10 all-gate pass.

New regime:

- Prompt contract: `lexical_negative_v1`
- New prompt metadata: `lexical_negative_contract`
- New instruction: both hidden labels should use comparable
  review/evidence/privacy/appeal/exit/dissent/repair vocabulary. The label
  should be revealed by practical availability, not by which side has more
  prosocial words.

Preserved artifacts:

- Same first-20 fault-authorship prompt slice.
- Same `Qwen/Qwen2.5-7B-Instruct` Modal HF lane.
- Same candidate tournament and generated-audit bundle.
- Prior `v1`, `v2`, and `v3` candidate batches stayed in the tournament.

Rejected alternative:

- `v4`: lexical-negative generation as a standalone replacement candidate.

## Execution

Generated `v4` from the same first-20 slice:

```bash
.venv/bin/python scripts/run_fault_class_modal_generation.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --limit 20 \
  --seed 45 \
  --temperature 0.85 \
  --top-p 0.92 \
  --raw-outputs /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/raw_outputs.jsonl \
  --examples-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/examples.jsonl \
  --scored-runs-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/pairs.jsonl \
  --prompts-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/activation_prompts.jsonl \
  --json-report-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/dataset.json \
  --markdown-report-output /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/dataset.md \
  --audit-output-dir /tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/audit_bundle
```

Then reran the tournament with `v1`, `v2`, `v3`, and `v4`:

```bash
.venv/bin/python scripts/run_fault_authorship_tournament.py \
  --candidate-raw-outputs v1=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000/raw_outputs.jsonl \
  --candidate-raw-outputs v2=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v2/raw_outputs.jsonl \
  --candidate-raw-outputs v3=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v3/raw_outputs.jsonl \
  --candidate-raw-outputs v4=/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/chunk_000_v4_lexical_negative/raw_outputs.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --limit 20 \
  --selected-raw-outputs /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_raw_outputs.jsonl \
  --examples-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_examples.jsonl \
  --scored-runs-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_pairs.jsonl \
  --prompts-output /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_activation_prompts.jsonl \
  --tournament-json-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/tournament.json \
  --tournament-markdown-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/tournament.md \
  --dataset-json-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_dataset.json \
  --dataset-markdown-report /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/selected_dataset.md \
  --audit-output-dir /tmp/social_cohesion_authorship_tournament_20260608/all_variants_first20_v4/audit_bundle
```

Generated data and audit artifacts remain in `/tmp` and are not committed.

## Results

Standalone `v4` improved lexical balance but weakened the behavioral contrast.

| Candidate | Score pass | Slack pass | Lexical pass | Core gates | Mean score margin | Mean slack margin | Mean cue margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1` | 6/10 | 4/10 | 5/10 | 1/10 | +0.037 | +0.091 | +1.000 |
| `v2` | 4/10 | 4/10 | 7/10 | 1/10 | +0.008 | +0.028 | -0.100 |
| `v3` | 10/10 | 6/10 | 2/10 | 1/10 | +0.058 | +0.067 | +1.500 |
| `v4` | 3/10 | 3/10 | 6/10 | 0/10 | -0.006 | +0.021 | -0.400 |

The four-candidate tournament selected 10/10 pairs, but selected no `v4`
pairs. The selected shard metrics were unchanged from the prior `v1`/`v2`/`v3`
tournament:

- Score gate pass rate: `1.000`
- Slack gate pass rate: `0.700`
- Lexical gate pass rate: `0.300`
- Core gate triad passed: `3/10`
- All required gates passed: `0/10`
- Mean score margin: `+0.054`
- Mean slack-preservation margin: `+0.119`
- Mean cue margin: `+1.400`
- Tournament status: `selected_dataset_needs_review`

Standalone `v4` lexical audit:

- Cue-solved pairs: `4/10`
- Cue-inverted pairs: `6/10`
- Cue-solved rate: `0.400`
- Mean cue margin: `-0.400`
- Activation readiness: `not_ready_for_activation`

Standalone `v4` slack audit:

- Slack prefers genuine: `3/10`
- Mean slack-preservation margin: `+0.021`
- Min slack-preservation margin: `-0.070`
- Required future options covered: `6/8`
- Missing explicit coverage: `evidence_access`, `privacy_choice`

## Finding

The lexical-negative regime was a useful negative result. It reduced and often
inverted the simple lexical cue margin, but it did not produce better hard
negatives for the tournament because the behavioral and slack margins collapsed.

The failure mode in sampled texts is clear: pseudo-cohesion examples often used
the same wholesome future-option vocabulary as genuine examples, then taxed
those options with conditions such as compelling evidence, private-only review,
time delays, or social-impact pressure. That is semantically close to the
target fault, but the current scorer and slack rubric do not reliably separate
those practical burdens from genuine availability.

In the self-revising discovery framing, `v4` is not a new accepted regime. It is
a rejected transition that exposes residual content:

- lexical balancing alone can overcorrect into cue inversion;
- scorer/slack gates need sharper sensitivity to practical availability;
- pair selection should preserve rejected lexical-negative candidates as
  adversarial training material for the next verifier;
- activation extraction remains blocked until lexical and slack gates pass
  together.

## Interpretation

The next transition should not be "more lexical balancing." It should add a
paired counterfactual availability verifier:

- For each pseudo/genuine pair, extract the listed future paths.
- Ask whether each path is public, timely, non-retaliatory, evidence-accessible,
  and usable without loyalty/tone tests.
- Require the pseudo side to lose availability on at least one path and the
  genuine side to preserve availability on the same path.
- Keep lexical cue balance as a side constraint, not the main objective.

This converts the important distinction from a word-count leak into a typed
availability audit: the same vocabulary can be present on both sides, but the
operation "use this future path without penalty" must remain possible only in
the genuine example.

## Claim Boundary

This is a generated-text benchmark and verifier audit only. It makes no human
behavioral, neural, clinical, deployment, or real-world social-effect claim.

