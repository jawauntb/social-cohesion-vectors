# Agent Handoff: Generated Fault-Class Availability Repair

Date: 2026-06-08

Use this note to continue the research loop in a fresh agent session. Generated
artifacts and model outputs are intentionally kept out of git; durable findings
are summarized in `docs/research/`.

## Repo State

Start from `origin/main`. The latest relevant merged commits are:

- `b67eae8` / PR #88: `Add strict availability-targeted generation loop`
- `522c486` / PR #89: `Add availability repair generation loop`

Recommended setup for the next session:

```bash
git fetch origin main
git checkout -B codex/availability-repair-v2 origin/main
```

Do not copy secrets from other repos. Modal auth worked from the local Modal
configuration in this session; `.env` files and generated artifacts must stay
out of git.

## What We Built

### Practical Availability Audit

The generated benchmark now has a path-level practical-availability verifier.
It checks whether future-option paths are actually usable in genuine examples
and taxed in pseudo examples, instead of merely being mentioned.

Relevant files:

- `src/social_cohesion_vectors/experiments/availability_audit.py`
- `src/social_cohesion_vectors/experiments/generated_audit_bundle.py`
- `src/social_cohesion_vectors/experiments/fault_authorship_tournament.py`

### Availability-Targeted Prompt Contracts

Prompt records now support:

- `lexical_negative_v1`
- `availability_targeted_v1`
- `availability_targeted_v2`
- `availability_repair_v1`

`availability_targeted_v2` was added because v1 sometimes generated pseudo
examples that were actually healthy. V2 requires concrete pseudo-side taxes and
forbids shortcut phrases like "without penalty" unless a cost/condition is
added immediately.

Relevant files:

- `src/social_cohesion_vectors/experiments/fault_generation.py`
- `scripts/run_fault_class_modal_generation.py`
- `scripts/run_fault_authorship_tournament.py`
- `tests/test_fault_generation.py`
- `tests/test_fault_class_modal_generation.py`
- `tests/test_fault_authorship_tournament.py`

### Repair-Targeted Generation

`availability_repair_v1` adds repair-target filtering:

```bash
--repair-target BASE=option,option
```

Example:

```bash
--repair-target fair_allocation=refusal,appeal,repair
```

This emits only the prompt pairs for named failed base contrasts while
preserving all tested paths in the prompt. Raw output rows preserve:

- `availability_repair_contract`
- `repair_focus_options`

This lets tournament-selected repair winners remain auditable.

## Live Runs Completed

### Five-Chunk Availability-Targeted Sweep

Model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Artifacts:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_chunks_000_001_002v2_003v2_004v2/
```

Best tournament before repair:

| Gate | Result |
| --- | ---: |
| Score | 10/10 |
| Slack | 10/10 |
| Lexical | 9/10 |
| Availability | 5/10 |
| Core | 4/10 |
| All gates | 2/10 |

Durable note:

```text
docs/research/2026-06-08-availability-targeted-modal-run.md
```

### Repair Sweep

Repair chunks:

- `repair_000`: five residual base contrasts, seed `505`, temperature `0.9`
- `repair_001`: same five residual base contrasts, seed `606`, temperature
  `0.75`
- `repair_002`: remaining three hard contrasts, seed `707`, temperature
  `0.65`, `--max-new-tokens 110`

Artifacts:

```text
/tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/
```

Accepted comparison tournament:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_000_001/
```

Best repaired tournament:

| Gate | Result |
| --- | ---: |
| Score | 10/10 |
| Slack | 10/10 |
| Lexical | 7/10 |
| Availability | 7/10 |
| Core | 4/10 |
| All gates | 2/10 |
| Availability path accuracy | 40/46 |

Selected repair winners:

- `accountability_after_harm`: `repair_000`, focus `repair`
- `forgiveness_after_harm`: `repair_001`, focus `repair`

Rejected:

- `repair_002`: improved lexical/length shape, but failed availability on all
  three focused pairs. When included, tournament selection picked a worse
  `autonomy_after_conflict` row because the current selection tuple can prefer
  lexical/length over availability margin after the availability gate is already
  false.

Durable note:

```text
docs/research/2026-06-08-availability-repair-run.md
```

## Current Bottleneck

Activation extraction remains blocked. The current bottleneck is joint
satisfaction of:

- practical availability;
- lexical balance;
- target length;
- score/slack separation.

Remaining failed paths in the accepted repair tournament:

- `autonomy_after_conflict`: `dissent`
- `belonging_norms`: `refusal`, `dissent`
- `fair_allocation`: `refusal`, `appeal`, `repair`

Current baseline to beat:

- availability above `7/10`;
- core above `4/10`;
- all gates above `2/10`;
- lexical above `7/10`;
- score and slack must remain `10/10`;
- all eight future-option paths must remain covered.

## Important Selection-Policy Detail

The tournament selection tuple currently ranks lexical and length before
availability margin among candidates that all fail the availability gate. That
is why `repair_002` could be selected for `autonomy_after_conflict` despite a
worse availability margin.

The next agent should decide whether the next regime move is:

1. Prompt-side: add `availability_repair_v2` with stricter one-paragraph,
   55-75-word, lexical-balance, and explicit same-path constraints.
2. Selection-side: revise tournament ordering so availability margin is not
   dominated by lexical/length among availability-failing candidates.

Recommended first move: prompt-side `availability_repair_v2`, because the
repair_000/001 examples proved content repair can improve availability, but
their most obvious flaw was being too long or lexically leaky.

## Suggested Next Experiment

Add `availability_repair_v2` with constraints like:

- exactly one paragraph;
- 55-75 words;
- preserve every tested path;
- focus only the named repair paths;
- pseudo must tax the focus paths without healthy shortcut phrases;
- genuine must make the same focus paths usable now;
- avoid adding extra healthy escape hatches not in the target paths;
- maintain comparable future-option vocabulary across labels.

Then generate only the remaining hard contrasts:

```bash
.venv/bin/python scripts/run_fault_class_modal_generation.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --variants neighborhood_forum \
  --availability-priority \
  --prompt-contract-version availability_repair_v2 \
  --repair-target autonomy_after_conflict=dissent \
  --repair-target belonging_norms=refusal,dissent \
  --repair-target fair_allocation=refusal,appeal,repair \
  --temperature 0.7 \
  --top-p 0.85 \
  --seed 808 \
  --max-new-tokens 105 \
  --raw-outputs /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/raw_outputs.jsonl \
  --examples-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/examples.jsonl \
  --scored-runs-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/pairs.jsonl \
  --prompts-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/activation_prompts.jsonl \
  --json-report-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/dataset.json \
  --markdown-report-output /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/dataset.md \
  --audit-output-dir /tmp/social_cohesion_modal_hf_qwen7_availability_repair_20260608/repair_v2_000/audit_bundle
```

Then rerun the tournament against:

- the five broad chunks from the availability-targeted sweep;
- accepted repair chunks `repair_000` and `repair_001`;
- the new repair v2 chunk.

## Verification Commands

Before committing:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pyright --pythonpath .venv/bin/python
.venv/bin/python -m pytest
```

The last completed verification before this handoff:

- `ruff`: passed;
- `pyright`: passed;
- `pytest`: `265 passed`.

## Claim Boundary

Everything above is generated-text benchmark construction. It does not support
human behavioral, neural, clinical, deployment, or real-world social-effect
claims. Human validation, Prolific, fMRI, EEG, fNIRS, or hyperscanning tracks
remain parked until generated-text and activation gates pass together.
