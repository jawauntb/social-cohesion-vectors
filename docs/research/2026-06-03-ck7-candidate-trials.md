---
title: 2026-06-03 CK-7 Candidate Trials
status: draft
date: 2026-06-03
origin: CK-7 boundary-preserving prosociality prompt-battery design
---

# 2026-06-03 CK-7 Candidate Trials

## Purpose

CK-7 should test boundary-preserving prosociality under pressure before any
larger scheduled activation-cocktail claim is promoted. The target behavior is:
helpful, principled, truth-calibrated, autonomy-preserving assistance without
sycophancy, hallucination, coercion, dependency lock-in, privacy or exit-right
erosion, or coercive boundary collapse.

The checked-in battery is deterministic and intentionally small: 12 hard prompts,
two per pressure mode, with paired boundary-preserving and boundary-eroding
exemplar responses. It is larger than the six held-out CK-4 prompts but still
compact enough for targeted extraction, lexical leakage checks, and reviewer
inspection.

## Artifact

- Module: `src/social_cohesion_vectors/experiments/ck7_candidate_trials.py`
- Exporter: `scripts/export_ck7_candidate_trials.py`
- Test: `tests/test_ck7_candidate_trials.py`

The exporter writes prompt records, scored exemplar runs, pairwise examples,
activation prompts, and JSON/Markdown reports. The prompt records are for
generation-only model trials; the paired exemplar responses are for deterministic
scoring and activation-prompt construction.

## Coverage

| Pressure mode | Intended stress |
| --- | --- |
| `urgent_help` | Help quickly without lying, guilt, or forced compliance. |
| `authority_status` | Resist flattering authority when evidence or fairness is unresolved. |
| `emotional_dependency` | Support distress without exclusive reliance or exit suppression. |
| `group_unity` | Preserve dissent, review, privacy, and voluntary repair under unity pressure. |
| `social_debt` | Ask for reciprocity without coercive obligation or sensitive-data overreach. |
| `high_stakes_uncertainty` | Avoid false certainty and group-alignment pressure when facts are pending. |

| Failure target | Included contrast |
| --- | --- |
| `sycophancy` | Authority or group preference is not treated as truth. |
| `hallucination` | Pending facts are not upgraded into guarantees. |
| `coercion` | Urgency and reciprocity do not remove refusal rights. |
| `dependency_lock_in` | Care does not become exclusive reliance or exit control. |
| `privacy_exit_erosion` | Loyalty does not justify hidden collection or blocked review. |
| `boundary_collapse` | Unity and repair do not erase dissent, appeal, or voluntary forgiveness. |

## Claim Boundary

CK-7 is a deterministic compute-only prompt battery for model-response stress
testing. It is not evidence of real human prosocial effects, neural effects,
pharmacological effects, or therapeutic benefit.

## Next Use

Use CK-7 as a candidate-trial gate for future CK recipes: a promoted recipe
should improve boundary-preserving responses on this battery without increasing
unsupported certainty, flattery, coercive pressure, privacy erosion, dependency
language, or forced-unity responses.
