# Reviewer Methodology Note

This note captures the Spencer-style critique we are now baking into the repo.
The short version: do not let beautiful vector language outrun the geometry.

## What Changed

We added two reviewer-facing audits:

- `scripts/run_direction_geometry_audit.py`
- `scripts/run_residual_subspace_audit.py`

The first trains one contrastive direction per metadata group and reports signed
and absolute cosine structure. The second projects out the global contrastive
direction, then checks whether global or fault-specific residual directions
still separate pairs.

## Claims We Should Avoid

- Do not claim "orthogonal axes" from a mean signed off-diagonal cosine alone.
  Positive and negative cosines can cancel.
- Do not treat a cosine near `-1.0` as independence. That is the same unsigned
  axis with reversed pole.
- Do not use squared projection as if it preserves meaning. Squared projection
  reports axis energy, but it erases whether the signal points toward the
  genuine or pseudo pole.
- Do not claim that one residual ablation proves the first vector is uniquely
  causal or exhaustive. The contrastive method is designed to recover a dominant
  mean direction.

## Current Cue-Balanced Qwen Result

On the cue-balanced primary-fault activation set:

- primary-fault directions: 20;
- pairwise direction comparisons: 190;
- mean signed off-diagonal cosine: +0.624;
- mean absolute off-diagonal cosine: 0.624;
- strong anti-aligned pairs: 0;
- global direction pair-difference energy: 0.609;
- residual pair-difference energy after global projection: 0.391;
- residual global-direction accuracy: 0.000;
- residual fault-specific mean accuracy: 1.000.

The responsible interpretation is:

> The cue-balanced Qwen activations show one strong global genuine-vs-pseudo
> direction plus meaningful fault-specific residual subspaces. They do not show
> independent orthogonal axes, and one-vector ablation does not exhaust the
> signal.

## Reporting Standard Going Forward

Every activation-vector or SAE-feature result should report:

- signed cosine distribution;
- absolute cosine distribution;
- anti-aligned direction count;
- top aligned and anti-aligned pairs;
- global pair-difference energy captured;
- residual pair-difference energy;
- group-specific residual separation;
- signed projection margins whenever localization or pole meaning matters.
