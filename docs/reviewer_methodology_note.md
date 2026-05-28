# Reviewer Methodology Note

This note captures the Spencer-style critique we are now baking into the repo.
The short version: do not let beautiful vector language outrun the geometry.

## What Changed

We added three reviewer-facing audits:

- `scripts/run_direction_geometry_audit.py`
- `scripts/run_residual_subspace_audit.py`
- `scripts/run_activation_subspace_probe.py`

The first trains one contrastive direction per metadata group and reports signed
and absolute cosine structure. The second projects out the global contrastive
direction, then checks whether global or fault-specific residual directions
still separate pairs. The third fits k-component signed SVD bases over
positive-minus-negative activation differences and reports signed-vote accuracy
separately from squared subspace-energy accuracy.

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

## Current Autonomy Stress Subspace Result

On the structural-autonomy stress suite, layer choice already matters:

- Qwen 0.5B layer -1: 0.875 leave-one-pair-out vector accuracy;
- Qwen 0.5B layer -2: 1.000 leave-one-pair-out vector accuracy;
- Qwen 0.5B layer -4: 1.000 leave-one-pair-out vector accuracy;
- Qwen 1.5B layer -1: 0.938 leave-one-pair-out vector accuracy;
- Qwen 1.5B layer -2: 1.000 leave-one-pair-out vector accuracy.

The strongest signed-vs-squared warning comes from Qwen 1.5B layer -2:

- best pair-LOO signed-vote subspace accuracy: 1.000;
- best pair-LOO squared-energy accuracy: 0.750;
- first pair-difference component energy: 0.170.

The responsible interpretation is:

> Signed activation structure separates the autonomy-preserving and
> autonomy-risk poles, but squared projection energy is not a substitute for
> pole direction. Localization reports must preserve sign.

## Reporting Standard Going Forward

Every activation-vector or SAE-feature result should report:

- signed cosine distribution;
- absolute cosine distribution;
- anti-aligned direction count;
- top aligned and anti-aligned pairs;
- global pair-difference energy captured;
- residual pair-difference energy;
- group-specific residual separation;
- signed-vs-squared subspace probe results across k components;
- signed projection margins whenever localization or pole meaning matters.

## Boundary-Prior Addition

The Sandved-Smith, Fields, Doctor, Laukkonen, and Hohwy preprint gives a useful
conceptual warning: do not confuse a pragmatically useful partition with an
ontologically fixed boundary. For this repo, the practical reviewer question is
whether an apparent cohesion signal is actually detecting one of three boundary
modes:

- rigid boundary reification: self/other, us/them, loyal/disloyal;
- flexible contextual relation: boundaries remain available but revisable;
- coercive boundary collapse: unity language removes refusal, dissent, exit, or
  verification.

The first and third are pseudo-cohesion risks. The second is the target.
