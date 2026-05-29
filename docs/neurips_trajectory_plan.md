# NeurIPS Trajectory Plan

The fastest path from the current repo to a NeurIPS-worthy claim is to stop
treating activation directions as only diagnostic probes. The claim has to
become causal, out-of-distribution, and safety constrained:

> A signed activation direction learned from controlled social-cohesion
> contrasts causally changes model behavior on unseen social decision prompts,
> generalizes across prompt families and model scales, and improves
> agency-preserving cohesion without increasing compliance, coercion, or
> pseudo-cohesion failures.

## Highest-Leverage Experiments

1. **Causal activation steering.** Train directions on cue-balanced pairs, inject
   them during generation, and score unseen responses under positive, zero, and
   negative steering strengths. A probe-only result is not enough.
2. **Generated hard-negative benchmark.** Replace deterministic wrappers with
   generated/API-authored paraphrases and pseudo-cohesion negatives. Require
   lexical leakage, scorer, geometry, residual, and signed-vs-squared audits.
3. **Anti-compliance controls.** Show steering does not merely make the model
   warmer, more agreeable, more sycophantic, or more boundary-collapsing.
4. **Cross-model replication.** Repeat on Qwen 0.5B/1.5B/3B and at least one
   non-Qwen open model if compute allows.
5. **Mechanistic decomposition.** Preserve the reviewer-corrected claim: one
   shared signed manifold plus residual mechanism subspaces, not independent
   orthogonal axes.
6. **Human validation only after compute controls.** A small Prolific pairwise
   validation becomes meaningful only after generated-text controls stop being
   trivially cue-solvable.

## First Causal Sprint Result

This branch adds a Modal generation hook and local report layer for causal
activation steering. It uses held-out social decision prompts, inserts a signed
direction at a selected transformer layer, generates responses at negative,
zero, and positive strengths, then scores the outputs with the local cohesion
rubric.

The first Qwen 0.5B boundary-prior and fault-class steering smokes are weak or
mixed rather than clean wins. Positive steering sometimes improves individual
prompts, but the mean rubric shift is near zero or negative under the tested
settings. This is useful: it means the current probe directions are not yet
reliable steering directions under naive activation addition.

## Immediate Next Move

The next high-value work is a steering-method sweep, not another probe-only
benchmark:

- compare residual-stream hook sites against MLP/block-output hook sites;
- sweep strengths with generation quality checks;
- test prompt-only steering versus generated-token-only steering;
- score with a stronger pairwise evaluator, not only the lexical local rubric;
- evaluate whether steering shifts the projection of the generated outputs back
  onto the learned direction;
- add explicit pseudo-cohesion and compliance regressions to every steering
  report.

If a setting yields monotonic positive-vs-negative behavior shifts while passing
anti-compliance controls, that becomes the first genuinely publication-grade
mechanistic claim in this repo.
