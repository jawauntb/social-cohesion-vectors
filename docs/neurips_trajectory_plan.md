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
4. **Affect residualization.** Show the cohesion signal survives coarse
   anger/sadness/fear/disgust/happy/neutral controls, then repeat the same
   regression in activation space before treating the direction as social
   reasoning rather than affect/style.
5. **Cross-model replication.** Repeat on Qwen 0.5B/1.5B/3B and at least one
   non-Qwen open model if compute allows.
6. **Mechanistic decomposition.** Preserve the reviewer-corrected claim: one
   shared signed manifold plus residual mechanism subspaces, not independent
   orthogonal axes.
7. **Human validation only after compute controls.** A small Prolific pairwise
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

The first method sweep makes that more precise. Small post-hook edits on
generated tokens are the least bad behavioral setting so far: 0.750
positive-vs-negative cohesion success with a +0.004 mean score delta. Stronger
post/generate/last steering at +/-4 is more mechanistically interesting but not
behaviorally good: re-embedded generated responses separate positive from
negative steering by +3.561 along the learned direction, while the local text
score moves -0.021 and positive steering remains below baseline projection.
That is a useful dissociation: the intervention can move generated text in the
learned representation without yet producing agency-preserving cohesion in the
output. A dense -6..+6 run confirms the current edit is not a symmetric
generative control knob: negative steering pushes generations down the learned
projection, positive steering does not lift them above baseline, and behavior
stays flat.

The hidden-state telemetry pass now localizes the failure more cleanly. At
layers -1, -2, and -4, the hook applies the requested displacement almost
exactly during generation: mean absolute delta error is 0.0073, 0.0018, and
0.0025 respectively. Post-hook projection shifts by roughly +11 along the
learned direction from negative to positive steering. The short generated text
score only moves +0.015 to +0.024, with earlier layers slightly better than the
final layer. This means the next publishable claim is not "we can steer
cohesion"; it is a precise control-bottleneck claim: the learned direction is a
reliable hidden-state displacement, but the current intervention does not yet
compose into stable semantic behavior.

## Immediate Next Move

The affect-control activation pass is now complete for Qwen 0.5B and 1.5B at
layers -1, -2, and -4. Raw cohesion directions and affect-residualized cohesion
directions both reach 1.000 leave-one-pair-out accuracy on the 72-pair
affect-control benchmark. This strengthens the representation claim: the
current boundary-prior signal is not erased by coarse
anger/sadness/fear/disgust/happy/neutral activation subspaces. It remains
synthetic text evidence, not human or EEG validation.

Layer-matched affect-residualized Qwen 0.5B directions are now saved for layers
-1, -2, and -4. All three are orthogonal to the five-dimensional affect-label
basis and keep positive in-sample margins on the 72 affect-control pairs:
+8.424, +1.952, and +1.652. Hidden telemetry shows accurate injection and
positive-minus-negative post-hook projection movement at every layer
(+3.883, +3.939, and +4.031). Short 24-token text-score movement is tiny but
positive (+0.009, +0.019, +0.019). The longer 64-token generation run is the
important result: layer -1 is flat/slightly positive (0.500 win rate, +0.002
score delta), while layers -2 and -4 become behaviorally worse (0.167/-0.026
and 0.083/-0.018). This strengthens the control-bottleneck claim: hidden
projection movement is not sufficient for stable semantic steering.

The next high-value work is therefore a monotonic steering protocol, not another
probe-only benchmark:

- compare residual-stream hook sites against MLP/block-output hook sites;
- sweep strengths with generation quality checks;
- test prompt-only steering versus generated-token-only steering;
- score with a stronger pairwise evaluator, not only the lexical local rubric;
- require steering to shift generated-output projections monotonically before
  treating the intervention as mechanistically active;
- require generated-output projection movement and pairwise behavioral
  improvement to agree before calling the edit prosocial;
- regress out affect directions learned from the new affect-control prompt set
  so a steering win cannot be explained as lower threat, warmer tone, or generic
  positive affect;
- add hidden-state telemetry to every steering sweep so failed runs can be
  localized to hook application, logit propagation, decoding, or scoring;
- add explicit pseudo-cohesion and compliance regressions to every steering
  report.

If a setting yields monotonic positive-vs-negative behavior shifts while passing
anti-compliance controls, that becomes the first genuinely publication-grade
mechanistic claim in this repo.
