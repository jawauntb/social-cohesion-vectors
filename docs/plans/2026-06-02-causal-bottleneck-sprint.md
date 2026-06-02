---
title: 2026-06-02 Causal Bottleneck Sprint
status: completed
date: 2026-06-02
origin: user request to continue experiments and find a high-value research direction today
---

# 2026-06-02 Causal Bottleneck Sprint

## Problem Frame

The repo has strong compute-only representation evidence: controlled
social-cohesion directions separate deterministic cue-balanced and
affect-controlled prompts, and residualization does not erase the signal. The
causal story is weaker: naive activation addition moves hidden projections, but
generated text barely improves. Today should therefore chase the bottleneck:
where does hidden-state steering fail to become semantic control?

Recent steering work points in the same direction. Activation steering can work
for controllable domains, but layer choice, steering strength, instruction
position, sparse/vector refinement, and side effects matter. The valuable claim
today is not another probe accuracy. It is a controlled causal diagnostic that
either finds a better intervention layer or cleanly localizes why the current
directions do not steer behavior yet.

## Scope

In scope:

- affect-residualized directions for Qwen 0.5B layers `-1`, `-2`, and `-4`;
- hidden-state telemetry across those matched layers;
- a summary table that compares hook accuracy, projection movement, and text
  score movement;
- documentation updates that preserve the no-human/no-neural-claim caveat.

Out of scope today:

- Prolific, EEG, fMRI, fNIRS, or human validation claims;
- claiming that activation steering works unless behavior and projections move
  together;
- committing generated `data/` artifacts.

## Today's Research Hypotheses

H1. The affect-residualized boundary-prior direction is steerable in hidden
state across layers, with low hook delta error at `-1`, `-2`, and `-4`.

H2. Text-score movement is layer-dependent. Earlier or mid-late layers may
propagate the same signed displacement into generated text better than the
final layer.

H3. A publishable negative result is possible if all layers show accurate hidden
displacement plus weak behavioral movement: the bottleneck is downstream of
vector injection and should be studied as projection-to-output coupling.

H4. A publishable positive result becomes plausible only if one matched
layer/direction has monotonic positive-minus-negative projection movement and
positive-minus-negative behavior movement without autonomy or
pseudo-cohesion regressions.

## Implementation Units

### U1: Layer-Matched Affect-Residualized Directions

Files:

- Use existing `scripts/export_affect_control_benchmark.py`
- Use existing `scripts/run_activation_layer_sweep.py`
- Use existing `scripts/train_affect_residualized_direction.py`

Approach:

- Re-export the affect-control benchmark in the fresh worktree.
- Extract Qwen 0.5B activations for layers `-1`, `-2`, and `-4`.
- Save one affect-residualized steering direction per layer.

Verification:

- Each direction report has affect rank > 0.
- Each saved vector has max absolute affect-basis dot near zero.
- Each residualized direction keeps positive pair margins.

### U2: Telemetry Grid Summary

Files:

- Create `scripts/summarize_steering_telemetry_reports.py`
- Create `tests/test_summarize_steering_telemetry_reports.py`

Approach:

- Summarize multiple telemetry JSON files into a sortable table with model,
  layer, hook, timing, position, strengths, hidden delta error,
  positive-minus-negative post-hook projection, positive-minus-baseline
  projection, positive-minus-negative score, and positive-minus-baseline score.
- Sort by score delta, then projection delta, while keeping the raw report names
  visible.

Verification:

- Synthetic telemetry reports render expected rows and numeric formatting.

### U3: Layer-Matched Telemetry Sweep

Files:

- Use existing `scripts/run_modal_steering_telemetry.py`
- Use new `scripts/summarize_steering_telemetry_reports.py`

Approach:

- Run telemetry at layers `-1`, `-2`, and `-4` with matched
  affect-residualized directions.
- Use `post/generate/last`, strengths `-2 0 2`, and short greedy generations
  to keep the sweep fast and comparable.
- Summarize all reports into one Markdown/JSON artifact under ignored `data/`.

Verification:

- Every report has near-zero hidden delta error.
- The summary identifies whether any layer improves the text-score delta.

### U4: Research Notes And Next Claim

Files:

- Update `README.md`
- Update `docs/neurips_trajectory_plan.md`
- Update `docs/papers/experiment_log.md`
- Optionally update `docs/papers/social_cohesion_paper_draft.md`

Approach:

- If a layer produces a meaningful score/projection improvement, document it as
  the next steering candidate.
- If all layers remain weak, document the stronger negative claim: hidden
  injection works across layers, but semantic behavior still bottlenecks
  downstream.

Verification:

- Docs state synthetic/open-model caveats.
- Docs do not imply human or neural validation.

## Success Bar

The most valuable result today is one of:

- a layer where affect-residualized steering has both positive projection and
  positive behavior movement, making it the next causal candidate; or
- a clean three-layer bottleneck result showing accurate injection and weak
  behavior, which is a defensible mechanistic negative result and a stronger
  paper claim than another probe.

Outcome: the sprint found the second result. Layer-matched
affect-residualized directions at `-1`, `-2`, and `-4` all inject accurately
and move post-hook hidden projections by about +3.9 to +4.0 from negative to
positive steering. Short 24-token score movement is tiny but positive, with
`-2`/`-4` slightly ahead. In 64-token generation, however, `-2` and `-4`
become behaviorally worse than the final layer. The strongest claim is now a
projection-to-output coupling bottleneck, not successful prosocial steering.

## Quality Gates

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `git diff --check`
