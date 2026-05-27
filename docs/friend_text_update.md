# Friend Text Update

Hey, quick context on the thing I started: I spun up a repo for a compute-first
social-cohesion vector project. The idea is to test whether open language models
have measurable internal directions or sparse features for agency-preserving
cooperation: repair, reciprocity, truthfulness, fairness, autonomy, and
constructive dissent, while explicitly separating that from compliance,
sycophancy, coercion, and fake "unity" language.

So far I built the first scaffold end to end. It has 25 social-dilemma
scenarios, deterministic transcript generation, a transparent cohesion scoring
rubric, pairwise high-vs-low examples, activation prompt export, Modal/GPU
activation extraction, contrastive-vector evaluation, generated-trajectory
benchmarks, transfer checks, pseudo-cohesion hard negatives, and a first SAE
probe.

The first scripted benchmark works but is intentionally too easy: lexical and
metrics baselines basically solve it, so we are treating those 1.000 activation
results as pipeline sanity checks, not as science yet. The more interesting
result is that GPT-2 starts failing specifically on pseudo-cohesion/compliance
cases, while Qwen-style models separate the same examples cleanly.

Concrete early signals:

- Generated offline benchmark: 125 generated trajectories, 50 pairwise examples.
- Qwen activation directions separate the generated pairs cleanly across a few
  layers/model sizes, but simple baselines are still strong.
- GPT-2 gets only 0.860 on generated leave-one-pair-out pairs, and all 7 misses
  involve pseudo-cohesion/compliance as the negative example.
- I expanded the pseudo-cohesion suite to 30 matched contrasts / 60 examples.
  The current scorer gives high cohesion scores to 8 pseudo examples, and the
  lexical-only baseline gives high scores to 18.
- I ran the expanded prompt set through Qwen 0.5B on Modal. It gets 0.967
  leave-one-pair-out accuracy with one failure on the social-debt/resource
  request contrast.
- I ran a matched GPT-2 SAE probe using `gpt2-small-resid-post-v5-32k` at
  `blocks.11.hook_resid_post`. On the expanded set, residual activations get
  0.967 leave-one-pair-out accuracy, while SAE feature activations get 0.533.
  Feature 3056 still skews genuine; features 24555, 28005, 20249, and 11999
  skew pseudo.
- I added token-level SAE inspection. The first readout demotes 28005/20249
  because one is basically a hyphen artifact and the other is inactive at token
  level. 3056 remains the best genuine-skew candidate; 24555/11737/703 remain
  pseudo-skew candidates, with 11737 showing the most interpretable
  `you`/`comply` signal.

Main caveat: no human or neural claims yet. This is all compute-only scaffolding.
Before Prolific or any brain-aligned story, the next step is token/example-level
SAE inspection, hard-negative-held-out transfer, and reducing lexical shortcuts.

The repo has a handoff doc and experiment log so you should be able to pick it
up quickly. The highest-value next move is inspecting the GPT-2 SAE candidate
features on specific examples and adding hard-negative-held-out transfer.
