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
- On the small hand-authored pseudo-cohesion set, Qwen 3B gets 4/4 while GPT-2
  gets 3/4 and misses a compliance-maximizing case.
- I ran a first matched GPT-2 SAE probe using
  `gpt2-small-resid-post-v5-32k` at `blocks.11.hook_resid_post`. Tiny n, but it
  surfaces candidate sparse features: 3056 and 28005 higher on genuine cohesion;
  24555 and 703 higher on pseudo-cohesion.

Main caveat: no human or neural claims yet. This is all compute-only scaffolding.
Before Prolific or any brain-aligned story, the next step is to expand the
pseudo-cohesion hard negatives, reduce lexical shortcuts, and see whether the
directions/features survive generated and held-out tests.

The repo has a handoff doc and experiment log so you should be able to pick it
up quickly. The highest-value next move is expanding the pseudo-cohesion dataset
and inspecting the GPT-2 SAE candidate features on more examples.
