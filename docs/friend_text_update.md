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
- I also expanded the pseudo-cohesion prompt batch into neutral genre variants:
  120 matched pairs / 240 prompts. On that harder batch, the inspected GPT-2 SAE
  features get 0.825 leave-one-pair-out accuracy as a signed mean-activation
  ensemble. The best single feature is 703 at 0.792; 3056 still skews genuine
  but only gets 0.600 alone, so it is more like a useful sub-feature than "the"
  cohesion feature.
- Then I added a cleaner variant batch with no genre wrapper text: 120 pairs /
  240 prompts using in-text term rewrites and hyphen normalization. The same
  inspected SAE feature ensemble improves to 0.892 leave-one-pair-out accuracy.
  A clean-only run without the seed prompts gets 0.889, and 28005 goes fully
  inactive, which confirms the hyphen-artifact read.

Main caveat: no human or neural claims yet. This is all compute-only scaffolding.
Before Prolific or any brain-aligned story, the next step is token/example-level
SAE inspection, LLM-authored hard negatives, and reducing deterministic rewrite
shortcuts.

The repo has a handoff doc and experiment log so you should be able to pick it
up quickly. The highest-value next move is inspecting the GPT-2 SAE candidate
features on specific examples and adding hard-negative-held-out transfer.
