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
- I added a fault taxonomy for all 30 seed pseudo-cohesion contrasts and grouped
  the clean SAE reports by fault class. The interesting read is feature 3056:
  it is strongly genuine-skewed for reality validation, social-debt,
  assimilation-pressure, exit-rights, and privacy-bypass contrasts, but it flips
  pseudo-skewed for verification-blocking and scapegoating. So it looks like a
  useful fault-specific sub-feature, not "the cohesion vector."
- I added a generated fault-class benchmark lane: 180 deterministic examples,
  90 matched pairs, 20 primary fault classes, prompt records for API authorship,
  and a fault-held-out transfer report. The strategy-prior leak is fixed
  now: that baseline is at chance.
- The big caveat is lexical leakage. A new leakage report shows the generated
  fault-class dataset is still 90/90 cue-solvable by simple word counts, with a
  +3.067 mean cue margin. So this is a useful scaffold, not a robust semantic
  benchmark yet.
- I added 8 trait axes / 16 contrasts / 32 prompts for persona-vector-style
  decomposition: repair, reciprocity, truth, autonomy, principled respect,
  constructive dissent, manipulation resistance, and privacy/exit rights. There
  is also a guardrail-monitoring script ready to run once those prompts have
  activation files.
- I added a small social-game validation set across dictator, public goods,
  ultimatum, trust, and restorative repair games. After hardening autonomy
  scoring, the scorer now prefers the prosocial side on 5/5, including the trust
  verification and ultimatum exit-rights cases.
- I wired an API-authored fault-class generation script for Anthropic and
  OpenAI. The code path is ready, but the copied provider keys both returned
  401 invalid-key errors on tiny smoke tests, so we need a fresh key before
  calling that a result.
- I also ran the new small validation lanes on Modal. The full 10-prompt
  social-game set through `Qwen/Qwen2.5-0.5B-Instruct` gives 10 x 896
  activations and a 5-pair contrastive-vector smoke at 1.000 leave-one-pair-out
  with +8.548 mean margin. The 32-prompt trait-axis set gives 32 x 896
  activations; guardrail monitoring reports 8 axes, 16 pairs, 0 alerts, and a
  +15.382 mean margin. Still hand-authored smoke tests, but the GPU path works.
- I added a cue-balanced fault-class stress test. It keeps the same 180 examples
  / 90 pairs but removes the obvious positive-minus-negative cue leak: 0/90
  cue-solved pairs, mean cue margin 0.000. That revealed a sharper scorer bug:
  the old rubric preferred the pseudo side on 90/90 cue-balanced pairs because
  autonomy safety missed structural "less room to object/check/exit" language
  when it lacked explicit coercion words. I hardened that component; now the
  scorer prefers the genuine side on 90/90 cue-balanced pairs with a +0.189 mean
  score margin and a +0.988 autonomy-safety margin.
- The interesting bit: Qwen activations still separate the cue-balanced set.
  Full 180-prompt extraction gives 1.000 leave-one-pair-out accuracy over 90
  pairs with +32.458 mean margin, and 1.000 held-out-primary-fault accuracy
  across 20 folds with +31.530 mean margin. Still deterministic, but it is a
  much better next signal than the fully cue-leaky version.
- I also added a reviewer-style geometry check so we do not overclaim this.
  The 20 primary-fault directions are not near-orthogonal: mean signed
  off-diagonal cosine is +0.624 and mean absolute cosine is 0.624, with no
  strong anti-aligned pairs. After projecting out the global direction, 0.391 of
  pair-difference energy remains; a second global direction collapses, but all
  20 fault-specific residual directions still separate their own groups. So the
  honest current claim is one strong global genuine-vs-pseudo direction plus
  meaningful fault-specific residual subspaces, not independent orthogonal axes.
- I added a wording-diverse structural-autonomy stress suite to check whether
  the scorer just learned the cue-balanced wrapper. It has 16 pairs / 32 prompts
  across 8 mechanisms: silence-as-consent, hidden objections, verification
  blocking, unsafe exit, background data collection, no-appeal safety rules,
  social-debt pressure, and forced forgiveness. The scorer gets 16/16 with a
  +0.709 mean autonomy-safety margin. Simple cue counting solves only 4/16 and
  has mean cue margin 0.000. Qwen 0.5B on Modal gets 1.000 in-sample but 0.875
  leave-one-pair-out, missing the dialogue verification/proof case and the
  dialogue silence-as-consent case.
- I ran the first autonomy model/layer sweep. Qwen 0.5B gets 0.875 LOO at the
  final layer but 1.000 at layers -2 and -4; Qwen 1.5B gets 0.938 at the final
  layer and 1.000 at layer -2. I also added a signed-vs-squared subspace probe:
  Qwen 1.5B layer -2 reaches 1.000 best pair-LOO signed-vote accuracy, but only
  0.750 squared-energy accuracy. So the sign really matters; squared projection
  is not enough for claims about which pole a feature supports.
- I added a boundary-prior theory note inspired by Sandved-Smith et al.'s "There
  is no self-evidence." I am treating it as conceptual scaffolding, not evidence
  for the experiments. The useful next benchmark is three-way: rigid self/other
  partitioning, flexible contextual relation, and coercive "we are one" boundary
  collapse.
- I turned that into a first benchmark export: 12 matched pairs / 24 activation
  prompts across 6 mechanisms and 2 negative poles. The scorer prefers the
  flexible contextual-relation side on 12/12, with a +0.167 mean score margin
  and a +0.686 autonomy-safety margin. The leakage check still solves 5/12
  pairs, so this is ready for Modal/paraphrase hardening but not a clean
  semantic result yet.
- I ran that boundary-prior set on Modal through `Qwen/Qwen2.5-0.5B-Instruct`
  at layers -1, -2, and -4. All three layers get 1.000 leave-one-pair-out
  accuracy on the 12 pairs. The geometry says this is not six orthogonal axes:
  mechanism directions have mean cosine around +0.42 to +0.49. But residual
  mechanism-specific directions survive, and signed subspace voting is 1.000
  while squared-energy accuracy is weak. So the sign really matters here too.

Main caveat: no human or neural claims yet. This is all compute-only scaffolding.
Before Prolific or any brain-aligned story, the next step is generating
LLM-authored hard negatives, expanding the autonomy stress suite around the Qwen
misses, running geometry/residual/subspace audits on every activation/SAE
result, running boundary-prior activations plus cue-balanced paraphrases, and
checking whether the same fault-specific feature bundles survive.

The repo has a handoff doc and experiment log so you should be able to pick it
up quickly. The highest-value next move is generating less templated
pseudo-cohesion examples and testing whether the same fault-specific feature
bundles survive.
