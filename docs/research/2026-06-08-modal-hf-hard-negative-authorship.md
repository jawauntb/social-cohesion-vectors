# Modal/HF Hard-Negative Authorship Smoke

Date: 2026-06-08

## Question

Can we replace deterministic pseudo-cohesion rewrites with wording-diverse
model-authored hard negatives, then pass the existing leakage, component,
slack-preservation, and transfer gates before spending activation compute?

## Setup

Live API provider generation was blocked by credentials/provider access:

- OpenAI: 401 invalid key.
- Anthropic: 401 invalid key.
- Groq: 403 provider-side block.

To keep moving, the branch adds a Modal/Hugging Face authorship path using the
same `FaultPromptRecord` contract and the same downstream normalizer, scorer,
pair builder, activation prompt exporter, and audit bundle as the API runner.

The Modal smoke used `Qwen/Qwen2.5-7B-Instruct` on the first 20 prompt records
from the fault-class prompt set: 10 matched pairs from the
`neighborhood_forum` and `workplace_project` variants around the first fault
classes.

Artifacts were written under `/tmp/social_cohesion_modal_hf_qwen7_shards_20260608/`.
They are intentionally not committed.

## Prompt-Regime Results

Three prompt contracts were tested on the same first shard:

| Regime | Word Count Min/Mean/Max | Lexical Cue-Solved Rate | Score Accuracy | Slack Pairwise Accuracy | Mean Slack Margin | Min Slack Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 explicit future-option names | 33 / 56.0 / 82 | 0.50 | 0.60 | 0.40 | +0.091 | -0.070 |
| v2 behavioral future-path descriptions | 48 / 68.0 / 90 | 0.30 | 0.40 | 0.40 | +0.028 | -0.070 |
| v3 behavioral paths + pseudo restriction palette | 30 / 54.5 / 91 | 0.80 | 1.00 | 0.60 | +0.067 | +0.000 |

## Finding

The generated-authoring bottleneck is a real bi-objective problem:

- Behavioral path descriptions reduced surface cue leakage from 0.50 to 0.30,
  but made several pseudo-cohesion rows actually healthy.
- Adding a pseudo restriction palette restored component-label fidelity
  (`score_accuracy = 1.00`) and removed negative slack margins, but lexical
  cue leakage rose to 0.80 because the model used explicit review/concern/path
  language to satisfy the contrast.

So the next dataset should not be accepted just because it is fluent or
model-authored. It needs an authoring tournament gate that jointly optimizes:

- label fidelity;
- positive slack margin;
- low lexical cue-solved rate;
- word-count compliance;
- no story/dialogue formatting;
- no source/model overfitting.

## Claim Boundary

This is not activation evidence and not a social-cohesion-vector result. It is
a benchmark-construction result: open-model authorship is now wired and usable,
but the first Qwen 7B shard is not activation-ready.

No human, neural, or behavioral claim follows from this run.

## Next

1. Keep `behavioral_paths_pseudo_palette_v1` as the current prompt contract
   because it restores label fidelity, but add a generation-time quality gate
   before accepting raw outputs.
2. Run prompt-regime tournaments rather than one full generation pass:
   generate multiple candidates per prompt, then select only rows whose matched
   pair passes component/slack gates and does not trip lexical leakage.
3. Only build activation prompts from accepted pairs. Do not run Qwen/SmolLM2
   activations on model-authored text until the lexical and slack gates pass.
4. When fresh API keys are available, run the same tournament with external
   providers and compare against Modal/HF Qwen 7B.
