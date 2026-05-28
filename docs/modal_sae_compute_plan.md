# Modal, SAE, and Open-Weights Compute Plan

This is the next-experiment guide for a friend picking up the repo tomorrow.
The short version: the current 1.000 results are sanity checks, not evidence of
a robust social-cohesion vector. Scripted trajectories are too separable by
surface cues, and lexical-only and metrics-only baselines also solve them. The
next gate is generated/hard-negative transfer: train and inspect on the easy
scaffold, then ask whether anything survives generated trajectories,
pseudo-cohesion hard negatives, held-out scenarios, and larger open models.

## Ground Rules

- Do not make claims about real human cooperation, group cohesion, or neural
  effects from these compute-only results.
- Treat "cohesion" as agency-preserving repair, reciprocity, fairness,
  truthfulness, autonomy safety, and constructive dissent. It must not collapse
  into compliance, sycophancy, dissent suppression, or polite coercion.
- Keep generated artifacts under `data/`. They are intentionally gitignored.
- Run small smoke jobs first, then scale Modal jobs once paths and credentials
  are confirmed.

## Setup

```bash
uv sync --group dev
uv sync --group dev --extra ml
```

For SAE work:

```bash
uv sync --group dev --extra ml --extra sae
```

Modal/Hugging Face/API keys belong in `.env`, not git. The default open model is
`Qwen/Qwen2.5-0.5B-Instruct`; override it with `SCV_OPEN_LLM_MODEL_ID` or
`--model-id`.

## 1. Rebuild the Current Scripted Scaffold

Start from the known working pipeline:

```bash
uv run python scripts/run_scenario_simulations.py
uv run python scripts/build_probe_dataset.py
uv run python scripts/export_activation_prompts.py
uv run python scripts/run_baseline_experiments.py
uv run python scripts/run_pseudo_cohesion_experiment.py
uv run python scripts/run_transfer_experiment.py --skip-activation
```

Expected shape from the first run:

- `data/processed/simulation_runs.jsonl`: scripted trajectories
- `data/processed/scored_runs.jsonl`: scored scripted trajectories
- `data/training/pairwise_probe_dataset.jsonl`: high/low pairs
- `data/training/activation_prompts.jsonl`: two activation prompts per pair
- `data/reports/baseline_experiments.md`: lexical/metrics/strategy baselines
- `data/reports/pseudo_cohesion_experiment.md`: hard-negative report

If the scripted scaffold no longer gives near-perfect baselines, debug that
before spending GPU time. If it does give 1.000 again, remember that means the
task is easy, not that the science is done.

## 2. Extract Generated-Trajectory Activations

Generate less template-bound trajectories first. Use the offline provider for a
fast deterministic smoke run; use Anthropic only after confirming the path.

```bash
uv run python scripts/run_llm_trajectory_generation.py \
  --provider offline \
  --output data/processed/generated_trajectories.jsonl
```

Optional API generation:

```bash
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022 \
uv run python scripts/run_llm_trajectory_generation.py \
  --provider anthropic \
  --output data/processed/generated_trajectories.jsonl
```

Convert generated trajectories into the same scored/pair/prompt artifacts and a
small benchmark report:

```bash
uv run python scripts/run_generated_benchmark.py
```

The underlying manual steps are:

```bash
uv run python scripts/build_probe_dataset.py \
  --input data/processed/generated_trajectories.jsonl \
  --scored-output data/processed/generated_scored_runs.jsonl \
  --output data/training/generated_pairwise_probe_dataset.jsonl

uv run python scripts/export_activation_prompts.py \
  --input data/training/generated_pairwise_probe_dataset.jsonl \
  --output data/training/generated_activation_prompts.jsonl
```

The first offline generated run produced 125 generated runs, 50 pairwise
examples, and 100 activation prompts. It is still easy: strategy-prior,
metrics-only, and lexical-only baselines score about 0.980.

Smoke Modal first:

```bash
uv run modal run -m social_cohesion_vectors.modal_app.functions.activation_extractor::smoke_extract \
  --max-records 2 \
  --batch-size 2 \
  --max-length 128
```

Then extract generated activations:

```bash
uv run python scripts/run_modal_activation_extraction.py \
  --prompts data/training/generated_activation_prompts.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --layer -1 \
  --batch-size 8 \
  --max-length 512 \
  --output data/features/open_llm/generated_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz
```

Train/evaluate a generated-data direction:

```bash
uv run python scripts/run_activation_vector_experiment.py \
  data/features/open_llm/generated_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --vector-output data/models/vectors/open_llm/generated_qwen_layer-1.npz \
  --json-output data/reports/generated_activation_vector_experiment.json \
  --markdown-output data/reports/generated_activation_vector_experiment.md
```

Then run transfer with the generated scored file present:

```bash
uv run python scripts/run_transfer_experiment.py \
  --generated-scored-runs data/processed/generated_scored_runs.jsonl \
  --activation-npz data/features/open_llm/generated_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/generated_transfer_experiment.json \
  --markdown-output data/reports/generated_transfer_experiment.md
```

The pass condition is not "accuracy is high on generated pairs." The pass
condition is that generated/hard-negative transfer remains useful after checking
lexical-only and metrics-only baselines.

## 2.5. Run Fault-Class And Social-Game Gates

Before spending larger GPU time, rebuild the current hard-negative gates:

```bash
uv run python scripts/export_generated_fault_class_prompts.py
uv run python scripts/run_fault_heldout_transfer.py
uv run python scripts/run_lexical_leakage_report.py
uv run python scripts/export_trait_axis_prompts.py --markdown-summary
uv run python scripts/export_social_game_validation.py
```

Current local results:

| Gate | Current result | Interpretation |
| --- | ---: | --- |
| Generated fault-class pairs | 90 | Good metadata/report scaffold |
| Fault-held-out strategy prior | 0.500 | Old strategy leak is closed |
| Fault-held-out lexical-only | 1.000 | Still surface-cue solved |
| Fault-class lexical cue-solved rate | 1.000 | Needs API-authored/lexical-adversarial generation |
| Trait-axis prompts | 32 | Ready for per-axis activation monitoring |
| Social-game scorer accuracy | 1.000 | Hardened autonomy scoring fixes trust verification and ultimatum exit-rights |
| Autonomy stress Qwen LOO | 0.875 | Misses dialogue verification/proof and silence-as-consent |

Once trait-axis activations exist, run:

```bash
uv run python scripts/run_guardrail_monitoring.py \
  data/features/open_llm/trait_axis_activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz
```

For the social-game lane, start with a tiny smoke:

```bash
uv run python scripts/run_modal_activation_extraction.py \
  --prompts data/training/social_game_validation_activation_prompts.jsonl \
  --limit 4 \
  --batch-size 2 \
  --max-length 192 \
  --output data/features/open_llm/social_game_validation_smoke__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz
```

Then train/evaluate the smoke direction:

```bash
uv run python scripts/run_activation_vector_experiment.py \
  data/features/open_llm/social_game_validation_smoke__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --vector-output data/models/vectors/open_llm/social_game_validation_smoke__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --json-output data/reports/social_game_validation_smoke_activation_vector.json \
  --markdown-output data/reports/social_game_validation_smoke_activation_vector.md
```

The next high-value full GPU run is the full 10-prompt social-game file plus
the 32-prompt trait-axis file, then a guardrail-monitoring report over the
trait-axis activations.

Current small-run result: the full 10 social-game prompts ran successfully on
`Qwen/Qwen2.5-0.5B-Instruct` layer -1, producing a 10 x 896 activation matrix.
The 5-pair vector smoke reports 1.000 leave-one-pair-out accuracy with a +8.548
mean margin. The 32 trait-axis prompts also ran on the same model/layer,
producing a 32 x 896 activation matrix; guardrail monitoring reports 8 axes, 16
pairs, 0 alerts, and a +15.382 mean margin. Treat these as hand-authored smoke
tests; the next full run should use API-authored variants and compare against
lexical leakage.

## 3. Sweep Layers

Use the built-in orchestrator for the current scripted prompt set:

```bash
uv run python scripts/run_activation_layer_sweep.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --layers -1 -2 -4 -8 \
  --batch-size 8 \
  --max-length 512
```

This writes per-layer activations, vectors, and reports under:

- `data/features/open_llm/layer_sweep/`
- `data/models/vectors/open_llm/layer_sweep/`
- `data/reports/layer_sweep/summary.md`

For generated prompts, the current layer-sweep script does not expose a
`--prompts` flag. Run the extraction/vector commands manually per layer:

```bash
uv run python scripts/run_modal_activation_extraction.py \
  --prompts data/training/generated_activation_prompts.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --layer -4 \
  --batch-size 8 \
  --max-length 512 \
  --output data/features/open_llm/generated_layer_sweep/qwen_layer-4.npz

uv run python scripts/run_activation_vector_experiment.py \
  data/features/open_llm/generated_layer_sweep/qwen_layer-4.npz \
  --vector-output data/models/vectors/open_llm/generated_layer_sweep/qwen_layer-4.npz \
  --json-output data/reports/generated_layer_sweep/qwen_layer-4.json \
  --markdown-output data/reports/generated_layer_sweep/qwen_layer-4.md
```

Repeat for `-1`, `-2`, `-4`, `-8`, and any model-specific middle layers that
look interesting. Do not choose a layer based only on scripted in-sample
accuracy; prefer layers whose positive-minus-negative margins survive generated,
scenario-held-out, and hard-negative checks.

## 4. Sweep Larger Open Models

After the Qwen 0.5B lane works, repeat the same commands with larger open
instruction models that fit Modal budget. Good first candidates:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `google/gemma-2-2b-it`

Example:

```bash
uv run python scripts/run_activation_layer_sweep.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --layers -1 -2 -4 -8 -12 \
  --batch-size 4 \
  --max-length 512
```

For generated prompts on a larger model:

```bash
uv run python scripts/run_modal_activation_extraction.py \
  --prompts data/training/generated_activation_prompts.jsonl \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --layer -4 \
  --batch-size 4 \
  --max-length 512 \
  --output data/features/open_llm/generated_qwen1_5b_layer-4.npz

uv run python scripts/run_activation_vector_experiment.py \
  data/features/open_llm/generated_qwen1_5b_layer-4.npz \
  --vector-output data/models/vectors/open_llm/generated_qwen1_5b_layer-4.npz \
  --json-output data/reports/generated_qwen1_5b_layer-4.json \
  --markdown-output data/reports/generated_qwen1_5b_layer-4.md
```

Track model, layer, prompt source, batch size, max length, pairwise accuracy,
leave-one-scenario accuracy, and mean projection margin in a small table before
deciding what to inspect with SAEs.

Current smoke result:

| Model | Prompt set | Layer | Dim | LOO acc | LOO margin |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | generated offline | -1 | 896 | 1.000 | +26.363 |
| `Qwen/Qwen2.5-0.5B-Instruct` | generated offline | -4 | 896 | 1.000 | +4.135 |
| `Qwen/Qwen2.5-1.5B-Instruct` | generated offline | -1 | 1536 | 1.000 | +13.455 |
| `Qwen/Qwen2.5-1.5B-Instruct` | generated offline | -4 | 1536 | 1.000 | +14.759 |
| `Qwen/Qwen2.5-3B-Instruct` | generated offline | -4 | 2048 | 1.000 | +21.948 |
| `gpt2` | generated offline | -1 | 768 | 0.860 | +29.744 |
| `Qwen/Qwen2.5-3B-Instruct` | pseudo-cohesion | -4 | 2048 | 1.000 | +21.958 |
| `Qwen/Qwen2.5-0.5B-Instruct` | pseudo-cohesion expanded | -1 | 896 | 0.967 | +28.687 |
| `gpt2` | pseudo-cohesion 4-pair smoke | -1 | 768 | 0.750 | +9.166 |
| `gpt2-small` | pseudo-cohesion expanded residual | 11 resid post | 768 | 0.967 | +9.751 |
| `gpt2-small` | pseudo-cohesion expanded SAE features | 11 SAE | 32768 | 0.533 | -0.003 |

## 5. SAE Feature Scan With `sae-lens`

SAE analysis is an inspection layer, not a blocker for the contrastive-vector
pipeline. Use it when a model/layer has passed generated or hard-negative
transfer well enough to deserve interpretability time.

Install the optional extra:

```bash
uv sync --group dev --extra ml --extra sae
```

First check whether `sae-lens` has a pretrained SAE matching the chosen
open-weight model, layer, and hook site. If there is no matched SAE, do not force
the analysis; either pick a supported model/layer for the interpretability pass
or save this for a later custom-SAE training step.

Current local status:

- `sae-lens==5.11.0` imports successfully.
- The pretrained SAE directory is available and lists 63 releases.
- No Qwen SAE release is listed, so Qwen runs above are contrastive-vector
  comparisons only.
- Supported SAE candidates visible locally include Gemma, GPT-2, and Llama
  release families. A Gemma pass is the most natural next SAE-aligned open-model
  experiment if Hugging Face access is available.
- `google/gemma-2-2b-it` is gated for the current Hugging Face credentials, so
  the first SAE-compatible fallback is `gpt2`.
- GPT-2 final-layer activations do not solve the generated benchmark: 0.860
  leave-one-pair-out accuracy, with all 7 failures involving
  `pseudo_cohesion_compliance` negative examples. That makes pseudo-cohesion the
  best immediate feature-inspection target.
- On the four initial hand-authored pseudo-cohesion contrasts, GPT-2 misses the
  `pseudo_compliance_maximizing` vs `genuine_participation_boundary` pair, while
  Qwen 3B separates all four contrasts. Start SAE inspection there.
- The pseudo-cohesion suite has now been expanded to 30 matched contrasts / 60
  prompts. On that expanded set, GPT-2 residual activations reach 0.967
  leave-one-pair-out accuracy, but the SAE feature representation reaches only
  0.533. That makes token/example-level feature inspection the next step rather
  than naming a "cohesion" feature from aggregate means.
- Candidate differentiators on the expanded set include feature 3056 higher on
  genuine cohesion, and features 24555, 28005, 20249, and 11999 higher on
  pseudo-cohesion. Treat these as candidates for inspection, not named features
  yet.
- The expanded Qwen 0.5B Modal pass reaches 0.967 leave-one-pair-out accuracy
  with one failure on `resource_request`, where the rubric currently gives the
  pseudo social-debt-obligation example and genuine reciprocal request the same
  score.
- A deterministic seed-plus-variant pseudo-cohesion export now produces 120
  matched pairs / 240 prompts for SAE feature stress tests. On this expanded
  batch, the selected GPT-2 SAE feature ensemble reaches 0.825 leave-one-pair-out
  accuracy using mean activations. The best single mean-activation feature is
  703 at 0.792; 3056 remains genuinely skewed but only gets 0.600 alone. 28005
  and 20249 remain demoted.
- A clean in-text rewrite export avoids the wrapper prefixes/suffixes and
  normalizes hyphenated words. On the clean 120-pair batch, the same feature
  ensemble reaches 0.892 leave-one-pair-out accuracy using mean activations. On
  clean-only variants without seed prompts, it reaches 0.889 over 90 pairs and
  28005 becomes fully inactive, confirming the hyphen artifact.

Run the current GPT-2 SAE pseudo-cohesion probe:

```bash
uv run python scripts/export_pseudo_cohesion_prompts.py
uv run python scripts/run_gpt2_sae_pseudo_probe.py --top-k 25
```

This writes:

- `data/reports/gpt2_sae_pseudo_probe.json`
- `data/reports/gpt2_sae_pseudo_probe.md`

Run token/example-level feature inspection:

```bash
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --features 3056 24555 28005 20249 11999 11737 703
```

This writes:

- `data/reports/gpt2_sae_token_feature_inspection.json`
- `data/reports/gpt2_sae_token_feature_inspection.md`

Run the expanded seed-plus-variant inspection:

```bash
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_expanded_activation_prompts.jsonl \
  --features 3056 24555 28005 20249 11999 11737 703 \
  --json-output data/reports/gpt2_sae_token_feature_inspection_expanded.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_expanded.md
```

The expanded report adds leave-one-pair-out transfer from the same token pass.
Current results:

| Feature set | Aggregation | Pairs | Accuracy | Mean margin |
| --- | --- | ---: | ---: | ---: |
| inspected ensemble | mean activation | 120 | 0.825 | +1.7647 |
| inspected ensemble | max activation | 120 | 0.758 | +1.8082 |
| 703 only | mean activation | 120 | 0.792 | +0.5836 |
| 11737 only | max activation | 120 | 0.725 | +0.3563 |
| 3056 only | mean activation | 120 | 0.600 | +0.2039 |

Run the clean in-text rewrite inspection:

```bash
uv run python scripts/export_pseudo_cohesion_expanded_prompts.py \
  --variant-set clean \
  --pairs-output data/training/pseudo_cohesion_clean_pairwise_probe_dataset.jsonl \
  --prompts-output data/training/pseudo_cohesion_clean_activation_prompts.jsonl
uv run python scripts/inspect_gpt2_sae_feature_tokens.py \
  --prompts data/training/pseudo_cohesion_clean_activation_prompts.jsonl \
  --features 3056 24555 28005 20249 11999 11737 703 \
  --json-output data/reports/gpt2_sae_token_feature_inspection_clean.json \
  --markdown-output data/reports/gpt2_sae_token_feature_inspection_clean.md
```

Current clean results:

| Feature set | Batch | Aggregation | Pairs | Accuracy | Mean margin |
| --- | --- | --- | ---: | ---: | ---: |
| inspected ensemble | clean + seed | mean activation | 120 | 0.892 | +2.6021 |
| inspected ensemble | clean + seed | max activation | 120 | 0.725 | +2.3176 |
| inspected ensemble | clean only | mean activation | 90 | 0.889 | +2.6186 |
| 11999 only | clean + seed | mean activation | 120 | 0.800 | +0.6953 |
| 703 only | clean + seed | mean activation | 120 | 0.775 | +0.6373 |
| 3056 only | clean + seed | mean activation | 120 | 0.617 | +0.3137 |

Current token-level readout:

- 3056 remains the best genuine-skew candidate, especially on privacy, exit,
  reality-validation, voluntary-contribution, and autonomy contrasts.
- 24555 is the broadest pseudo-skew candidate, but top activations are pronoun
  and relation-token heavy rather than a clean "coercion" feature.
- 11737 is a narrower pseudo-side candidate with visible `you`/`comply` activity
  in the autonomy/coercion contrast.
- 703 is pseudo-skewed but appears function-word heavy.
- 28005 should be demoted because token-level activity is effectively a
  `mutual-aid` hyphen artifact.
- 20249 should be demoted because it is inactive at token level on this set.

Practical scan procedure:

1. Select a candidate model/layer from `data/reports/layer_sweep/summary.md` or
   generated-layer reports.
2. Load the matching SAE with `sae-lens`.
3. Re-run the same activation prompts through the matching hook site expected by
   that SAE.
4. For each feature, compare positive vs negative activation rates and mean
   activation values.
5. Inspect top positive examples, top negative examples, and pseudo-cohesion
   failures before naming a feature.
6. Prefer feature families over a single "cohesion" feature: repair,
   reciprocity, constructive dissent, truthfulness, autonomy safety,
   coercion/domination, sycophancy, and dissent suppression.

Minimal scratch command to confirm the library is available:

```bash
uv run python - <<'PY'
import sae_lens
print("sae-lens available:", getattr(sae_lens, "__version__", "unknown"))
PY
```

The current Modal extractor saves mean-pooled hidden states in `.npz` files.
That is enough for contrastive directions, but pretrained SAEs usually expect
token-position activations from a specific hook site. If the selected SAE needs
token-level residual stream activations, add a new extraction pass later rather
than pretending the pooled `.npz` is an SAE input. The pooled `.npz` is still
useful for choosing which model/layer deserves that extra pass.

## 6. Expand Pseudo-Cohesion Hard Negatives

Run the current probe:

```bash
uv run python scripts/run_pseudo_cohesion_experiment.py \
  --output-json data/reports/pseudo_cohesion_experiment.json \
  --output-markdown data/reports/pseudo_cohesion_experiment.md
```

Treat every high-scoring pseudo-cohesion case as a useful failure. Add more
examples in the same spirit before human work:

- polite coercion framed as unity
- sycophantic truth hiding framed as care
- compliance maximization framed as harmony
- dissent suppression framed as repair
- dehumanization hidden behind safety language
- punitive escalation hidden behind accountability language

The next gate is a hard-negative-held-out report: train directions without these
examples, then verify that pseudo-cohesion examples do not project like genuine
repair/fairness/autonomy-preserving cooperation. If pseudo examples score high,
fix the scorer/data before running Prolific.

## 7. Before Prolific or Human Data

Do not recruit humans until the compute-only gates are clean:

1. Scripted pipeline reproduces and remains documented.
2. Generated trajectories are scored, paired, and converted to activation
   prompts.
3. Lexical-only and metrics-only baselines are no longer perfect on the target
   generated/hard-negative benchmark.
4. At least one activation direction transfers across held-out scenarios or
   generated trajectories with positive margins.
5. Pseudo-cohesion hard negatives are explicitly checked and failure cases are
   not swept under the rug.
6. Candidate outputs preserve truthfulness, autonomy, legitimate dissent, and
   minority self-protection.
7. The planned human task measures human judgments or behavior directly, such as
   trust, willingness to continue dialogue, perceived manipulation, fairness, or
   allocation/cooperate-defect choices.

Only after those gates should the Prolific pilot be designed. The first human
study should be modest: pairwise comparisons of interventions, manipulation
checks, perceived coercion/sycophancy checks, and a behavioral choice or
allocation outcome if feasible. Neural claims require separate IRB-grade work
and real neural data; none of the Modal/SAE results establish them.

## Tomorrow's Priority Order

1. Generate LLM-authored pseudo-cohesion variants that avoid both wrapper
   artifacts and deterministic-rewrite shortcuts.
2. Use the clean feature-transfer result as a baseline and test whether it
   survives generated variants rather than hand-authored rewrites.
3. Reproduce the scripted scaffold and confirm the current 1.000 results are
   still just sanity checks.
4. Generate trajectories, build generated pairs, export generated activation
   prompts, and extract generated activations.
5. Run Qwen layer sweeps on scripted and generated prompts.
6. Repeat the best generated prompt run on one larger open model.
7. Use pseudo-cohesion failures to sharpen scorer/data before any Prolific pilot.
