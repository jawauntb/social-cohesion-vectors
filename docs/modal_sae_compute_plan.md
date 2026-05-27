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

Convert generated trajectories into the same scored/pair/prompt artifacts:

```bash
uv run python scripts/build_probe_dataset.py \
  --input data/processed/generated_trajectories.jsonl \
  --scored-output data/processed/generated_scored_runs.jsonl \
  --output data/training/generated_pairwise_probe_dataset.jsonl

uv run python scripts/export_activation_prompts.py \
  --input data/training/generated_pairwise_probe_dataset.jsonl \
  --output data/training/generated_activation_prompts.jsonl
```

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

1. Reproduce the scripted scaffold and confirm the current 1.000 results are
   still just sanity checks.
2. Generate trajectories, build generated pairs, export generated activation
   prompts, and extract generated activations.
3. Run Qwen layer sweeps on scripted and generated prompts.
4. Repeat the best generated prompt run on one larger open model.
5. Expand pseudo-cohesion hard negatives and use failures to sharpen the data.
6. Only then spend time on `sae-lens` feature naming or a Prolific pilot.
