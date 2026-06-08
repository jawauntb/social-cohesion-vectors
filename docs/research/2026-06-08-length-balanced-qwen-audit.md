# Length-Balanced Qwen Activation Audit

Date: 2026-06-08

## Summary

The length-balanced generated-fault bundle removes the deterministic
token-count shortcut and still separates in Qwen activation space. This is the
current strongest compute-only result for the generated pseudo-cohesion lane:
the lexical/length baseline is at chance, while signed activation directions
over Qwen 0.5B/1.5B layers -1 and -2 all reach 1.000 leave-one-pair-out
accuracy across 180 matched pairs.

This is not a human, behavioral, or neural claim. The benchmark is still
synthetic deterministic text. The result says that the open-model representation
signal survives the currently enforced lexical, length, source-diversity, slack,
and metadata-transfer gates.

## Inputs

The generated source bundle was regenerated locally at:

`/tmp/social_cohesion_length_balanced_qwen_audit`

Bundle summary:

| Field | Value |
| --- | ---: |
| Styles | `length_balanced`, `length_balanced_alt` |
| Sources | 2 |
| Scored runs | 360 |
| Pairwise examples | 180 |
| Activation prompts | 360 |
| Prompt records | 180 |
| Audit not-ready steps | 0 |
| Audit warnings | 0 |

Non-activation gates:

| Gate | Result |
| --- | ---: |
| Simple cue-solved pairs | 0/180 |
| Slack-prefers-genuine pairs | 180/180 |
| Source duplicate / near-duplicate text pairs | 0 / 0 |
| Max cross-source text similarity | 0.567 |
| Best single lexical diagnostic feature | 0.500 |
| `__log_token_count__` pairwise accuracy | 0.500 |
| `lexical_only` held-out fault/source accuracy | 0.500 / 0.500 |
| `metrics_only` held-out fault/source accuracy | 1.000 / 1.000 |

## Activation Runs

Model/layer sweep:

| Model | Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | -1 | 360 | 896 | 1.000 | 1.000 | +22.742 |
| `Qwen/Qwen2.5-0.5B-Instruct` | -2 | 360 | 896 | 1.000 | 1.000 | +5.729 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -1 | 360 | 1536 | 1.000 | 1.000 | +10.817 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -2 | 360 | 1536 | 1.000 | 1.000 | +19.120 |

Primary activation artifact:

`data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz`

Primary vector report:

`data/reports/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.json`

Primary layer -1 results:

| Metric | Value |
| --- | ---: |
| In-sample pairwise accuracy | 1.000 |
| In-sample mean projection margin | +22.764 |
| LOO pairwise accuracy | 1.000 |
| LOO mean projection margin | +22.742 |
| LOO min projection margin | +19.304 |

Held-out primary-fault transfer:

| Metric | Value |
| --- | ---: |
| Folds | 20 |
| Test pairs | 180 |
| Mean test accuracy | 1.000 |
| Mean test margin | +22.739 |
| Minimum per-fold margin | +19.287 |
| Metadata coverage readiness | `metadata_coverage_ready` |
| Transfer readiness | `transfer_ready` |

The combined source-bundle manifest with the activation file attached reports
`bundle_ready`, 8 ready audit steps, 0 skipped steps, and 0 warnings.

## Geometry

The activation result is strong, but the geometry is more collapsed than the
earlier cue-balanced fault-class geometry. This persists across the expanded
model/layer sweep:

| Model | Layer | Mean signed cosine | Mean absolute cosine | Anti-aligned pairs |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | -1 | +0.978 | 0.978 | 0 |
| `Qwen/Qwen2.5-0.5B-Instruct` | -2 | +0.977 | 0.977 | 0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -1 | +0.963 | 0.963 | 0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -2 | +0.964 | 0.964 | 0 |

This means the current length-balanced deterministic source mostly induces one
shared genuine-vs-pseudo manifold. It does not support an independent
fault-axis claim.

Residual audit:

| Model | Layer | Global acc | Residual global acc | Residual group mean acc | Positive residual groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | -1 | 1.000 | 0.000 | 0.956 | 20/20 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -1 | 1.000 | 0.000 | 0.992 | 20/20 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -2 | 1.000 | 0.000 | 1.000 | 20/20 |

After the global direction is removed, a second global residual direction
collapses, but every primary fault group retains a positive residual signal.
The responsible interpretation is therefore:

> One very strong shared genuine-vs-pseudo direction, plus meaningful
> fault-specific residual structure.

Signed-vs-squared subspace probe:

| Probe | Value |
| --- | ---: |
| Best pair-LOO signed-vote accuracy | 1.000 |
| Best pair-LOO squared-energy accuracy | 1.000 |
| Best basis | `raw_pair_difference_svd` |
| Best components | 1 |

Unlike the boundary-prior subspace audits, squared energy does not lag signed
vote here. On this deterministic source, a one-component subspace already
captures the separation.

## Commands

```bash
.venv/bin/python scripts/run_generated_fault_source_bundle.py \
  --scored-runs-output /tmp/social_cohesion_length_balanced_qwen_audit/scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_length_balanced_qwen_audit/pairs.jsonl \
  --prompts-output /tmp/social_cohesion_length_balanced_qwen_audit/prompts.jsonl \
  --prompt-records-output /tmp/social_cohesion_length_balanced_qwen_audit/prompt_records.jsonl \
  --dataset-json-report /tmp/social_cohesion_length_balanced_qwen_audit/dataset.json \
  --dataset-markdown-report /tmp/social_cohesion_length_balanced_qwen_audit/dataset.md \
  --audit-output-dir /tmp/social_cohesion_length_balanced_qwen_audit/audit \
  --pipeline-json-report /tmp/social_cohesion_length_balanced_qwen_audit/pipeline.json \
  --pipeline-markdown-report /tmp/social_cohesion_length_balanced_qwen_audit/pipeline.md

.venv/bin/python scripts/run_activation_layer_sweep.py \
  --dataset-name length_balanced_fault_bundle \
  --prompts /tmp/social_cohesion_length_balanced_qwen_audit/prompts.jsonl \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --layers -1 -2 \
  --batch-size 8 \
  --max-length 512 \
  --skip-existing

.venv/bin/python scripts/run_activation_metadata_transfer.py \
  data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --pairs /tmp/social_cohesion_length_balanced_qwen_audit/pairs.jsonl \
  --metadata-key primary_fault_class \
  --coverage-metadata-key source \
  --coverage-metadata-key generated_style \
  --coverage-metadata-key generated_variant \
  --required-coverage-metadata-key source \
  --required-coverage-metadata-key generated_style \
  --json-output /tmp/social_cohesion_length_balanced_qwen_audit/activation_metadata_transfer.json \
  --markdown-output /tmp/social_cohesion_length_balanced_qwen_audit/activation_metadata_transfer.md

.venv/bin/python scripts/run_direction_geometry_audit.py \
  data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --pairs /tmp/social_cohesion_length_balanced_qwen_audit/pairs.jsonl \
  --metadata-key primary_fault_class \
  --json-output /tmp/social_cohesion_length_balanced_qwen_audit/direction_geometry.json \
  --markdown-output /tmp/social_cohesion_length_balanced_qwen_audit/direction_geometry.md

.venv/bin/python scripts/run_residual_subspace_audit.py \
  data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --pairs /tmp/social_cohesion_length_balanced_qwen_audit/pairs.jsonl \
  --metadata-key primary_fault_class \
  --json-output /tmp/social_cohesion_length_balanced_qwen_audit/residual_subspace.json \
  --markdown-output /tmp/social_cohesion_length_balanced_qwen_audit/residual_subspace.md

.venv/bin/python scripts/run_activation_subspace_probe.py \
  data/features/open_llm/layer_sweep/length_balanced_fault_bundle__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz \
  --pairs /tmp/social_cohesion_length_balanced_qwen_audit/pairs.jsonl \
  --metadata-key primary_fault_class \
  --json-output /tmp/social_cohesion_length_balanced_qwen_audit/subspace_probe.json \
  --markdown-output /tmp/social_cohesion_length_balanced_qwen_audit/subspace_probe.md
```

## Next Step

The non-Qwen architecture check is now recorded in
`docs/research/2026-06-08-cross-architecture-alignment-audit.md`: SmolLM2
1.7B also separates the length-balanced bundle, and Qwen-to-SmolLM2 linear maps
preserve the signed direction on a bounded pair-LOO diagnostic. The next open
question has therefore moved from "does the signal survive outside Qwen?" to
"does the cross-model-compatible manifold persist across API-authored,
wording-diverse hard negatives?"
