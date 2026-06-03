---
title: 2026-06-03 Raw EEG Encoder Scaffold
status: draft
date: 2026-06-03
origin: CK6 raw EEG bridge lane
---

# 2026-06-03 Raw EEG Encoder Scaffold

## Scope

This scaffold advances the raw EEG bridge lane from a metadata-only
THINGS-EEG2-style manifest to the first local encoder dry run:

```text
stimulus feature arrays -> EEG response embedding arrays
```

It is intentionally a stub-array workflow. It can run against tiny local `.npy`
fixtures that mirror the manifest paths, but it must not load restricted raw
EEG, downloaded stimulus images, generated full-scale feature banks, trained
encoders, or subject-level artifacts into git.

This is representation-learning infrastructure only. It may support a future
statement that documented stimulus features predict held-out EEG response
embeddings above shuffled-stimulus controls under a declared split policy. It
does not support neural synchrony, prosociality, social cohesion, empathy,
bonding, cooperation, subjective-state, or human-intervention claims.

## First Encoder Shape

The first encoder should be deliberately small:

- Input manifest: JSONL rows from `raw-eeg-bridge-manifest-v0`.
- Feature source: one relative feature path per row, initially
  `features.visual_embedding_path`.
- Response source: one relative `brain_response.response_path` per row.
- Inclusion gate: `quality.include == true`.
- Split gate: train on rows with `split.split_name == "train"` and evaluate on
  one held-out split such as `test` or `val`.
- Model: ridge-regularized linear map from stimulus features to response
  embeddings.
- Baseline: same model class trained after a deterministic train-row response
  shuffle.
- Metric: held-out mean squared error and improvement over the shuffled
  baseline.

The encoder treats EEG responses as generic response embeddings. Any
time-windowing, channel aggregation, preprocessing, epoching, denoising, or
subject pooling must be represented upstream in the response arrays and
manifest provenance before the encoder reads them.

## Required Report Fields

The scaffold report should include:

- `dataset`
- `manifest_version`
- `feature_field`
- `train_rows`
- `eval_rows`
- `feature_dim`
- `response_dim`
- `ridge_alpha`
- `split_policy`
- `eval_split`
- `mse`
- `shuffle_mse`
- `mse_delta_vs_shuffle`
- `claim_boundary`

The report should avoid causal language. It should describe only the local
stub-array join, fit, and held-out reconstruction metric.

## Guardrails

- Use relative paths resolved under an explicit local data root.
- Reject absolute feature or response paths.
- Reject missing files, missing arrays, non-1D row arrays, and feature/response
  dimension mismatches.
- Reject manifests without train rows and held-out evaluation rows.
- Keep outputs small JSON summaries unless the artifact is explicitly ignored
  by git and stored outside the repository.
- Keep weak social-scene features as controls, not outcomes.

## Example Command

```bash
uv run python scripts/run_raw_eeg_encoder_stub.py \
  --manifest data/local_stubs/things_eeg2/manifest.jsonl \
  --data-root data/local_stubs/things_eeg2 \
  --feature-field visual_embedding_path \
  --eval-split test \
  --output data/processed/things_eeg2_encoder_stub_report.json
```

The command is intended for tiny metadata and array stubs. Full-scale
THINGS-EEG2 feature banks, EEG responses, trained encoders, and restricted
stimuli stay outside git and require the access/provenance gates documented in
the raw EEG dataset checklist.
