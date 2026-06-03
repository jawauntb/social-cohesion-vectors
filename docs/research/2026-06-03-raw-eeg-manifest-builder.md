---
title: 2026-06-03 Raw EEG Manifest Builder
status: draft
date: 2026-06-03
origin: THINGS-EEG2 bridge lane
---

# 2026-06-03 Raw EEG Manifest Builder

## Scope

The first builder is a tiny THINGS-EEG2-first manifest path for local metadata
stubs. It reads CSV, TSV, or JSONL tables that name provenance, stimulus ids,
feature paths, response paths, subject/session/trial keys, and split labels. It
does not read raw EEG arrays, image bytes, feature arrays, model artifacts, or
subject-level data.

This is representation-learning infrastructure only. It supports the narrow
claim that documented visual, semantic, affective, or weak social-scene feature
families can be joined to EEG response metadata under an auditable split policy.
It does not support any human behavioral, intervention, empathy, bonding,
cooperation, social-cohesion, or neural-synchrony claim.

## Inputs

The builder expects five small local tables:

- `dataset_access.tsv` or `.csv`: one THINGS-EEG2 row with source URL, access
  date, terms, redistribution note, DOI, citation key, download method, and
  checksum-manifest path.
- `stimuli.csv` or `.jsonl`: one row per stable THINGS-EEG2 stimulus id with
  source image id, source corpus, stimulus type, restricted image path,
  checksum, concept id/label, and presentation duration.
- `features.csv` or `.jsonl`: one row per stimulus id and feature version with
  relative visual, semantic, affective, and weak-social feature paths plus model
  provenance.
- `responses.csv` or `.jsonl`: one row per subject/session/run/trial/stimulus
  id with a relative response path, EEG response metadata, timing, repetition,
  inclusion status, and quality flags.
- `splits.csv` or `.jsonl`: one row per stimulus id or full trial key with
  split name, split policy, and split version.

All paths are metadata pointers. They should be relative to the local data root
and should point outside git when they reference restricted stimuli, responses,
or generated arrays.

## Output

`scripts/build_raw_eeg_manifest_stub.py` writes JSONL rows using
`raw-eeg-bridge-manifest-v0`. Each row contains:

- access and provenance fields from the dataset access stub;
- stimulus identity and source-corpus fields;
- feature version, feature paths, and model provenance;
- response path, modality, response level, preprocessing state, and channel
  space;
- subject/session/run/trial/repetition/timing fields;
- split policy fields;
- inclusion and quality fields;
- an explicit representation-learning-only claim boundary.

## Validation

`validate_raw_eeg_manifest` checks that rows are non-empty, use THINGS-EEG2 by
default, retain the claim boundary, include required sections, keep image,
feature, and response paths relative, avoid duplicate response trial keys, and
keep response stimulus ids aligned with stimulus rows.

`build_raw_eeg_manifest` additionally rejects missing dataset access rows,
missing required columns, duplicate table keys, orphaned response stimulus ids,
or split rows that do not match the response/stimulus keys.

## Example Command

```bash
uv run python scripts/build_raw_eeg_manifest_stub.py \
  --dataset-access data/local_stubs/things_eeg2/dataset_access.tsv \
  --stimuli data/local_stubs/things_eeg2/stimuli.csv \
  --features data/local_stubs/things_eeg2/features.jsonl \
  --responses data/local_stubs/things_eeg2/responses.csv \
  --splits data/local_stubs/things_eeg2/splits.csv \
  --output data/processed/things_eeg2_raw_eeg_manifest_stub.jsonl
```

The example paths are intentionally local metadata stubs. Do not commit raw
data, downloaded stimulus images, generated feature arrays, EEG response arrays,
or trained encoders.
