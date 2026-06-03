---
title: 2026-06-03 Raw EEG/fMRI Dataset Checklist
status: draft
date: 2026-06-03
origin: raw EEG/fMRI bridge lane
---

# 2026-06-03 Raw EEG/fMRI Dataset Checklist

## Purpose

This checklist gates dataset access, provenance capture, and the first tiny
manifest for the raw EEG/fMRI bridge lane.

The first concrete target is THINGS-EEG2. Alljoined is the scale replay after
the THINGS-EEG2 path is auditable. NSD and BOLD5000 are fMRI sanity checks.
FACED and IAPS are affective-control references, not social-intervention
evidence.

This is representation-learning infrastructure only. It does not support any
claim that an intervention changes cooperation, empathy, bonding, trust,
social cohesion, subjective state, or neural synchrony.

## Global Access Rules

- Download no raw brain data until the dataset row below has an access status,
  access date, source URL or DOI, license/access note, allowed-use summary, and
  redistribution constraint.
- Keep raw data, downloaded stimuli, generated features, trained encoders, and
  subject-level outputs out of git.
- Store only small schemas, manifests with non-sensitive relative paths, audit
  notes, and reproducible scripts in the repository.
- Treat the most restrictive upstream source as controlling for any derivative
  release. This includes source image corpora as well as brain-response files.
- Record whether each file path points to raw data, minimally processed data,
  derived response arrays, stimulus images, stimulus metadata, or generated
  features.
- Separate stimulus-level features from trial-level brain responses. Join them
  only through explicit dataset, stimulus, subject, session, run, trial, and
  split keys.
- Do not interpret social-scene annotations as measurements of real social
  effect. They are weak covariates to be residualized against visual, semantic,
  and affective controls.

## Dataset Access Checklist

### 1. THINGS-EEG2

Status: first dataset to unlock.

Source anchors:

- THINGS Initiative page: https://things-initiative.org/
- OSF project listed by THINGS: https://osf.io/3jk45/
- Figshare record: https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912
- Paper DOI: https://doi.org/10.1016/j.neuroimage.2022.119754
- Code: https://github.com/gifale95/eeg_encoding

Known fit:

- EEG visual object recognition dataset with 10 subjects, 82,160 trials per
  subject, and 16,740 image conditions from THINGS.
- Includes raw EEG, preprocessed EEG, stimulus images, supporting resources,
  resting-state data, and code links.
- Best first bridge because stimulus identity, repetitions, and EEG responses
  are designed for image encoding/decoding baselines.

Before download:

- [ ] Record access date and exact source URL used for each downloaded bundle.
- [ ] Capture license/access terms for the Figshare/OSF resources and any
  linked THINGS stimulus/image resources.
- [ ] Record whether raw EEG, preprocessed EEG, image set, DNN feature maps,
  and code are downloaded together or separately.
- [ ] Verify subject ids, session/run structure, event markers, stimulus ids,
  repetition counts, channel metadata, sampling rate, reference, filters, and
  preprocessing state.
- [ ] Verify train/test or official split definitions before generating model
  features.
- [ ] Confirm whether stimulus images can be stored locally, used for feature
  extraction, and referenced in derivative manifests.
- [ ] Record any citation language required by the dataset, THINGS, OSF,
  Figshare, or code repository.

Minimum first manifest:

- One row per stimulus image condition before any subject/session joins.
- Add response-path rows only after the stimulus table can round-trip against
  official THINGS-EEG2 ids and splits.
- Keep baseline features visual/semantic/affective only before adding weak
  social-scene annotations.

Allowed first claim:

- A controlled stimulus-feature encoder predicts THINGS-EEG2 responses above
  shuffled-stimulus controls under a documented preprocessing and split policy.

Disallowed first claim:

- Any social-cohesion, human-intervention, empathy, bonding, cooperation, or
  synchrony claim.

### 2. Alljoined-1.6M

Status: scale replay after THINGS-EEG2.

Source anchors:

- Hugging Face dataset: https://huggingface.co/datasets/Alljoined/Alljoined-1.6M
- Project/blog page: https://www.alljoined.com/blog/introducing-alljoined-1-6m
- Paper: https://arxiv.org/abs/2508.18571

Known fit:

- EEG image-viewing dataset in the THINGS lineage with about 1.69M rows, 20
  participants, four sessions per participant, and 16,740 unique images.
- Raw and preprocessed EEG are available, but the full EEG artifacts are not
  the same as the simplified Hugging Face loader subset.
- Useful as a noisy, lower-cost EEG scale test after the THINGS-EEG2 encoder
  path is stable.

Before download:

- [ ] Record access date, repository commit or dataset revision, and download
  route for raw EEG, preprocessed EEG, metadata, and stimulus resources.
- [ ] Capture Alljoined terms of use and the non-commercial
  research/education constraint before deriving features or training models.
- [ ] Confirm whether image files are bundled, linked, or inherited from
  THINGS resources, and record the applicable image-use terms.
- [ ] Verify subject/session ids, device/channel montage, event markers,
  stimulus ids, trial counts, preprocessing state, and quality flags.
- [ ] Map Alljoined stimulus ids to the THINGS-EEG2 manifest where possible;
  explicitly mark unmatched or ambiguous images.
- [ ] Add hardware/noise fields before cross-dataset comparison:
  acquisition_device, channel_count, bad_channels, artifact_rate,
  retained_trials, and quality_notes.

Allowed replay claim:

- Alljoined does or does not reproduce a THINGS-EEG2-style image-feature EEG
  encoder under noisier consumer-grade acquisition and documented controls.

Disallowed replay claim:

- Affordable EEG scale proves any human social effect or intervention effect.

### 3. NSD

Status: fMRI sanity check after the EEG manifest is stable.

Source anchors:

- Dataset site: https://www.naturalscenesdataset.org/
- Data manual: https://naturalscenesdataset.org/NSD_Data_Manual_v1.5.pdf
- AWS registry: https://registry.opendata.aws/nsd/

Known fit:

- 7T fMRI dataset with subjects viewing many natural scenes from COCO under a
  recognition task.
- Strong fMRI stimulus-response alignment, but access requires the NSD data
  access process.
- Useful to test whether the same stimulus-feature table predicts spatially
  stable fMRI responses in visual/semantic/face/body/social-scene-relevant
  regions.

Before download:

- [ ] Complete and archive the data access agreement status, approval date,
  approved user, and permitted storage location.
- [ ] Record exact NSD release/version, data manual version, AWS path or
  download route, and access date.
- [ ] Capture COCO/source-image constraints separately from fMRI constraints.
- [ ] Verify subject, session, run, trial, image id, repetition, shared-image
  flags, split policy, beta version, ROI labels, and preprocessing state.
- [ ] Record whether analysis uses raw BOLD, precomputed betas, ROIs, or
  derived response matrices.

Allowed sanity-check claim:

- The same image-feature families predict NSD fMRI responses under a specified
  ROI/voxel, split, and preprocessing policy.

Disallowed sanity-check claim:

- fMRI alignment validates a prosocial model direction or content
  intervention.

### 4. BOLD5000

Status: smaller fMRI sanity check and diverse-image comparator.

Source anchors:

- Official code/download entry: https://github.com/BOLD5000-dataset/BOLD5000
- Scientific Data paper: https://doi.org/10.1038/s41597-019-0052-3

Known fit:

- fMRI dataset with roughly 5,000 real-world images drawn from SUN, COCO, and
  ImageNet sources.
- Release 2.0 supports functional-response modeling with stimuli, behavior,
  raw MRI resources, and derived materials linked from the project pages.
- Useful because diverse scene images may stress visual, semantic, affective,
  and weak social-scene controls differently from THINGS object images.

Before download:

- [ ] Record Release 2.0 source URL, access date, and all bundle names used.
- [ ] Distinguish code license from data, stimulus, SUN, COCO, and ImageNet
  image-use terms.
- [ ] Verify subject/session/run/trial ids, stimulus source corpus, image file
  names, behavioral task, response type, timing, preprocessing state, and
  beta/percent-signal-change files.
- [ ] Confirm whether any images have source-specific redistribution limits
  that prevent storing thumbnails, features, or image ids in a public artifact.

Allowed sanity-check claim:

- A documented image-feature encoder predicts BOLD5000 fMRI response estimates
  above controls for the selected response representation.

Disallowed sanity-check claim:

- Diverse natural-image fMRI prediction is evidence of human social cohesion
  or intervention effects.

### 5. FACED

Status: affect-control comparator.

Source anchors:

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/
- Synapse project DOI: https://doi.org/10.7303/syn50614194

Known fit:

- EEG dataset with 123 subjects watching 28 emotion-elicitation video clips
  across nine emotion categories.
- Good for affective-control modeling, not for visual-object EEG bridge
  kickoff and not for prosocial claims.

Before download:

- [ ] Record Synapse account/access status, project id, approval or terms
  acceptance state, access date, and permitted use.
- [ ] Capture whether video stimuli, EEG recordings, behavioral labels, and
  metadata have separate redistribution constraints.
- [ ] Verify subject ids, session/trial/video ids, emotion labels, timing,
  sampling rate, channel montage, preprocessing state, and any excluded trials.
- [ ] Add stimulus_type = video and avoid joining FACED rows to the THINGS
  image-only manifest without a separate video manifest extension.

Allowed comparator claim:

- Affective labels or video-derived features predict FACED EEG responses under
  the documented protocol and controls.

Disallowed comparator claim:

- Emotion-category prediction measures empathy, bonding, prosociality, or
  social cohesion.

### 6. IAPS

Status: controlled affective-stimulus reference only.

Source anchors:

- IAPS request path noted by Image Emotion: https://www.imageemotion.org/
- Primary IAPS access should be recorded from the official distributor used by
  the researcher.

Known fit:

- Standard affective image set with valence/arousal-style normative ratings.
- It is a stimulus set, not a brain-response dataset by itself.
- Useful only as an affective reference, future pilot stimulus source, or
  normative-label comparator after access terms are clear.

Before download or use:

- [ ] Record official distributor, request status, access date, license/access
  terms, citation requirements, storage rules, and redistribution limits.
- [ ] Do not commit IAPS images, thumbnails, file names that reveal restricted
  content, or derived public artifacts prohibited by the license.
- [ ] Record normative fields separately from generated affect estimates.
- [ ] Mark IAPS rows as stimulus_reference_only unless paired with a separately
  approved EEG/fMRI study.

Allowed reference claim:

- IAPS norms provide an affective reference table under controlled access.

Disallowed reference claim:

- IAPS use alone validates neural, behavioral, prosocial, intervention, or
  social-cohesion effects.

## Tiny Manifest Shape

Use JSON Lines or Parquet with this logical shape. Keep paths relative to a
local data root and keep restricted raw data out of git.

```yaml
manifest_version: "raw-eeg-bridge-manifest-v0"
dataset: "THINGS-EEG2"
dataset_version: "recorded release/version or source revision"
access:
  source_url: "https://..."
  access_date: "YYYY-MM-DD"
  license_or_terms: "short human-readable note"
  allowed_use: "research-only / CC BY 4.0 / access-agreement / controlled"
  redistribution: "none / metadata-only / features-ok / cite-required"
provenance:
  source_doi: "https://doi.org/..."
  citation_key: "Gifford2022THINGSEEG2"
  downloaded_by: "initials or system user"
  download_method: "osfclient / datalad / browser / aws / synapse"
  raw_checksum_manifest: "relative/path/to/checksums.tsv"
stimulus:
  stimulus_id: "dataset-native stable id"
  source_image_id: "THINGS / COCO / SUN / ImageNet / IAPS id"
  source_corpus: "THINGS"
  stimulus_type: "image"
  image_path: "restricted/raw/path/or/null"
  image_checksum: "sha256-or-null"
  concept_id: "optional concept id"
  concept_label: "optional object/category label"
  presentation_duration_ms: 100
features:
  feature_row_id: "dataset:stimulus_id:feature_version"
  feature_version: "YYYY-MM-DD-models-v0"
  visual_embedding_path: "features/visual/..."
  semantic_embedding_path: "features/semantic/..."
  affect_feature_path: "features/affect/..."
  weak_social_feature_path: "features/weak_social/..."
  feature_models:
    visual: "model/version"
    semantic: "model/version"
    affect: "model/prompt/version"
    weak_social: "model/prompt/version"
brain_response:
  modality: "EEG"
  response_level: "raw | preprocessed | epoch | beta | roi"
  response_path: "restricted/responses/..."
  response_checksum: "sha256-or-null"
  preprocessing_state: "raw-or-named-pipeline"
  channel_or_roi_space: "64ch BrainVision / ROI atlas / voxel mask"
subject_session:
  subject_id: "sub-01"
  session_id: "ses-01"
  run_id: "run-01"
  trial_id: "trial-000001"
  repetition_index: 1
  onset_ms: 12345
split:
  split_name: "train | val | test"
  split_policy: "held-out-image | held-out-session | held-out-subject"
  split_version: "YYYY-MM-DD-v0"
quality:
  include: true
  exclusion_reason: null
  quality_flags: []
```

## Required Tables

Keep the first implementation small and explicit:

- `dataset_access.tsv`: one row per dataset/source bundle with terms,
  access date, storage location, and redistribution notes.
- `stimuli.parquet`: one row per stimulus id with source corpus, image path,
  image checksum, concept/category labels, stimulus type, and provenance.
- `features.parquet`: one row per stimulus id and feature version with paths
  to visual, semantic, affective, and weak social feature arrays plus model
  provenance.
- `responses.parquet`: one row per subject/session/run/trial/stimulus id with
  response path, modality, response level, preprocessing state, timing, and
  quality flags.
- `splits.parquet`: one row per stimulus or trial key with split name, split
  policy, and split version.

## First THINGS-EEG2 Gate

The THINGS-EEG2 gate is complete when:

- [ ] `dataset_access.tsv` records source URLs, dates, license/access notes,
  and redistribution constraints for raw EEG, preprocessed EEG, stimuli, code,
  and any generated features.
- [ ] `stimuli.parquet` can enumerate THINGS-EEG2 image ids and source paths
  without reading EEG.
- [ ] `responses.parquet` can enumerate subject/session/run/trial/stimulus ids
  and response paths without loading full arrays into git-tracked artifacts.
- [ ] `splits.parquet` has a documented held-out-image baseline split.
- [ ] A small hand-audited subset confirms that stimulus ids, image paths,
  response paths, subject/session ids, and splits join without duplicates or
  orphaned rows.
- [ ] The first encoder report states that it is a representation-learning
  baseline and contains no human-intervention or social-cohesion claim.
