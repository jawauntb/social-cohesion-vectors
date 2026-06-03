---
title: 2026-06-03 Raw EEG/fMRI Bridge Registry
status: kickoff
date: 2026-06-03
origin: Spencer transcript lane 2
---

# 2026-06-03 Raw EEG/fMRI Bridge Registry

## Decision

Do not start this bridge with Tribe or any other derived brain-aligned proxy.
Start from raw or minimally processed human brain-response datasets where the
stimulus identity, stimulus timing, subject/session metadata, and provenance
can be audited.

The Spencer-transcript lesson is methodological: if we want a credible bridge
between model-derived social-state features and human responses, the first
bridge should be stimulus-to-brain encoding, not a claim that a black-box
brain-aligned model has discovered social cohesion. The immediate task is to
map each stimulus to semantic, affective, and weak social features, then test
whether those features predict EEG or fMRI responses under strict controls.

This is representation-learning infrastructure, not evidence that any content
changes real human cooperation, bonding, empathy, synchrony, or prosocial
behavior.

## Bridge Question

Can stimulus-level features derived from images, captions, affect models, and
social-scene annotations predict measured brain responses in open EEG/fMRI
datasets?

The first answer should be an encoder baseline:

```text
stimulus image/text/features -> subject/session-aware encoder -> EEG/fMRI response
```

Only after this baseline is stable should the project ask whether CK-1-style
model activation directions or social-state features add incremental predictive
value beyond ordinary visual, semantic, and affective controls.

## Registry

| Dataset | Modality | Stimulus/task fit | Raw-data fit | Why it matters | Caveats |
| --- | --- | --- | --- | --- | --- |
| Alljoined-1.6M | EEG | THINGS-lineage image-viewing corpus; 20 participants; about 1.69M rows; 16,740 unique images | Raw and preprocessed EEG are available, but not through the simple Hugging Face datasets loader | Best scale-first kickoff substrate; enough trials to test noisy consumer-grade EEG encoders and subject/session generalization | License is non-commercial research/education; consumer-grade 32-channel wet electrode data should be treated as noisy; verify image rights and downstream model-training terms before any release |
| THINGS-EEG2 | EEG | 10 participants; 82,160 trials per subject; 16,740 THINGS image conditions | Strong raw/preprocessed EEG fit, with stimuli and supporting resources | Cleaner lab-grade comparator for Alljoined; ideal first benchmark for image-feature-to-EEG encoding | Visual object recognition task, not social cognition; high temporal resolution but low spatial specificity |
| THINGS-EEG1 | EEG | 49 participants; 22,248 images from 1,854 object concepts in RSVP streams | Raw data hosted separately on OpenNeuro, with Figshare RDM resources | Useful cross-dataset generalization check across a different THINGS EEG design | The main Figshare artifact emphasizes RDMs; raw-data workflow needs OpenNeuro/OSF handling |
| NSD | 7T fMRI | 8 subjects viewing thousands of natural scenes during recognition task | Strong fMRI response/stimulus alignment after access agreement | Lower-noise spatial sanity check for visual/semantic/social-scene features; good bridge to fMRI image-encoding literature | Requires data access agreement; natural scenes and recognition are not social-intervention tasks |
| BOLD5000 | fMRI | About 5,000 real-world images from SUN, COCO, and ImageNet | Release 2.0 supports functional response modeling; raw MRI/stimuli/behavioral resources are linked from project pages | Smaller fMRI sanity check; diverse image sources may include richer social or scene content than THINGS objects | Code is BSD-3-Clause, but stimulus/source-image terms must be checked separately |
| FACED | EEG | 123 subjects watching 28 emotion-elicitation video clips across nine emotion categories | Open-access for research purposes through Synapse | Relevant affect-control comparator: can test valence/arousal/emotion labels against raw EEG before any social-feature claim | Video stimuli and emotion labels are affective, not prosocial; access terms and stimulus rights need review |
| IAPS | Stimulus set, not EEG/fMRI by itself | Standardized affective images with valence/arousal norms | No brain data unless paired with another study | Useful only as an affective-stimulus reference or for future controlled pilots | Not an open image corpus in the same sense; access/licensing is controlled, and stimuli can be emotionally intense |

## Provenance And Licensing Notes

- Alljoined-1.6M currently states non-commercial research and educational use,
  with raw/preprocessed EEG requiring full repository download rather than the
  Hugging Face datasets subset.
- THINGS-EEG2 is listed on Figshare under CC BY 4.0 and includes raw EEG,
  preprocessed EEG, stimuli, resting-state data, tutorials, and code through
  linked resources.
- THINGS-EEG1's Figshare record is CC BY 4.0 and points to OpenNeuro for raw
  data. Keep Figshare RDM artifacts separate from raw EEG.
- NSD is public only after a data access agreement; cite the access date and
  follow the NSD data manual.
- BOLD5000 code is BSD-3-Clause, while dataset/stimulus use depends on the
  dataset download terms and the underlying SUN/COCO/ImageNet image licenses.
- FACED is described as open-access for research purposes. Confirm Synapse
  terms before downloading, redistributing, or training release-bound models.
- IAPS should be treated as controlled affective research material, not a free
  open image dataset.

No generated derivatives, trained encoders, stimulus tables, or model weights
should be released until the most restrictive upstream term has been checked.

## Feature Map

The first feature table should be stimulus-indexed, not trial-indexed. Trial
rows can then join stimulus features to subject/session/timing metadata.

Required feature families:

- Visual controls: CLIP/DINO/ResNet embeddings, low-level image statistics,
  object category, scene category, face/person presence, and salience.
- Semantic features: image caption, noun/verb/object tags, WordNet or THINGS
  concept labels, text embeddings, and category hierarchy.
- Affective features: valence, arousal, dominance, threat, warmth, tenderness,
  joy, anger, disgust, fear, sadness, and neutral estimates.
- Weak social features: people count, faces, gaze/contact, dyads/groups,
  helping/harm cues, cooperation/conflict cues, agency, boundary/contact cues,
  and whether the image depicts direct social interaction.
- Provenance fields: dataset, stimulus id, source corpus, license/access note,
  image availability, subject/session ids, presentation time, repetition index,
  and preprocessing state.

The social features are hypotheses and covariates, not labels for true human
social effect. They should be residualized against visual, semantic, and affect
features before interpreting any incremental brain-response prediction.

## First Encoder Experiment

### Objective

Train a minimal, auditable encoder that predicts EEG responses from stimulus
features on THINGS-EEG2, then replay the same pipeline on Alljoined-1.6M.

### Data Path

1. Build a stimulus manifest for THINGS-EEG2 with image id, THINGS concept,
   presentation split, repetition count, and source paths.
2. Extract frozen visual and semantic embeddings for each image.
3. Add affective and weak social annotations with model name, prompt/version,
   confidence, and manual-audit flags.
4. Join stimulus features to raw or preprocessed EEG epochs with subject,
   session, trial, time window, and channel metadata.
5. Preserve a no-social-controls baseline before adding social features.

### Model

- Baseline: ridge regression or partial least squares from frozen stimulus
  features to EEG channels/time windows.
- Evaluation: held-out image conditions, held-out sessions, and held-out
  subjects where feasible.
- Metrics: correlation/R2 by subject, channel group, and time window; pairwise
  identification accuracy; permutation tests with shuffled stimulus labels.
- Controls: low-level visual-only, category-only, affect-only, social-feature
  residual after visual+semantic+affect, and random-feature negative controls.

### Initial Time Windows

Use coarse windows before fine-grained modeling:

- 0-100 ms: early visual response controls.
- 100-250 ms: object/category-sensitive response.
- 250-500 ms: richer semantic and affective candidate response.
- 500-800 ms: late decision/attention/task effects.

Do not interpret late effects as social cognition without behavioral/task
support. THINGS is an object-recognition substrate.

### Success Criteria

- Visual and semantic baselines predict above shuffled controls.
- Feature importances are stable across subjects or explicitly subject-specific.
- Social-scene features add incremental prediction only after visual, semantic,
  and affective controls.
- The same pipeline can run on Alljoined with an explicit noise/quality report.
- Negative controls fail cleanly.

## Alljoined Replay

Alljoined is the scale stress test, not the first source of truth. After the
THINGS-EEG2 encoder works:

1. Rebuild the same stimulus manifest for Alljoined's THINGS-lineage images.
2. Compare stimulus overlap and concept alignment with THINGS-EEG2.
3. Train within-subject, cross-session, and cross-subject encoders.
4. Quantify what consumer-grade hardware changes: channel reliability,
   artifact burden, time-window stability, and decoding ceiling.
5. Report whether extra scale compensates for noisier acquisition.

The strongest possible Alljoined claim at this stage is about affordable EEG
encoding feasibility. It is not a claim about intervention, social bonding, or
human cooperation.

## fMRI Sanity Check

Use NSD or BOLD5000 after the EEG encoder has a clean manifest and controls.
The fMRI question is whether the same stimulus feature table predicts more
spatially stable brain responses in visual, semantic, face/body, affective, or
social-scene-relevant regions.

Candidate analysis:

- Train voxel/ROI encoders from the same feature families.
- Compare visual-only, semantic, affect-only, and residual social-feature
  models.
- Treat face/person/social-scene signals as scene-understanding signals unless
  the task and behavior justify stronger language.
- Use fMRI only as a lower-noise encoding sanity check, not as validation that
  generated content changes social behavior.

## Claim Boundaries

Allowed claims after a successful first pass:

- A stimulus-feature encoder predicts measured EEG/fMRI responses above
  shuffled controls.
- Certain semantic, affective, or weak social-scene features have incremental
  predictive value in a specific dataset, task, and preprocessing pipeline.
- Alljoined can or cannot reproduce THINGS-EEG2-style encoding under noisier,
  cheaper EEG acquisition.

Disallowed claims without additional validation:

- The project has found a human social-cohesion signature.
- CK-1 or any generated content changes human cooperation, empathy, bonding,
  affiliation, trust, or neural synchrony.
- EEG/fMRI alignment proves a model activation direction is prosocial.
- Affect labels, social-scene annotations, or decoder outputs measure real
  subjective states.
- Any intervention is therapeutic, entactogenic, psychedelic-like, or a
  biological drug analog.

Any real human-effect claim requires an appropriate behavioral or neural study:
Prolific, in-person experiments, fMRI, EEG, fNIRS, or hyperscanning with IRB
review where applicable, preregistered endpoints, adverse-effect monitoring,
and manipulation/autonomy safeguards.

## Immediate Tasks

1. Download no raw data until license/access terms are recorded in a manifest.
2. Create a dataset-access checklist for Alljoined, THINGS-EEG2, THINGS-EEG1,
   NSD, BOLD5000, FACED, and IAPS.
3. Build the THINGS-EEG2 stimulus manifest first.
4. Generate visual/semantic/affective/social feature columns for a small image
   subset and manually audit annotation errors.
5. Train the no-social-controls encoder.
6. Add residual social-scene features only after the baseline is reproducible.
7. Replay on Alljoined and write the noise/scale comparison.

## Source Anchors

- Alljoined-1.6M: https://huggingface.co/datasets/Alljoined/Alljoined-1.6M
- THINGS-EEG2: https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912
- THINGS-EEG1: https://figshare.com/articles/dataset/THINGS-EEG/14721282
- NSD: https://www.naturalscenesdataset.org/ and https://registry.opendata.aws/nsd/
- BOLD5000: https://github.com/BOLD5000-dataset/BOLD5000
- FACED paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/
- IAPS overview/reference path: https://www.imageemotion.org/
