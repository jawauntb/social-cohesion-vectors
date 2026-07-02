# Architecture

## Objective, restated

The naive objective is "maximize resemblance to successful content." The better
objective — the one this repo is built around — is:

> **Infer the latent cognitive trajectory associated with successful social
> engagement, and engineer for it.**

That reframes optimization from imitating content to engineering the cognitive
experience content reliably produces.

## Data flow

```
platform APIs / manifests
   → ingest_engagement.py → Postgres (Video, Metric, Comment, Caption)
        → Modal pipeline.process_batch
             → features.extract        (transcript, visual, audio, rhythm)   [baseline arm]
             → tribe_inference         (TRIBE v2 brain trajectory + dynamics) [brain arm]
        → Feature / BrainEmbedding rows
   → scoring.run_ablation  (baseline vs. baseline+brain, grouped CV by creator)
   → engagement models → Prediction rows
   → Next.js: dashboard, latent map, pre-post scoring + EditorNote
```

## The seven stages (from the design note)

1. **Engagement dataset** — one row per video; preserve every target separately.
2. **Latent structure** — try weighted / PCA / Pareto; expect ~2 modes (curiosity vs.
   identity/social-signaling), not one "engagement" scalar.
3. **Brain-state prediction** — TRIBE v2 estimates per-frame cortical representation →
   a trajectory through latent brain space.
4. **Learn mappings** — brain trajectory → likes / comments / shares / retention.
5. **Compare creators** — one shared latent space; which regimes generate which
   engagement; what's unique to us.
6. **Editor feedback** — trajectory landmarks → actionable notes.
7. **Pre-publication scoring** — draft → predicted engagement before posting.

## Why dynamics, not snapshots

We summarize each trajectory by **velocity, curvature, novelty, surprise** (see
`services/modal/modal_app/dynamics.py`) and compare successful videos by *path shape*:

```
high uncertainty → rapid compression → reward → identity signal
```

vs. unsuccessful videos that wander. Path shape is far more reusable for editors than
"copy this style."

## The ablation is the product's integrity

`scoring.run_ablation` trains each target twice — baseline vs. baseline+brain — under
**GroupKFold by creator**, so we measure generalization to held-out creators (the
Lectures on Tap control), not memorization. `AblationResult.verdict()` is the go/no-go
gate. If the brain features don't add R², we ship the multimodal predictor and say so.

## Honest limits

TRIBE-style decoders recover broad semantic/visual structure, not a viewer's full
internal state. The brain trajectory is treated as **one feature source among many**, and
its value is *measured*, never assumed. No human-behavioral or neural claims are made from
model activations alone.

## Deploy

- **Modal** hosts TRIBE inference at the ASGI label `philo-video-analyzer`.
- **Railway** hosts the Next.js app from `philo-video-brainlab/`; use a separate Railway
  project/service from the repository root static app.
- **Postgres** can be attached through Railway or Supabase for catalog and ablation views,
  but the first live `/predict` path does not require a database.
- Set `MODAL_PREDICT_ENDPOINT` in Railway to the deployed Modal URL.
