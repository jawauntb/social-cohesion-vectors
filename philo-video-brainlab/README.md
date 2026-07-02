# philo-video-brainlab

> Stop recreating the content that worked. **Recreate the cognitive trajectory that made it work.**

A pipeline and review app for predicting social engagement from **brain-response
trajectories** — the estimated fMRI-like path a viewer's cortex takes while watching a
video — using Meta's **TRIBE v2** encoder alongside conventional multimodal features.

## The thesis

Optimizing content by copying "the same kind of content" optimizes the wrong target.
What actually worked was the *cognitive experience* a video produced. So instead of
matching pixels, we try to match **reactions** — the latent cognitive state associated
with successful engagement (likes, comments, shares, saves, retention).

The MVP is deliberately **not** "brain optimization." The MVP is a falsifiable question:

> **Do TRIBE-derived brain trajectories improve prediction of engagement beyond ordinary
> video / audio / text features?**

If yes, the brain features are capturing something genuinely useful and we've earned the
right to build the rest. If no, we still ship a strong multimodal engagement predictor.
Every result is reported as an **ablation** (with-brain vs. without-brain) so the answer
stays honest.

## What it does

1. **Ingest** every video (ours + competitors like *Lectures on Tap*, the Met, etc.) with
   full engagement metrics, captions, and comments.
2. **Extract features** per video — transcript, audio/prosody, visual frames, editing
   rhythm (shot length / cut pacing), and a **TRIBE v2 brain-response trajectory**.
3. **Score engagement** keeping the four+ targets *separate* first (a great philosophy
   video may be high-save / high-comment, not merely high-like), then optionally learn a
   latent structure (weighted / PCA / Pareto).
4. **Learn mappings** from brain trajectory → each engagement target, with the ablation
   against non-brain features baked in.
5. **Compare creators** in a shared latent space — which cognitive regimes reliably
   produce sharing vs. comments vs. retention.
6. **Pre-publication scoring** — before you post a draft, the app estimates expected
   likes / comments / shares / retention and gives **editor notes** ("curiosity collapses
   after second 17", "semantic novelty plateaus here").

## Trajectory, not snapshot

We optimize for brain **dynamics**, not a single reconstructed state. Success on social
media is likely caused less by *where* the brain is than by *how it moves* — surprise,
uncertainty reduction, prediction error, resolution, curiosity, emotional shifts. Every
video is treated as a **path through a cognitive landscape**, and successful videos are
compared by *path shape* rather than content.

## The honest caveat

Current brain-decoding models reconstruct broad semantic/visual representations, not a
viewer's full internal state. So the inferred brain state is a **useful proxy, not ground
truth** — which is exactly why the pipeline treats it as *one feature source among many*
(visual embeddings, audio/prosody, transcript embeddings, editing rhythm, brain
trajectory) and lets the model discover how much it actually adds.

## Architecture

```
Videos + captions + comments + metrics
        │
        ▼
 feature extraction ── transcript · audio/prosody · visual frames · editing rhythm
        │                                        · TRIBE v2 brain trajectory
        ▼
 engagement model ──── likes · comments · shares · saves/reposts · retention
        │
        ▼
 Next.js review app ── pre-post prediction + editor notes
        │
        ▼
 Modal GPU backend
```

## Monorepo layout

```
philo-video-brainlab/
  apps/web/          Next.js dashboard + pre-post scoring UI
  services/modal/    TRIBE v2 inference + feature extraction (Modal GPU jobs)
  packages/db/       Prisma / Postgres schema (videos, metrics, comments, …)
  packages/scoring/  engagement metrics + latent structure + predictive models (Python)
  data/              manifests only — no raw video, no secrets
  scripts/           ingestion helpers
```

## Tech

- **Modal** — serverless GPU for TRIBE v2 batch inference and model serving.
- **Meta TRIBE v2** — predicts fMRI-like brain responses to video/audio/text (gated on
  Hugging Face; request access logged in, then use a read token in backend secrets).
- **Brain2Qwerty** — optional, for language-side brain interpretation.
- **Next.js** on Railway or Vercel; **Postgres** via Prisma.

## Quickstart

```bash
# 1. env
cp .env.example .env    # fill Modal, HF token, DATABASE_URL, platform keys

# 2. install JS workspaces
npm install

# 3. database
npm run db:generate && npm run db:push

# 4. python envs (uv recommended)
uv pip install -e packages/scoring -e services/modal

# 5. run a live prediction smoke on Modal
cd services/modal && modal run -m modal_app.serve::smoke && cd ../..

# 6. dev the dashboard
npm run dev -w apps/web
```

## Roadmap

- [ ] Upload our catalog + a matched *Lectures on Tap* control set (weak / okay / strong)
      at `/training-data`.
- [ ] Verify the pipeline predicts a held-out creator's engagement (proves it generalizes).
- [ ] Report the with-brain vs. without-brain ablation as the go/no-go gate.
- [ ] Cross-creator latent-space map.
- [ ] Editor-notes generation from trajectory landmarks.

See `docs/ARCHITECTURE.md` for details.

## Live deploy

The live app is two services:

- Modal serves the GPU prediction API at `https://<workspace>--philo-video-analyzer.modal.run/`.
- Railway serves the Next.js app from this folder and proxies `/api/predict` to Modal.

Deploy Modal first:

```bash
cd services/modal
modal deploy -m modal_app.serve
curl https://<workspace>--philo-video-analyzer.modal.run/health
curl -X POST https://<workspace>--philo-video-analyzer.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"modal-smoke","caption":"why boredom is a signal"}'
```

Deploy Railway as its own project/service. Use `--new` for the first deploy so the root
`social-cohesion-vectors` Railway app is not overwritten:

```bash
railway up philo-video-brainlab --path-as-root --new --name philo-video-brainlab --detach
railway variable set MODAL_PREDICT_ENDPOINT=https://<workspace>--philo-video-analyzer.modal.run/ --service <philo-service>
railway variable set MODAL_PREDICT_TIMEOUT_MS=300000 --service <philo-service>
railway variable set NEXT_PUBLIC_APP_NAME=philo-video-brainlab --service <philo-service>
railway add --database postgres --json
railway variable set 'DATABASE_URL=${{Postgres.DATABASE_URL}}' --service <philo-service>
railway up philo-video-brainlab --path-as-root --project <philo-project-id> --service <philo-service> --detach
```

`used_brain: true` means the response used real TRIBE predictions. `used_brain: false`
means the endpoint used a deterministic fallback trajectory because TRIBE, media loading,
or credentials were unavailable.

Historical training CSVs are uploaded at `/training-data`. The page persists the raw CSV
and validation summary to Postgres, then the "Process pending uploads" action imports
rows into `Competitor`, `Video`, `Metric`, and `Caption` records. Feature extraction and
model training run after that normalized dataset exists.
