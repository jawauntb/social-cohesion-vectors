# philo-video-brainlab Deploy Notes

## Prerequisites

- Railway CLI logged in.
- Modal CLI logged in.
- Hugging Face token with TRIBE v2 access in either `philo-video-brainlab/.env` or the repo root `.env`.
- A100 access on Modal for real TRIBE inference.

## Environment

Copy the example env and fill secrets locally:

```bash
cd philo-video-brainlab
cp .env.example .env
```

The Modal service reads `philo-video-brainlab/.env` first and falls back to the repo root
`.env`. `HF_TOKEN` is preferred; `HUGGINGFACE_TOKEN` is also accepted and copied into
`HF_TOKEN` inside the Modal container.

## Modal

Deploy the prediction endpoint:

```bash
cd philo-video-brainlab/services/modal
modal deploy -m modal_app.serve
```

Expected endpoint:

```text
https://generalintelligencecompany--philo-video-analyzer.modal.run/
```

Smoke it:

```bash
curl https://<workspace>--philo-video-analyzer.modal.run/health
curl -X POST https://<workspace>--philo-video-analyzer.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"modal-smoke","caption":"why boredom is a signal"}'
```

`/health.loaded` reports whether TRIBE loaded. A scoring response with `used_brain: false`
is a deterministic fallback, not a neural claim.

Current production smoke:

```json
{"ok":true,"model":"facebook/tribev2","loaded":true}
```

```json
{"video_id":"modal-prod-smoke","model_version":"facebook/tribev2","used_brain":true,"likes":0.5526,"comments":0.6505,"shares":0.7213,"saves":0.0112,"retention":0.0001,"latent_score":0.3872,"editor_notes":[]}
```

## Railway

The current repository root may already be linked to the static `social-cohesion-vectors`
Railway service. Create a separate project/service for this app:

```bash
railway up philo-video-brainlab --path-as-root --new --name philo-video-brainlab --detach --message "deploy philo video brainlab web"
```

Set runtime env on the new service:

```bash
railway variable set MODAL_PREDICT_ENDPOINT=https://<workspace>--philo-video-analyzer.modal.run/ --service <philo-service>
railway variable set NEXT_PUBLIC_APP_NAME=philo-video-brainlab --service <philo-service>
```

Redeploy the same philo service:

```bash
railway up philo-video-brainlab --path-as-root --project <philo-project-id> --service <philo-service> --detach --message "wire modal endpoint"
```

Smoke the deployed web API:

```bash
curl -X POST https://<philo-railway-url>/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"railway-smoke","caption":"why boredom is a signal"}'
```

Current Railway deployment:

```text
URL: https://philo-video-brainlab-production.up.railway.app
Project ID: 753d7bbc-ea1b-4a5b-b81d-69588b351496
Service ID: 5852896f-d84b-44ca-a1cd-fb3597c8948f
Environment ID: b0ab95c3-01cf-41ba-b7d9-d553e2bf1b29
Deployment ID: 3d4a3695-0b5a-4f3c-bf37-327da7ca7ac3
Image digest: sha256:4a71dac9ff522bf84c62b38b71bf7813c9474170d6f830ac616d5814a72b3571
```

Current end-to-end smoke:

```json
{"video_id":"railway-smoke","model_version":"facebook/tribev2","used_brain":true,"likes":0.5526,"comments":0.6505,"shares":0.7213,"saves":0.0112,"retention":0.0001,"latent_score":0.3872,"editor_notes":[]}
```

## Database

The live scoring flow does not require Postgres. When catalog or ablation pages are added,
attach Railway Postgres or Supabase, set `DATABASE_URL`, then run:

```bash
npm run db:generate
npm run db:migrate
```

## Response Contract

Keep this response shape stable for the web app:

```json
{
  "video_id": "draft-001",
  "model_version": "facebook/tribev2",
  "used_brain": true,
  "likes": 0.51,
  "comments": 0.52,
  "shares": 0.53,
  "saves": 0.49,
  "retention": 0.5,
  "latent_score": 0.51,
  "editor_notes": [{ "tSec": 4.5, "kind": "novelty", "message": "...", "severity": 1 }]
}
```

## TODO

- Update `TRIBE_GIT_REF` when intentionally adopting a newer TRIBE v2 commit.
- Replace the heuristic engagement head with a trained model over the CSV/catalog history.
- Add batch feature persistence into `packages/db` once the catalog ingest is ready.
