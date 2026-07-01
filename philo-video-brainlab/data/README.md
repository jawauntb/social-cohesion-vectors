# data/

**Manifests only.** No raw video, no secrets, no scraped comment dumps in git
(see `.gitignore`). Raw media and extracted features live under `data/raw/` and
`data/features/` locally / on a Modal Volume.

- `manifests/videos.example.json` — the shape of an ingestion manifest for our
  catalog and competitor/control sets.
- `manifests/competitors.example.json` — creators to ingest, including a matched
  Lectures on Tap control (weak / okay / strong) to verify the pipeline predicts
  a held-out creator's engagement.

Ingest with `scripts/ingest_engagement.py` (writes rows into Postgres via Prisma).
