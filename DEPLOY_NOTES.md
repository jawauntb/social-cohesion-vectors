# Deploy notes — video analyzer + site (hand this to your local agent)

Everything below is already committed to `main` of
`jawauntb/social-cohesion-vectors`. This note is for running it from a local
machine that has Modal + the HF token. Two deliverables:

1. **Modal endpoint** — TRIBE v2 brain-response → engagement (`video_analyzer`).
2. **Static site** — `site/` (overview + `train.html`: CSV trainer + a panel that
   calls the endpoint). Deploys to GitHub Pages.

Do them in this order: **env → deploy Modal → verify → configure Pages → point site
at the endpoint → smoke test.**

---

## 0. Prereqs

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/), Node 20+.
- A Modal account + CLI: `pip install modal && modal token new`.
- A Hugging Face token **with TRIBE v2 access granted** (request access at
  https://huggingface.co/facebook/tribev2 while logged in).
- An A100 is required — TRIBE's trimodal pipeline needs ~40 GB VRAM (the code
  sets `gpu="A100"`).

```bash
git clone https://github.com/jawauntb/social-cohesion-vectors.git
cd social-cohesion-vectors
uv sync --group dev            # add: --extra ml if that extra exists in your pyproject
```

---

## 1. Sync the env

The Modal app builds its secret **from the local `.env`** automatically
(`src/social_cohesion_vectors/modal_app/app.py` → `dotenv_secret()`), so the
container gets whatever is in `.env`. Just fill it:

```bash
cp .env.example .env
```

Set at minimum:

- `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) — TRIBE-enabled token.
- `TRIBE_MODEL_ID=facebook/tribev2` (default)
- `TRIBE_GPU=A100`, `TRIBE_TR_SEC=1.49` (defaults are fine)
- `TRIBE_GIT_REF` — **pin to a known-good tribev2 commit** for reproducible image
  builds (defaults to `main`).
- `MODAL_WORKSPACE` — your Modal workspace (affects the deployed URL).

> The `.env` is gitignored — keep it that way.

If you'd rather use a named Modal secret than `.env`, create one and it's already
picked up as long as the keys are also present locally when you deploy:
`modal secret create hf-token HF_TOKEN=hf_xxx`.

---

## 2. Deploy the Modal endpoint

```bash
modal deploy social_cohesion_vectors.modal_app.functions.video_analyzer
```

- Publishes an ASGI web app labeled `video-analyzer`:
  `https://<workspace>--video-analyzer.modal.run/`
  (for workspace `generalintelligencecompany` this is exactly
  `https://generalintelligencecompany--video-analyzer.modal.run/`, which is the
  site's default).
- If Modal says the `video-analyzer` subdomain is already owned by
  `audience-vectors-dev`, deploy into that app name instead:
  ```bash
  SCV_MODAL_APP_BASE_NAME=audience-vectors-dev \
    modal deploy social_cohesion_vectors.modal_app.functions.video_analyzer
  ```
- The image installs TRIBE with `exca==0.5.25` pinned + a build-time import
  preflight (see `docs/runbooks/tribe-modal-startup-fix.md`). **If the image build
  fails at the preflight**, pin `TRIBE_GIT_REF` to a validated commit and confirm
  the `exca` version still matches TRIBE's `neuralset` dependency, then redeploy.
- First request cold-starts the container and downloads TRIBE weights to a Modal
  Volume (`scv-tribe-weights`); subsequent calls are warm.

CLI smoke (no web):

```bash
modal run social_cohesion_vectors.modal_app.functions.video_analyzer::smoke
```

---

## 3. Verify the endpoint

```bash
# health — should report the model loaded
curl -s https://<workspace>--video-analyzer.modal.run/health
# -> {"ok":true,"model":"facebook/tribev2","loaded":true}

# a real prediction (video URL or caption)
curl -s -X POST https://<workspace>--video-analyzer.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"t1","url":"https://youtu.be/XXXX","caption":"opening hook"}'
```

Contract returned (this is what the site expects — keep it stable):

```json
{
  "video_id": "t1", "model_version": "facebook/tribev2", "used_brain": true,
  "likes": 0.7, "comments": 0.6, "shares": 0.5, "saves": 0.6, "retention": 0.8,
  "latent_score": 0.64,
  "editor_notes": [ {"tSec": 17, "kind": "curiosity", "message": "...", "severity": 2} ]
}
```

If `"loaded": false` or `"used_brain": false`, TRIBE didn't load (token/access/
image) and it's returning the deterministic fallback — fix the token/build before
trusting numbers.

---

## 4. Deploy the site (GitHub Pages)

GitHub Pages serves the `main` branch root. The root `index.html` and
`train.html` are tiny redirects into `site/`, and `.nojekyll` keeps GitHub from
running a Jekyll build over the repository.

1. GitHub → repo **Settings → Pages → Build and deployment → Source: Deploy from
   a branch**.
2. Branch: `main`; folder: `/ (root)`.
3. Live at: `https://jawauntb.github.io/social-cohesion-vectors/train.html`

(The repo is public, so no paid plan needed. `site/index.html` = overview,
`site/train.html` = the tool; root `train.html` redirects there.)

---

## 5. Point the site at the endpoint

`train.html` Step 5 has an **"Analyzer endpoint"** field, remembered in the
browser (localStorage), defaulting to
`https://generalintelligencecompany--video-analyzer.modal.run`.

- If your workspace is `generalintelligencecompany`, it already matches —
  nothing to do.
- Otherwise paste your `https://<workspace>--video-analyzer.modal.run/` into that
  field once, **or** change the default in `site/train.html` (search for
  `brainlab_video_endpoint`) and push.

CORS is handled server-side (the ASGI app sets `allow_origins=["*"]`), so the
static Pages site can call it directly from the browser.

---

## 6. Use it

- **CSV trainer** (Steps 1–4, fully client-side): upload video history with
  engagement columns → trains per-target predictors with cross-validated R² →
  scores future videos (single or bulk CSV). No backend needed.
- **Brain pipeline** (Step 5): paste a video URL → calls the Modal endpoint →
  predicted engagement + editor notes from the cognitive trajectory.

---

## Known TODOs (optional, not blockers)

- **Pin `TRIBE_GIT_REF`** to a validated commit (reproducible builds).
- **Trained engagement head:** `_engagement_from_dynamics` in
  `video_analyzer.py` is a heuristic over trajectory dynamics (velocity /
  curvature / novelty / surprise). To make it real, fit a regressor from your CSV
  history onto these dynamics features and load it from the weights volume — then
  the endpoint's numbers are learned from *your* engagement, not a heuristic. The
  ablation harness for "does brain beat baseline" lives in
  `philo-video-brainlab/packages/scoring` (the sibling scaffold in this repo).
- **`audio_path` branch / batch `.map` scoring** over a CSV of URLs — easy to add
  if you want bulk brain-pipeline scoring instead of one-at-a-time.

## Key files

| Path | What |
|------|------|
| `src/social_cohesion_vectors/modal_app/functions/video_analyzer.py` | endpoint: TRIBE `predict()` → dynamics → engagement + notes |
| `src/social_cohesion_vectors/modal_app/image_factory.py` | `tribe_video_image()` (exca pin + preflight) |
| `src/social_cohesion_vectors/modal_app/app.py` | shared Modal app + `.env` secret |
| `docs/runbooks/tribe-modal-startup-fix.md` | TRIBE import fix |
| `site/train.html` | CSV trainer + brain-pipeline panel |
| `index.html`, `train.html`, `.nojekyll` | GitHub Pages root redirects/config |
| `.env.example` | all env vars |
