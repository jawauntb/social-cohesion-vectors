# Project site

A self-contained static site for **social-cohesion-vectors** — an overview of the
research question (genuine vs. pseudo-cohesion in model activations), the current
findings, the symbolic fault taxonomy, the method lanes, and an honest scope of
what the evidence does and doesn't support.

## Files

- `index.html` — the page (no build step, no dependencies)
- `styles.css` — styling
- `app.js` — scroll reveals + the background "vector field" metaphor

## View locally

Open `index.html` directly, or serve the folder:

```bash
python -m http.server -d site 8000
# then visit http://localhost:8000
```

## Deploy

`.github/workflows/deploy-site.yml` publishes this folder to GitHub Pages on every
push to `main` that touches `site/`. Enable it once under
**Settings → Pages → Build and deployment → Source: GitHub Actions**.

The page is plain HTML/CSS/JS, so it also drops straight into Netlify, Vercel,
Cloudflare Pages, or any static host with the publish directory set to `site`.
