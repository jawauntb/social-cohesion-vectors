"""Modal video analyzer: TRIBE v2 brain-response trajectory -> engagement prediction.

Built on this repo's existing Modal infrastructure — the shared ``app``, the
``.env``-backed secrets (so ``HF_TOKEN``/``HUGGINGFACE_TOKEN`` reach the
container), the config's ``TRIBE_MODEL_ID``, and ``tribe_video_image`` from
``image_factory`` (which pins TRIBE per the startup runbook).

Real TRIBE v2 API (github.com/facebookresearch/tribev2):

    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)
    df = model.get_events_dataframe(video_path="clip.mp4")   # or text_path / audio_path
    preds, segments = model.predict(events=df)                # preds: (n_timesteps, ~20k vertices)

``preds`` is the predicted whole-brain fMRI response on the fsaverage5 mesh —
i.e. the viewer's cognitive *trajectory* over the clip, which is exactly what we
summarize (velocity / curvature / novelty / surprise) and map to engagement.
TRIBE needs ~40 GB VRAM, so these functions run on an A100.

Endpoint the site calls (CORS-enabled, handles the browser preflight):

    POST /   { "video_id", "url"?, "caption"?, "fps"? }
    ->        { "video_id","model_version","used_brain",
                "likes","comments","shares","saves","retention","latent_score",
                "editor_notes":[ {"tSec","kind","message","severity"} ] }

Deploy:  modal deploy social_cohesion_vectors.modal_app.functions.video_analyzer
"""

from __future__ import annotations

import importlib
import math
import os
import tempfile
import urllib.request
from typing import Any

import modal

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.app import app, modal_secrets
from social_cohesion_vectors.modal_app.image_factory import tribe_video_image

_CFG = get_config()
TRIBE_MODEL_ID = _CFG.model_ids.tribe
# TRIBE's trimodal pipeline needs ~40 GB VRAM — override the repo default GPU.
_GPU = os.environ.get("TRIBE_GPU", "A100")
# fMRI temporal resolution (CNeuroMod TR); used to place editor notes in seconds.
_TR_SEC = float(os.environ.get("TRIBE_TR_SEC", "1.49"))

# Persist downloaded TRIBE / HF weights across warm containers.
_weights = modal.Volume.from_name("scv-tribe-weights", create_if_missing=True)
_WEIGHTS_DIR = "/weights"


# --------------------------------------------------------------------------- #
# Trajectory dynamics — success is how the brain moves, not where it is.
# --------------------------------------------------------------------------- #
def _dynamics(traj) -> dict[str, float]:
    import numpy as np

    if traj.shape[0] < 2:
        return {"velocity_mean": 0.0, "curvature_mean": 0.0,
                "novelty_decay": 0.0, "surprise_mean": 0.0}
    d = np.diff(traj, axis=0)
    vel = np.linalg.norm(d, axis=1)
    if d.shape[0] >= 2:
        a, b = d[:-1], d[1:]
        na = np.linalg.norm(a, axis=1) + 1e-8
        nb = np.linalg.norm(b, axis=1) + 1e-8
        curv = np.arccos(np.clip(np.sum(a * b, axis=1) / (na * nb), -1, 1))
    else:
        curv = np.zeros(0)
    nov = np.array([np.linalg.norm(traj[i] - traj[:i].mean(axis=0)) if i else 0.0
                    for i in range(traj.shape[0])])
    sur = np.zeros(traj.shape[0])
    for t in range(2, traj.shape[0]):
        forecast = traj[t - 1] + (traj[t - 1] - traj[t - 2])
        sur[t] = np.linalg.norm(traj[t] - forecast)
    decay = float(np.polyfit(np.arange(nov.shape[0]), nov, 1)[0]) if nov.shape[0] >= 3 else 0.0
    return {
        "velocity_mean": float(vel.mean()),
        "curvature_mean": float(curv.mean()) if curv.size else 0.0,
        "novelty_decay": decay,
        "surprise_mean": float(sur.mean()),
    }


def _landmarks(traj, sec_per_step: float) -> list[dict]:
    import numpy as np

    notes: list[dict] = []
    if traj.shape[0] < 4:
        return notes
    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    peak = int(np.argmax(vel[: max(2, vel.size // 2)]))
    tail = vel[peak + 1:]
    if tail.size and tail.mean() < 0.5 * vel[peak]:
        t = round((peak + 1) * sec_per_step, 1)
        notes.append({"tSec": t, "kind": "curiosity", "severity": 2,
                      "message": f"Curiosity collapses around second {t:g} (representational change flattens)."})
    nov = np.array([np.linalg.norm(traj[i] - traj[:i].mean(axis=0)) if i else 0.0
                    for i in range(traj.shape[0])])
    if nov.shape[0] >= 3 and np.polyfit(np.arange(nov.shape[0]), nov, 1)[0] < 0:
        t = round(int(np.argmax(nov)) * sec_per_step, 1)
        notes.append({"tSec": t, "kind": "novelty", "severity": 1,
                      "message": f"Semantic novelty peaks near second {t:g}, then plateaus."})
    return notes


def _engagement_from_dynamics(dyn: dict) -> dict:
    """Heuristic mapping until a trained head is loaded from the weights volume.

    Sharing/comments track surprise + curvature; retention tracks non-decaying
    novelty. Replace by loading a fitted regressor over these dynamics features.
    """
    sig = lambda x: 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))
    v, s, c, nd = (dyn["velocity_mean"], dyn["surprise_mean"],
                   dyn["curvature_mean"], dyn["novelty_decay"])
    # dynamics on ~20k-vertex fMRI are large-magnitude; squash to comparable scale
    z = lambda x: x / 50.0
    return {
        "likes": sig(0.6 * z(v) + 0.4 * z(s)),
        "comments": sig(0.8 * z(s) + 0.3 * c),
        "shares": sig(0.7 * z(s) + 0.5 * c),
        "saves": sig(0.5 * z(v) - 0.6 * nd),
        "retention": sig(-1.2 * nd),
    }


# --------------------------------------------------------------------------- #
# Media fetch + TRIBE forward pass.
# --------------------------------------------------------------------------- #
def _download(url: str) -> str:
    dst = tempfile.mktemp(suffix=".mp4")
    if any(h in url for h in ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")):
        ytdlp = importlib.import_module("yt_dlp")
        with ytdlp.YoutubeDL({"outtmpl": dst, "format": "mp4/best", "quiet": True}) as ydl:
            ydl.download([url])
    else:
        urllib.request.urlretrieve(url, dst)
    return dst


def _load_model():
    """Load TRIBE v2, caching weights on the volume. None on failure (-> fallback)."""
    try:
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained(TRIBE_MODEL_ID, cache_folder=_WEIGHTS_DIR)
        _weights.commit()
        return model
    except Exception as exc:  # weights/token/deps missing
        print(f"[video_analyzer] TRIBE load failed ({type(exc).__name__}: {exc}); using fallback")
        return None


def _trajectory(model, url: str | None, caption: str | None, fps: float):
    """Return (trajectory ndarray [n_timesteps, n_vertices], model_id, used_brain, sec_per_step)."""
    import numpy as np

    try:
        if model is None:
            raise RuntimeError("model not loaded")
        if url:
            path = _download(url)
            df = model.get_events_dataframe(video_path=path)
        elif caption:
            tp = tempfile.mktemp(suffix=".txt")
            with open(tp, "w", encoding="utf-8") as fh:
                fh.write(caption)
            df = model.get_events_dataframe(text_path=tp)
        else:
            raise ValueError("provide a video url or a caption")
        preds, _segments = model.predict(events=df)
        traj = np.asarray(preds, dtype=float)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
        return traj, TRIBE_MODEL_ID, True, _TR_SEC
    except Exception as exc:
        print(f"[video_analyzer] fallback trajectory ({type(exc).__name__}: {exc})")
        seed = abs(hash((url or "", caption or ""))) % (2**32)
        rng = np.random.default_rng(seed)
        steps = max(6, int(fps * 20))
        walk = np.cumsum(rng.normal(scale=0.3, size=(steps, 64)), axis=0)
        walk[3:6] += rng.normal(scale=1.5, size=(3, 64))
        return walk.astype("float32"), f"{TRIBE_MODEL_ID}#fallback", False, 1.0 / max(fps, 1e-6)


def analyze_payload(model, payload: dict[str, Any]) -> dict[str, Any]:
    video_id = str(payload.get("video_id") or "draft")
    url = payload.get("url") or None
    caption = payload.get("caption") or None
    fps = float(payload.get("fps") or 1.0)

    traj, model_id, used_brain, sec_per_step = _trajectory(model, url, caption, fps)
    dyn = _dynamics(traj)
    targets = _engagement_from_dynamics(dyn)
    notes = _landmarks(traj, sec_per_step)

    return {
        "video_id": video_id,
        "model_version": model_id,
        "used_brain": used_brain,
        **{k: round(float(v), 4) for k, v in targets.items()},
        "latent_score": round(float(sum(targets.values()) / len(targets)), 4),
        "editor_notes": notes,
    }


# --------------------------------------------------------------------------- #
# Modal function (batch/CLI) + CORS ASGI web endpoint (browser-callable).
# --------------------------------------------------------------------------- #
@app.function(
    image=tribe_video_image(),
    gpu=_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
)
def analyze_video(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_payload(_load_model(), payload)


@app.function(
    image=tribe_video_image(),
    gpu=_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
    min_containers=0,
)
@modal.asgi_app(label="video-analyzer")
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Load TRIBE once per container, not per request.
    model = _load_model()

    api = FastAPI(title="video-analyzer")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/")
    def analyze(payload: dict):
        return analyze_payload(model, payload)

    @api.get("/health")
    def health():
        return {"ok": True, "model": TRIBE_MODEL_ID, "loaded": model is not None}

    return api


@app.local_entrypoint()
def smoke():
    """modal run social_cohesion_vectors.modal_app.functions.video_analyzer::smoke"""
    out = analyze_video.remote({"video_id": "demo-001", "caption": "why boredom is a signal"})
    print(out)
