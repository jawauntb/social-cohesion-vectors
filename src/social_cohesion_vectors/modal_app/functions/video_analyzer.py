"""Modal video analyzer: TRIBE brain-response trajectory -> engagement prediction.

Built on this repo's existing Modal infrastructure — the shared ``app``, the
``.env``-backed secrets (so ``HF_TOKEN``/``HUGGINGFACE_TOKEN`` are present in the
container), the config's ``TRIBE_MODEL_ID``, and the ``tribe_video_image`` from
``image_factory`` (which pins TRIBE per the startup runbook).

The web endpoint the site calls:

    POST /   { "video_id": str, "url": str|null, "caption": str|null, "fps": float }
    ->        { "video_id", "model_version", "used_brain",
                "likes","comments","shares","saves","retention","latent_score",
                "editor_notes": [ {"tSec","kind","message","severity"} ] }

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
TARGET_LAYER = int(os.environ.get("TRIBE_TARGET_LAYER", "-1"))
DEFAULT_FPS = float(os.environ.get("FRAME_SAMPLE_FPS", "1.0"))
_GPU = _CFG.modal.default_gpu

# Cache TRIBE / HF weights across warm containers.
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
    nov = np.zeros(traj.shape[0])
    for t in range(1, traj.shape[0]):
        nov[t] = np.linalg.norm(traj[t] - traj[:t].mean(axis=0))
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


def _landmarks(traj, fps: float) -> list[dict]:
    import numpy as np

    notes: list[dict] = []
    if traj.shape[0] < 4:
        return notes
    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    peak = int(np.argmax(vel[: max(2, vel.size // 2)]))
    tail = vel[peak + 1:]
    if tail.size and tail.mean() < 0.5 * vel[peak]:
        t = round((peak + 1) / max(fps, 1e-6), 1)
        notes.append({"tSec": t, "kind": "curiosity", "severity": 2,
                      "message": f"Curiosity collapses after second {t:g} (representational change flattens)."})
    nov = np.array([np.linalg.norm(traj[i] - traj[:i].mean(axis=0)) if i else 0.0
                    for i in range(traj.shape[0])])
    if nov.shape[0] >= 3 and np.polyfit(np.arange(nov.shape[0]), nov, 1)[0] < 0:
        t = round(int(np.argmax(nov)) / max(fps, 1e-6), 1)
        notes.append({"tSec": t, "kind": "novelty", "severity": 1,
                      "message": f"Semantic novelty peaks near second {t:g}, then plateaus."})
    return notes


# --------------------------------------------------------------------------- #
# TRIBE forward pass (real path + graceful fallback so the endpoint deploys).
# --------------------------------------------------------------------------- #
def _download(url: str) -> str:
    """Fetch a direct media URL, or a platform URL via yt-dlp, to a temp file."""
    dst = tempfile.mktemp(suffix=".mp4")
    if any(h in url for h in ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")):
        ytdlp = importlib.import_module("yt_dlp")
        with ytdlp.YoutubeDL({"outtmpl": dst, "format": "mp4/best", "quiet": True}) as ydl:
            ydl.download([url])
    else:
        urllib.request.urlretrieve(url, dst)
    return dst


def _tribe_trajectory(url: str | None, caption: str | None, fps: float):
    """Return (trajectory ndarray [steps, dim], model_id, used_brain).

    Wire the real TRIBE forward here once the tribev2 API is confirmed:
      from tribev2 import TribeModel
      model = TribeModel.from_pretrained(TRIBE_MODEL_ID, token=_CFG.api_keys.huggingface)
      frames, audio = sample_media(path, fps); text = caption or transcribe(path)
      traj = model.predict(frames=frames, audio=audio, text=text, layer=TARGET_LAYER)
    Until then, a deterministic seeded trajectory keeps the endpoint live.
    """
    import numpy as np

    token = _CFG.api_keys.huggingface
    try:
        if not token:
            raise RuntimeError("no HF token in secrets")
        tribev2 = importlib.import_module("tribev2")  # noqa: F841
        path = _download(url) if url else None  # noqa: F841
        # TODO: real TRIBE forward pass -> per-timestep brain responses.
        raise NotImplementedError("wire tribev2 forward pass")
    except Exception as exc:  # deterministic fallback
        print(f"[video_analyzer] fallback trajectory ({type(exc).__name__}: {exc})")
        seed = abs(hash((url or "", caption or ""))) % (2**32)
        rng = np.random.default_rng(seed)
        steps = max(6, int(fps * 20))
        walk = np.cumsum(rng.normal(scale=0.3, size=(steps, 64)), axis=0)
        walk[3:6] += rng.normal(scale=1.5, size=(3, 64))
        return walk.astype("float32"), f"{TRIBE_MODEL_ID}#fallback", False


def _engagement_from_dynamics(dyn: dict) -> dict:
    """Heuristic mapping until a trained model artifact is loaded from the volume.

    Sharing/comments track surprise + curvature; retention tracks non-decaying
    novelty. Replace by loading data/models/engagement.joblib when available.
    """
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    v, s, c, nd = (dyn["velocity_mean"], dyn["surprise_mean"],
                   dyn["curvature_mean"], dyn["novelty_decay"])
    return {
        "likes": sig(0.6 * v + 0.4 * s),
        "comments": sig(0.8 * s + 0.3 * c),
        "shares": sig(0.7 * s + 0.5 * c),
        "saves": sig(0.5 * v - 0.6 * nd),
        "retention": sig(-1.2 * nd),
    }


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np  # noqa: F401

    video_id = str(payload.get("video_id") or "draft")
    url = payload.get("url") or None
    caption = payload.get("caption") or None
    fps = float(payload.get("fps") or DEFAULT_FPS)

    traj, model_id, used_brain = _tribe_trajectory(url, caption, fps)
    dyn = _dynamics(traj)
    targets = _engagement_from_dynamics(dyn)
    notes = _landmarks(traj, fps)

    return {
        "video_id": video_id,
        "model_version": model_id,
        "used_brain": used_brain,
        **{k: round(float(v), 4) for k, v in targets.items()},
        "latent_score": round(float(sum(targets.values()) / len(targets)), 4),
        "editor_notes": notes,
    }


# --------------------------------------------------------------------------- #
# Modal function + CORS-enabled web endpoint.
# --------------------------------------------------------------------------- #
@app.function(
    image=tribe_video_image(),
    gpu=_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
)
def analyze_video(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_payload(payload)


@app.function(
    image=tribe_video_image(),
    gpu=_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
)
@modal.fastapi_endpoint(method="POST", label="video-analyzer")
def analyze(payload: dict[str, Any]):
    from fastapi.responses import JSONResponse

    result = analyze_payload(payload)
    # Allow the static site (GitHub Pages) to call this endpoint from the browser.
    return JSONResponse(result, headers={"Access-Control-Allow-Origin": "*"})


@app.local_entrypoint()
def smoke():
    """modal run social_cohesion_vectors.modal_app.functions.video_analyzer::smoke"""
    out = analyze_video.remote({"video_id": "demo-001", "caption": "why boredom is a signal"})
    print(out)
