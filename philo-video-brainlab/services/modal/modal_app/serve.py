"""Browser-callable philo-video-brainlab Modal endpoint.

The Railway app posts draft video or caption payloads here through
`apps/web/app/api/predict/route.ts`. Real TRIBE predictions set
`used_brain: true`; deterministic fallback trajectories keep the UI testable
but always set `used_brain: false`.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import modal

from modal_app.app import app, modal_secrets
from modal_app.dynamics import landmarks, summarize
from modal_app.image_factory import tribe_video_image
from modal_app.schemas import EngagementPrediction, VideoInput

TRIBE_MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "facebook/tribev2")
TRIBE_GPU = os.environ.get("TRIBE_GPU", "A100")
TRIBE_TR_SEC = float(os.environ.get("TRIBE_TR_SEC", "1.49"))
MODEL_VERSION = os.environ.get("BRAINLAB_MODEL_VERSION", "heuristic-0.1")

_weights = modal.Volume.from_name("philo-brainlab-tribe-weights", create_if_missing=True)
_WEIGHTS_DIR = "/weights"


def _ensure_hf_token() -> None:
    if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGINGFACE_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]


def _download(url: str) -> str:
    dst = tempfile.mktemp(suffix=".mp4")
    if any(host in url for host in ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")):
        ytdlp = importlib.import_module("yt_dlp")
        with ytdlp.YoutubeDL({"outtmpl": dst, "format": "mp4/best", "quiet": True}) as ydl:
            ydl.download([url])
    else:
        urllib.request.urlretrieve(url, dst)
    return dst


def _load_model():
    """Load TRIBE v2, caching weights on a Modal volume. None means fallback."""

    try:
        _ensure_hf_token()
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained(TRIBE_MODEL_ID, cache_folder=_WEIGHTS_DIR)
        _weights.commit()
        return model
    except Exception as exc:  # noqa: BLE001 - dependency/HF failures should not kill UI.
        print(f"[philo-video-brainlab] TRIBE load failed ({type(exc).__name__}: {exc}); using fallback")
        return None


def _fallback_trajectory(video: VideoInput):
    import numpy as np

    key = f"{video.video_id}|{video.url or ''}|{video.caption or ''}"
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    steps = max(6, int((video.fps or 1.0) * 20))
    walk = np.cumsum(rng.normal(scale=0.3, size=(steps, 64)), axis=0)
    walk[3:6] += rng.normal(scale=1.5, size=(3, 64))
    return walk.astype("float32")


def _trajectory(model, video: VideoInput):
    """Return (trajectory, model_version, used_brain, seconds_per_step)."""

    import numpy as np

    try:
        if model is None:
            raise RuntimeError("TRIBE model is not loaded")

        if video.url:
            path = _download(video.url)
            df = model.get_events_dataframe(video_path=path)
        elif video.local_path:
            path = Path(video.local_path)
            if not path.exists():
                raise FileNotFoundError(video.local_path)
            df = model.get_events_dataframe(video_path=str(path))
        elif video.caption:
            text_path = tempfile.mktemp(suffix=".txt")
            with open(text_path, "w", encoding="utf-8") as fh:
                fh.write(video.caption)
            df = model.get_events_dataframe(text_path=text_path)
        else:
            raise ValueError("provide a video url, local_path, or caption")

        preds, _segments = model.predict(events=df)
        traj = np.asarray(preds, dtype=float)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
        return traj, TRIBE_MODEL_ID, True, TRIBE_TR_SEC
    except Exception as exc:  # noqa: BLE001 - media/model failures use transparent fallback.
        print(f"[philo-video-brainlab] fallback trajectory ({type(exc).__name__}: {exc})")
        fps = video.fps or 1.0
        return _fallback_trajectory(video), f"{TRIBE_MODEL_ID}#fallback", False, 1.0 / max(fps, 1e-6)


def _engagement_from_dynamics(dyn: dict[str, float]) -> dict[str, float]:
    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

    v = dyn["velocity_mean"]
    s = dyn["surprise_mean"]
    c = dyn["curvature_mean"]
    nd = dyn["novelty_decay"]

    def z(x: float) -> float:
        return x / 50.0

    return {
        "likes": sig(0.6 * z(v) + 0.4 * z(s)),
        "comments": sig(0.8 * z(s) + 0.3 * c),
        "shares": sig(0.7 * z(s) + 0.5 * c),
        "saves": sig(0.5 * z(v) - 0.6 * nd),
        "retention": sig(-1.2 * nd),
    }


def analyze_payload(model, payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    video = VideoInput.model_validate(payload)
    traj, model_id, used_brain, sec_per_step = _trajectory(model, video)
    dyn = summarize(traj)
    targets = _engagement_from_dynamics(dyn)
    notes = landmarks(traj, 1.0 / max(sec_per_step, 1e-6))
    model_version = model_id if used_brain else f"{MODEL_VERSION}:{model_id}"

    return EngagementPrediction(
        video_id=video.video_id,
        model_version=model_version,
        used_brain=used_brain,
        likes=round(float(targets["likes"]), 4),
        comments=round(float(targets["comments"]), 4),
        shares=round(float(targets["shares"]), 4),
        saves=round(float(targets["saves"]), 4),
        retention=round(float(targets["retention"]), 4),
        latent_score=round(float(np.mean(list(targets.values()))), 4),
        editor_notes=notes,
    ).model_dump()


@app.function(
    image=tribe_video_image(),
    gpu=TRIBE_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
)
def analyze_video(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_payload(_load_model(), payload)


@app.function(
    image=tribe_video_image(),
    gpu=TRIBE_GPU,
    secrets=modal_secrets,
    volumes={_WEIGHTS_DIR: _weights},
    timeout=1800,
    min_containers=0,
)
@modal.asgi_app(label="philo-video-analyzer")
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    model = _load_model()
    api = FastAPI(title="philo-video-brainlab")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/")
    def analyze(payload: dict[str, Any]):
        return analyze_payload(model, payload)

    @api.get("/health")
    def health():
        return {"ok": True, "model": TRIBE_MODEL_ID, "loaded": model is not None}

    return api


@app.local_entrypoint()
def smoke():
    """Run with `modal run -m modal_app.serve::smoke` from services/modal."""

    out = analyze_video.remote({"video_id": "demo-001", "caption": "why boredom is a signal"})
    print(out)
