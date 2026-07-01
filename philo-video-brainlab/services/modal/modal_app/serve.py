"""Pre-publication scoring endpoint.

The Next.js app POSTs a draft (video ref + caption); this runs feature +
trajectory extraction, applies the trained engagement models, and returns
predicted likes/comments/shares/saves/retention plus editor notes derived from
trajectory landmarks. The engagement model artifact is loaded from the models
Volume; until one is trained, a transparent heuristic keeps the endpoint live.
"""

from __future__ import annotations

import os

import modal

from .schemas import VideoInput, EngagementPrediction
from .features import extract as extract_features
from .tribe_inference import brain_trajectory
from . import dynamics

APP_NAME = os.environ.get("BRAINLAB_MODAL_APP", "philo-video-brainlab")
app = modal.App(f"{APP_NAME}-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scikit-learn>=1.4", "pydantic>=2.6")
)

volume = modal.Volume.from_name("brainlab-models", create_if_missing=True)
MODEL_DIR = "/models"
MODEL_VERSION = os.environ.get("BRAINLAB_MODEL_VERSION", "heuristic-0.1")


def _predict_targets(brain, used_brain: bool) -> dict:
    """Load trained per-target models if present; else a dynamics heuristic.

    Heuristic rationale (until models are trained): sharing/comments track
    surprise + curvature (things worth reacting to move the brain), retention
    tracks a *non-decaying* novelty profile.
    """
    import numpy as np

    model_path = os.path.join(MODEL_DIR, "engagement.joblib")
    if used_brain and os.path.exists(model_path):
        import joblib

        bundle = joblib.load(model_path)
        traj = np.asarray(brain.trajectory, dtype=float).reshape(brain.steps, brain.dim)
        feats = np.array([[brain.velocity_mean or 0, brain.curvature_mean or 0,
                           brain.novelty_decay or 0, brain.surprise_mean or 0]])
        return {t: float(m.predict(feats)[0]) for t, m in bundle["models"].items()}

    v = brain.velocity_mean or 0.0
    s = brain.surprise_mean or 0.0
    c = brain.curvature_mean or 0.0
    nd = brain.novelty_decay or 0.0
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    return {
        "likes": float(sig(0.6 * v + 0.4 * s)),
        "comments": float(sig(0.8 * s + 0.3 * c)),
        "shares": float(sig(0.7 * s + 0.5 * c)),
        "saves": float(sig(0.5 * v - 0.6 * nd)),
        "retention": float(sig(-1.2 * nd)),
    }


@app.function(image=image, gpu=os.environ.get("MODAL_DEFAULT_GPU", "A10G"),
              volumes={MODEL_DIR: volume}, timeout=1800)
@modal.fastapi_endpoint(method="POST")
def predict(video: VideoInput) -> dict:
    mm = extract_features.remote(video)  # noqa: F841 (kept for baseline arm / logging)
    brain = brain_trajectory.remote(video)
    targets = _predict_targets(brain, used_brain=True)

    import numpy as np

    traj = np.asarray(brain.trajectory, dtype=float).reshape(brain.steps, brain.dim)
    notes = dynamics.landmarks(traj, brain.fps)

    return EngagementPrediction(
        video_id=video.video_id,
        model_version=MODEL_VERSION,
        used_brain=True,
        latent_score=float(np.mean(list(targets.values()))),
        editor_notes=notes,
        **targets,
    ).model_dump()
