"""TRIBE v2 brain-response trajectory extraction on Modal GPU.

TRIBE v2 (Meta) predicts fMRI-like brain responses to video/audio/text. It is
gated on Hugging Face: request access while logged in, then set HF_TOKEN as a
Modal secret. Until weights are wired, `_load_tribe` raises with instructions
and `brain_trajectory` falls back to a deterministic stand-in so the rest of the
pipeline (and the ablation harness) is runnable end-to-end.
"""

from __future__ import annotations

import hashlib
import os

import modal
import numpy as np

from . import dynamics
from .schemas import BrainTrajectory, VideoInput

APP_NAME = os.environ.get("BRAINLAB_MODAL_APP", "philo-video-brainlab")
app = modal.App(f"{APP_NAME}-tribe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "numpy>=1.26",
        "decord",
        "librosa",
        "pydantic>=2.6",
    )
)

# TRIBE weights are large + gated; cache them on a Volume across runs.
volume = modal.Volume.from_name("brainlab-models", create_if_missing=True)
MODEL_DIR = "/models"

TRIBE_MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "facebook/tribe-v2")
TARGET_LAYER = int(os.environ.get("TRIBE_TARGET_LAYER", "-1"))


def _load_tribe(model_id: str):
    """Load the gated TRIBE v2 encoder. Requires HF_TOKEN with granted access."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Request TRIBE v2 access on Hugging Face while logged in, "
            "then add it as a Modal secret: `modal secret create hf-token HF_TOKEN=...`."
        )
    # NOTE: wire the real loader once access is granted, e.g.:
    #   from transformers import AutoModel, AutoProcessor
    #   proc = AutoProcessor.from_pretrained(model_id, token=token)
    #   model = AutoModel.from_pretrained(model_id, token=token).to("cuda").eval()
    #   return model, proc
    raise NotImplementedError(
        f"TRIBE loader for {model_id} not wired yet — grant HF access and implement _load_tribe."
    )


def _stub_trajectory(video: VideoInput, dim: int = 64) -> np.ndarray:
    """Deterministic placeholder trajectory so the pipeline runs before weights.

    Seeded by video id, so features are stable across runs but clearly synthetic.
    Replace with the real TRIBE forward pass — do NOT ship results off the stub.
    """
    import numpy as np

    seed = int(hashlib.sha256(video.video_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    steps = max(4, int((video.fps or 1.0) * 20))  # ~20s of a 1fps trajectory
    # smooth random walk with an early "surprise" bump, to exercise dynamics code
    walk = np.cumsum(rng.normal(scale=0.3, size=(steps, dim)), axis=0)
    walk[3:6] += rng.normal(scale=1.5, size=(3, dim))
    return walk.astype("float32")


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_DEFAULT_GPU", "A10G"),
    volumes={MODEL_DIR: volume},
    timeout=1800,
)
def brain_trajectory(video: VideoInput) -> BrainTrajectory:
    """Estimate the fMRI-like brain-response trajectory for one video."""

    try:
        model, proc = _load_tribe(TRIBE_MODEL_ID)
        # TODO: sample frames/audio/text at `video.fps`, run the encoder, and
        # collect the per-timestep hidden states at TARGET_LAYER into `traj`.
        raise NotImplementedError  # placeholder until the forward pass is wired
    except (RuntimeError, NotImplementedError) as exc:
        print(f"[tribe] using stub trajectory: {exc}")
        traj = _stub_trajectory(video)
        model_id = f"{TRIBE_MODEL_ID}#stub"

    summ = dynamics.summarize(traj)
    return BrainTrajectory(
        model_id=model_id,
        layer=TARGET_LAYER,
        fps=video.fps or 1.0,
        dim=int(traj.shape[1]),
        steps=int(traj.shape[0]),
        trajectory=traj.reshape(-1).tolist(),
        **summ,
    )


@app.local_entrypoint()
def smoke():
    """`modal run services/modal/modal_app/tribe_inference.py::smoke`"""
    out = brain_trajectory.remote(VideoInput(video_id="demo-001", fps=1.0))
    print(
        f"trajectory: {out.steps} steps x {out.dim} dim | "
        f"velocity_mean={out.velocity_mean:.3f} surprise_mean={out.surprise_mean:.3f}"
    )
