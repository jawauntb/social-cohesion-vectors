"""Conventional multimodal features — the baseline arm of the ablation.

Transcript, visual, audio/prosody embeddings + editing-rhythm summaries. The
brain trajectory has to beat *these* to justify itself, so this arm matters as
much as TRIBE. Heavy models load inside the Modal image; a deterministic stub
keeps the pipeline runnable before weights are wired.
"""

from __future__ import annotations

import hashlib
import os

import modal

from .schemas import VideoInput, MultimodalFeatures

APP_NAME = os.environ.get("BRAINLAB_MODAL_APP", "philo-video-brainlab")
app = modal.App(f"{APP_NAME}-features")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "sentence-transformers",
        "open-clip-torch",
        "librosa",
        "decord",
        "numpy>=1.26",
        "pydantic>=2.6",
    )
)


def _stub_vec(key: str, dim: int) -> list[float]:
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    import numpy as np

    return np.random.default_rng(seed).normal(size=dim).astype("float32").tolist()


@app.function(image=image, gpu=os.environ.get("MODAL_DEFAULT_GPU", "A10G"), timeout=1800)
def extract(video: VideoInput) -> MultimodalFeatures:
    """Extract transcript/visual/audio embeddings + editing rhythm for one video."""
    # Real path (wire once weights available):
    #   - transcript: Whisper -> sentence-transformers embedding
    #   - visual:     sample frames at FRAME_SAMPLE_FPS -> CLIP -> mean-pool
    #   - audio:      librosa prosody (pitch, energy, tempo) -> vector
    #   - rhythm:     shot boundaries -> shot count / mean length / cut rate
    vid = video.video_id
    print(f"[features] stub features for {vid} (wire Whisper/CLIP/librosa for real values)")
    return MultimodalFeatures(
        transcript_embed=_stub_vec(f"txt:{vid}", 384),
        visual_embed=_stub_vec(f"vis:{vid}", 512),
        audio_prosody_embed=_stub_vec(f"aud:{vid}", 64),
        shot_count=None,
        mean_shot_len_sec=None,
        cut_rate_per_sec=None,
        loudness_var=None,
        extractor_version="0.1.0-stub",
    )
