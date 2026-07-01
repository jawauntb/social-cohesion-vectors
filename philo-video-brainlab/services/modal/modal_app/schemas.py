"""Shared typed payloads passed between Modal functions and the web app."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    video_id: str
    url: str | None = None          # remote source, or...
    local_path: str | None = None   # ...a path on a mounted volume
    caption: str | None = None
    fps: float = 1.0                # frame sampling rate for the trajectory


class MultimodalFeatures(BaseModel):
    transcript_embed: list[float] = Field(default_factory=list)
    visual_embed: list[float] = Field(default_factory=list)
    audio_prosody_embed: list[float] = Field(default_factory=list)
    shot_count: int | None = None
    mean_shot_len_sec: float | None = None
    cut_rate_per_sec: float | None = None
    loudness_var: float | None = None
    extractor_version: str = "0.1.0"


class BrainTrajectory(BaseModel):
    model_id: str
    layer: int
    fps: float
    dim: int
    steps: int
    trajectory: list[float]         # steps * dim, row-major
    velocity_mean: float | None = None
    curvature_mean: float | None = None
    novelty_decay: float | None = None
    surprise_mean: float | None = None


class FeatureRecord(BaseModel):
    video_id: str
    multimodal: MultimodalFeatures
    brain: BrainTrajectory


class EngagementPrediction(BaseModel):
    video_id: str
    model_version: str
    used_brain: bool
    likes: float | None = None
    comments: float | None = None
    shares: float | None = None
    saves: float | None = None
    retention: float | None = None
    latent_score: float | None = None
    editor_notes: list[dict] = Field(default_factory=list)
