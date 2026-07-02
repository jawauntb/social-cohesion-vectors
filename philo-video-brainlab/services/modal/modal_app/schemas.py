"""Shared typed payloads passed between Modal functions and the web app."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    video_id: str
    url: Optional[str] = None          # remote source, or...
    local_path: Optional[str] = None   # ...a path on a mounted volume
    caption: Optional[str] = None
    fps: float = 1.0                # frame sampling rate for the trajectory


class MultimodalFeatures(BaseModel):
    transcript_embed: list[float] = Field(default_factory=list)
    visual_embed: list[float] = Field(default_factory=list)
    audio_prosody_embed: list[float] = Field(default_factory=list)
    shot_count: Optional[int] = None
    mean_shot_len_sec: Optional[float] = None
    cut_rate_per_sec: Optional[float] = None
    loudness_var: Optional[float] = None
    extractor_version: str = "0.1.0"


class BrainTrajectory(BaseModel):
    model_id: str
    layer: int
    fps: float
    dim: int
    steps: int
    trajectory: list[float]         # steps * dim, row-major
    velocity_mean: Optional[float] = None
    curvature_mean: Optional[float] = None
    novelty_decay: Optional[float] = None
    surprise_mean: Optional[float] = None


class FeatureRecord(BaseModel):
    video_id: str
    multimodal: MultimodalFeatures
    brain: BrainTrajectory


class EngagementPrediction(BaseModel):
    video_id: str
    model_version: str
    used_brain: bool
    likes: Optional[float] = None
    comments: Optional[float] = None
    shares: Optional[float] = None
    saves: Optional[float] = None
    retention: Optional[float] = None
    latent_score: Optional[float] = None
    editor_notes: list[dict] = Field(default_factory=list)
