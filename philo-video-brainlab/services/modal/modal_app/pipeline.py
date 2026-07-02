"""End-to-end: video -> multimodal features + TRIBE brain trajectory -> record.

Fans a batch of videos across Modal containers, returning one FeatureRecord each.
Persist the records into Postgres (packages/db) via the web app or a small ETL.
"""

from __future__ import annotations

import os

import modal

from .features import app as features_app
from .features import extract as extract_features
from .features import image as features_image
from .schemas import FeatureRecord, VideoInput
from .tribe_inference import app as tribe_app
from .tribe_inference import brain_trajectory

APP_NAME = os.environ.get("BRAINLAB_MODAL_APP", "philo-video-brainlab")
app = modal.App(f"{APP_NAME}-pipeline")
app.include(features_app)
app.include(tribe_app)


@app.function(image=features_image, timeout=3600)
def process_one(video: VideoInput) -> FeatureRecord:
    mm = extract_features.remote(video)
    brain = brain_trajectory.remote(video)
    return FeatureRecord(video_id=video.video_id, multimodal=mm, brain=brain)


@app.function(image=features_image, timeout=7200)
def process_batch(videos: list[VideoInput]) -> list[FeatureRecord]:
    return list(process_one.map(videos))


@app.local_entrypoint()
def smoke():
    """`modal run services/modal/modal_app/pipeline.py::smoke`"""
    demo = [VideoInput(video_id=f"demo-{i:03d}", fps=1.0) for i in range(3)]
    for rec in process_batch.remote(demo):
        b = rec.brain
        print(
            f"{rec.video_id}: brain {b.steps}x{b.dim} "
            f"(v={b.velocity_mean:.2f}) | mm dims "
            f"{len(rec.multimodal.transcript_embed)}/{len(rec.multimodal.visual_embed)}"
        )
