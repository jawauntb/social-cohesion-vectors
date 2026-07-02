"""Modal image builders for open-model activation and video-analysis jobs."""

from __future__ import annotations

import os

import modal


def open_llm_image() -> modal.Image:
    """Build a compact GPU-capable image for Hugging Face activation extraction."""

    return modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy>=1.26,<3.0",
        "pydantic[email]==2.12.5",
        "python-dotenv==1.1.0",
        "torch>=2.5,<2.8",
        "transformers>=4.46,<5.0",
        "accelerate>=1.0,<2.0",
        "safetensors>=0.4,<1.0",
    )


# Pin the TRIBE install per docs/runbooks/tribe-modal-startup-fix.md so the
# predictor imports cleanly. Pin ``TRIBE_GIT_REF`` to a validated commit for
# reproducible rebuilds.
_TRIBE_GIT_REF = os.environ.get("TRIBE_GIT_REF", "main")
_TRIBE_EXCA_VERSION = "0.5.25"
_TRIBE_IMPORT_PREFLIGHT = (
    'python -c "import exca.steps.base as exca_base; exca_base.NoValue(); '
    'from tribev2 import TribeModel; print(TribeModel.__name__)"'
)


def tribe_video_image() -> modal.Image:
    """GPU image for TRIBE brain-response extraction from video / audio / text.

    Follows the TRIBE Modal startup runbook: install TRIBE and ``exca==0.5.25``
    in the same layer, then preflight the import at build time so a dependency
    mismatch fails the image build instead of a production container.
    """

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "git")
        .pip_install(
            "numpy>=1.26,<3.0",
            "pydantic[email]==2.12.5",
            "python-dotenv==1.1.0",
            "torch>=2.5,<2.8",
            "transformers>=4.46,<5.0",
            "accelerate>=1.0,<2.0",
            "safetensors>=0.4,<1.0",
            "huggingface_hub>=0.24",
            "decord>=0.6",
            "librosa>=0.10",
            "soundfile>=0.12",
            "yt-dlp>=2024.8",
            "fastapi[standard]",
        )
        .pip_install(
            f"git+https://github.com/facebookresearch/tribev2.git@{_TRIBE_GIT_REF}",
            f"exca=={_TRIBE_EXCA_VERSION}",
        )
        .run_commands(_TRIBE_IMPORT_PREFLIGHT)
    )


__all__ = ["open_llm_image", "tribe_video_image"]
