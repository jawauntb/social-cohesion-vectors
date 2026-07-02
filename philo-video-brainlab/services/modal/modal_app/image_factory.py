"""Modal image builders for the philo TRIBE video analyzer."""

from __future__ import annotations

import os

import modal

_TRIBE_GIT_REF = os.environ.get("TRIBE_GIT_REF", "af58661791a351a448a489042a28f6c37e1c14b7")
_TRIBE_EXCA_VERSION = "0.5.25"
_TRIBE_IMPORT_PREFLIGHT = (
    'python -c "import exca.steps.base as exca_base; exca_base.NoValue(); '
    'from tribev2 import TribeModel; print(TribeModel.__name__)"'
)


def tribe_video_image() -> modal.Image:
    """GPU image for TRIBE brain-response extraction from video or text."""

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
        .pip_install(f"git+https://github.com/facebookresearch/tribev2.git@{_TRIBE_GIT_REF}")
        .run_commands(f"python -m pip install --no-deps exca=={_TRIBE_EXCA_VERSION}")
        .run_commands(_TRIBE_IMPORT_PREFLIGHT)
    )


__all__ = ["tribe_video_image"]
