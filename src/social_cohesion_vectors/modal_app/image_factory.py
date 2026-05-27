"""Modal image builders for open-model activation jobs."""

from __future__ import annotations

import modal


def open_llm_image() -> modal.Image:
    """Build a compact GPU-capable image for Hugging Face activation extraction."""

    return modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy>=1.26,<3.0",
        "torch>=2.5,<2.8",
        "transformers>=4.46,<5.0",
        "accelerate>=1.0,<2.0",
        "safetensors>=0.4,<1.0",
    )


__all__ = ["open_llm_image"]
