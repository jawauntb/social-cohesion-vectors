"""Shared Modal app and secret configuration."""

from __future__ import annotations

from pathlib import Path

import modal

from social_cohesion_vectors.config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_PATH = _REPO_ROOT / ".env"


def dotenv_secret(env_path: Path = _DOTENV_PATH) -> modal.Secret:
    """Create a Modal secret from the local ``.env`` when it exists."""

    if env_path.exists():
        return modal.Secret.from_dotenv(path=env_path.parent, filename=env_path.name)
    return modal.Secret.from_dict({})


_CONFIG = get_config()
modal_secrets = [dotenv_secret()]
app = modal.App(name=_CONFIG.modal.app_base_name, secrets=modal_secrets)

__all__ = ["app", "dotenv_secret", "modal_secrets"]
