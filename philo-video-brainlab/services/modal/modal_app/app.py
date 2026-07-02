"""Shared Modal app and secret configuration for philo-video-brainlab."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "philo-video-brainlab"

_MODAL_APP_DIR = Path(__file__).resolve().parent
_MODAL_SERVICE_DIR = _MODAL_APP_DIR.parent
_PHILO_ROOT = _MODAL_SERVICE_DIR.parent.parent
_REPO_ROOT = _PHILO_ROOT.parent


def dotenv_secret() -> modal.Secret:
    """Load philo `.env` first, then the repo root `.env` used locally."""

    for env_path in (_PHILO_ROOT / ".env", _REPO_ROOT / ".env"):
        if env_path.exists():
            return modal.Secret.from_dotenv(path=env_path.parent, filename=env_path.name)
    return modal.Secret.from_dict({})


modal_secrets = [dotenv_secret()]
app = modal.App(name=APP_NAME, secrets=modal_secrets)

__all__ = ["app", "dotenv_secret", "modal_secrets"]
