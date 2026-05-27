"""Environment-backed config for the social cohesion vector pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default)
    return (_REPO_ROOT / raw).resolve() if raw.startswith("./") else Path(raw).resolve()


class Paths(BaseModel):
    repo_root: Path = _REPO_ROOT
    data_root: Path = _env_path("DATA_ROOT", "./data")
    raw: Path = _env_path("RAW_DIR", "./data/raw")
    processed: Path = _env_path("PROCESSED_DIR", "./data/processed")
    features: Path = _env_path("FEATURES_DIR", "./data/features")
    labels: Path = _env_path("LABELS_DIR", "./data/labels")
    training: Path = _env_path("TRAINING_DIR", "./data/training")
    models: Path = _env_path("MODELS_DIR", "./data/models")
    vectors: Path = _env_path("VECTORS_DIR", "./data/models/vectors")
    reports: Path = _env_path("REPORTS_DIR", "./data/reports")
    scenarios: Path = _env_path("SCENARIOS_DIR", "./data/scenarios")

    def ensure(self) -> None:
        for path in [
            self.data_root,
            self.raw,
            self.processed,
            self.features,
            self.labels,
            self.training,
            self.models,
            self.vectors,
            self.reports,
            self.scenarios,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class ApiKeys(BaseModel):
    openai: str = _env("OPENAI_API_KEY")
    anthropic: str = _env("ANTHROPIC_API_KEY")
    google: str = _env("GOOGLE_API_KEY")
    huggingface: str = _env("HUGGINGFACE_TOKEN") or _env("HF_TOKEN")
    wandb: str = _env("WANDB_API_KEY")


class ModelIds(BaseModel):
    open_llm: str = _env("SCV_OPEN_LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    activation_layer: int = _env_int("SCV_OPEN_LLM_ACTIVATION_LAYER", -1)
    tribe: str = _env("TRIBE_MODEL_ID", "facebook/tribev2")
    vjepa: str = _env("VJEPA_MODEL_ID", "facebook/vjepa2-vitl-fpc64-256")


class Pipeline(BaseModel):
    random_seed: int = _env_int("RANDOM_SEED", 42)
    dev_scenario_limit: int = _env_int("DEV_SCENARIO_LIMIT", 24)


class ModalConfig(BaseModel):
    app_base_name: str = _env("SCV_MODAL_APP_BASE_NAME", "social-cohesion-vectors")
    default_gpu: str = _env("MODAL_DEFAULT_GPU", "A10G")
    workspace: str = _env("MODAL_WORKSPACE")


class Config(BaseModel):
    project_name: str = Field(
        default_factory=lambda: _env("PROJECT_NAME", "social_cohesion_vectors")
    )
    env: str = Field(default_factory=lambda: _env("ENV", "dev"))
    log_level: str = Field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    paths: Paths = Field(default_factory=Paths)
    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    model_ids: ModelIds = Field(default_factory=ModelIds)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    pipeline: Pipeline = Field(default_factory=Pipeline)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
