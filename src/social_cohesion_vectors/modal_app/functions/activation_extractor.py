"""Modal-backed activation extraction for open language models."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.app import app, modal_secrets
from social_cohesion_vectors.modal_app.image_factory import open_llm_image

DEFAULT_PROMPT_PATH = Path("data/training/activation_prompts.jsonl")
DEFAULT_MODEL_ID = get_config().model_ids.open_llm
DEFAULT_LAYER = get_config().model_ids.activation_layer


def read_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    """Read activation prompt records from JSONL or JSON."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")

    if prompt_path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError(
                "JSON prompt files must be a list or contain records=[...]."
            )

    return _coerce_prompt_records(records)


def extract_activations(
    *,
    prompt_path: str | Path | None = DEFAULT_PROMPT_PATH,
    records: Sequence[Mapping[str, Any]] | None = None,
    output_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    batch_size: int = 4,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Pool hidden states for prompt records and optionally write compressed NPZ."""

    prompt_records = (
        read_prompt_records(prompt_path)
        if records is None and prompt_path is not None
        else _coerce_prompt_records(records or [])
    )
    if not prompt_records:
        raise ValueError("At least one prompt record is required.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_tokenizer_and_model(
        transformers=transformers,
        torch=torch,
        model_id=model_id,
        device=device,
    )

    activation_batches = []
    for start in range(0, len(prompt_records), batch_size):
        batch = prompt_records[start : start + batch_size]
        texts = [record["text"] for record in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        if not -len(hidden_states) <= layer < len(hidden_states):
            raise ValueError(
                f"Layer {layer} is outside model hidden-state range "
                f"[-{len(hidden_states)}, {len(hidden_states) - 1}]."
            )
        pooled = _mean_pool(hidden_states[layer], encoded["attention_mask"])
        activation_batches.append(pooled.float().cpu().numpy())

    activations = np.concatenate(activation_batches, axis=0).astype(np.float32)
    resolved_output_path = (
        write_activation_npz(
            output_path,
            records=prompt_records,
            activations=activations,
            model_id=model_id,
            layer=layer,
        )
        if output_path is not None
        else None
    )
    return _activation_summary(
        records=prompt_records,
        activations=activations,
        model_id=model_id,
        layer=layer,
        output_path=resolved_output_path,
    )


def write_activation_npz(
    output_path: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    model_id: str,
    layer: int,
) -> Path:
    """Write pooled activations and prompt metadata in contrastive-compatible form."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activations=activations,
        sample_ids=np.array([record["sample_id"] for record in records], dtype=str),
        pair_ids=np.array([record.get("pair_id", "") for record in records], dtype=str),
        labels=np.array([record.get("label", "") for record in records], dtype=str),
        target_scores=np.array(
            [record.get("target_score", np.nan) for record in records],
            dtype=np.float32,
        ),
        texts=np.array([record["text"] for record in records], dtype=str),
        model_id=np.array(model_id),
        layer=np.array(layer, dtype=np.int64),
        pooling=np.array("attention_mean"),
    )
    return path


@app.function(
    image=open_llm_image(),
    secrets=modal_secrets,
    gpu=get_config().modal.default_gpu,
    timeout=900,
)
def smoke_extract(
    prompt_path: str | None = None,
    output_path: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    max_records: int = 2,
    batch_size: int = 2,
    max_length: int = 256,
) -> dict[str, Any]:
    """Remote smoke-test entrypoint for Modal activation extraction."""

    if prompt_path is None:
        records = _smoke_records()
    else:
        records = read_prompt_records(prompt_path)[:max_records]
    return extract_activations(
        prompt_path=None,
        records=records,
        output_path=output_path,
        model_id=model_id,
        layer=layer,
        batch_size=batch_size,
        max_length=max_length,
    )


def _load_tokenizer_and_model(
    *,
    transformers: Any,
    torch: Any,
    model_id: str,
    device: Any,
) -> tuple[Any, Any]:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    mask = attention_mask.to(hidden_states.device).unsqueeze(-1).type_as(hidden_states)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _activation_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    model_id: str,
    layer: int,
    output_path: Path | None,
) -> dict[str, Any]:
    norms = np.linalg.norm(activations, axis=1)
    return {
        "model_id": model_id,
        "layer": int(layer),
        "count": int(activations.shape[0]),
        "activation_dim": int(activations.shape[1]),
        "output_path": str(output_path) if output_path is not None else None,
        "records": [
            {
                "sample_id": str(record["sample_id"]),
                "label": str(record.get("label", "")),
                "target_score": _serializable_score(record.get("target_score")),
                "activation_norm": float(norm),
            }
            for record, norm in zip(records, norms, strict=True)
        ],
    }


def _coerce_prompt_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coerced = []
    for index, record in enumerate(records):
        text = str(record.get("text", "")).strip()
        if not text:
            raise ValueError(f"Prompt record {index} is missing non-empty text.")
        coerced.append(
            {
                "sample_id": str(record.get("sample_id") or record.get("id") or index),
                "pair_id": str(record.get("pair_id", "")),
                "label": str(record.get("label", "")),
                "target_score": _optional_float(
                    record.get("target_score", record.get("score"))
                ),
                "text": text,
            }
        )
    return coerced


def _optional_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _serializable_score(value: Any) -> float | None:
    score = _optional_float(value)
    return None if np.isnan(score) else score


def _smoke_records() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "smoke-positive",
            "pair_id": "smoke",
            "label": "positive",
            "target_score": 1.0,
            "text": "Two people repair a disagreement by naming shared goals.",
        },
        {
            "sample_id": "smoke-negative",
            "pair_id": "smoke",
            "label": "negative",
            "target_score": 0.0,
            "text": "Two people escalate a disagreement by assigning blame.",
        },
    ]


__all__ = [
    "DEFAULT_PROMPT_PATH",
    "extract_activations",
    "read_prompt_records",
    "smoke_extract",
    "write_activation_npz",
]
