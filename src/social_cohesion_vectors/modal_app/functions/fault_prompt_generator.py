"""Modal-backed open-model authorship for fault-class benchmark prompts."""

from __future__ import annotations

import importlib
import os
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.app import app, modal_secrets
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    DEFAULT_MODEL_ID,
)
from social_cohesion_vectors.modal_app.image_factory import open_llm_image


@app.function(
    image=open_llm_image(),
    secrets=modal_secrets,
    gpu=get_config().modal.default_gpu,
    timeout=1800,
)
def generate_fault_prompt_outputs(
    records: list[dict[str, Any]],
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 2,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 140,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate one open-model-authored output for each prompt record."""

    if not records:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    random.seed(seed)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_generation_model(
        transformers=transformers,
        torch=torch,
        model_id=model_id,
        device=device,
    )

    outputs: list[dict[str, str]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            _chat_prompt(
                tokenizer,
                system_prompt=str(record.get("system_prompt", "")),
                user_prompt=str(record.get("user_prompt", "")),
            )
            for record in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(encoded["input_ids"].shape[1])
        for record, sequence in zip(batch, generated, strict=True):
            text = tokenizer.decode(
                sequence[prompt_width:],
                skip_special_tokens=True,
            )
            outputs.append(
                {
                    "prompt_id": str(record.get("prompt_id", "")),
                    "text": clean_generated_text(text),
                }
            )
    return outputs


def clean_generated_text(text: str) -> str:
    """Trim common chat-template and analysis spillover from generated messages."""

    stripped = text.strip()
    stripped = re.sub(r"^(assistant|message|output)\s*:\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"^assistant\s+", "", stripped, flags=re.I)
    stripped = stripped.strip().strip('"').strip()
    return stripped


def _load_generation_model(
    *,
    transformers: Any,
    torch: Any,
    model_id: str,
    device: Any,
) -> tuple[Any, Any]:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None
    print(f"[modal] loading generation tokenizer model_id={model_id}", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(
        f"[modal] loading generation model model_id={model_id} dtype={dtype}",
        flush=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )
    model.to(device)
    model.eval()
    print(f"[modal] generation model ready model_id={model_id}", flush=True)
    return tokenizer, model


def _chat_prompt(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except (AttributeError, TypeError, ValueError):
        return f"{system_prompt}\n\n{user_prompt}\n\n"


def prompt_record_payloads(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-serializable prompt records for Modal transfer."""

    return [
        {
            "prompt_id": str(record.get("prompt_id", "")),
            "system_prompt": str(record.get("system_prompt", "")),
            "user_prompt": str(record.get("user_prompt", "")),
        }
        for record in records
    ]


__all__ = [
    "clean_generated_text",
    "generate_fault_prompt_outputs",
    "prompt_record_payloads",
]
