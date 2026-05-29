"""Modal-backed generation with activation-direction steering."""

from __future__ import annotations

import importlib
import random
from collections.abc import Sequence
from typing import Any, Literal

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.app import app, modal_secrets
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    _load_tokenizer_and_model,
)
from social_cohesion_vectors.modal_app.image_factory import open_llm_image

SteeringPosition = Literal["last", "all"]


@app.function(
    image=open_llm_image(),
    secrets=modal_secrets,
    gpu=get_config().modal.default_gpu,
    timeout=1800,
)
def generate_with_activation_steering(
    records: list[dict[str, Any]],
    direction: list[float],
    strengths: list[float],
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    max_new_tokens: int = 128,
    max_length: int = 512,
    steering_position: SteeringPosition = "last",
    use_chat_template: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate text while adding a signed activation direction at a layer."""

    if not records:
        raise ValueError("At least one prompt record is required.")
    if not strengths:
        raise ValueError("At least one steering strength is required.")

    random.seed(seed)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_tokenizer_and_model(
        transformers=transformers,
        torch=torch,
        model_id=model_id,
        device=device,
    )
    layers = _transformer_layers(model)
    layer_index = _resolve_layer(layer, len(layers))
    block = layers[layer_index]
    direction_tensor = torch.tensor(direction, device=device, dtype=model.dtype)
    if direction_tensor.ndim != 1:
        raise ValueError("direction must be one-dimensional.")
    if int(direction_tensor.numel()) != int(model.config.hidden_size):
        raise ValueError(
            f"direction dim {int(direction_tensor.numel())} does not match "
            f"model hidden size {int(model.config.hidden_size)}."
        )
    direction_tensor = direction_tensor / direction_tensor.norm().clamp(min=1e-12)

    generations: list[dict[str, Any]] = []
    for strength in strengths:
        handle = block.register_forward_hook(
            _steering_hook(
                torch=torch,
                direction=direction_tensor,
                strength=float(strength),
                position=steering_position,
            )
        )
        try:
            for record in records:
                prompt = str(record["text"]).strip()
                encoded_text = _format_prompt(
                    tokenizer,
                    prompt,
                    use_chat_template=use_chat_template,
                )
                encoded = tokenizer(
                    encoded_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                prompt_tokens = int(encoded["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                new_tokens = generated[0, prompt_tokens:]
                generated_text = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
                generations.append(
                    {
                        "prompt_id": str(record.get("prompt_id", record.get("id", ""))),
                        "mechanism": str(record.get("mechanism", "")),
                        "prompt": prompt,
                        "strength": float(strength),
                        "model_id": model_id,
                        "layer": int(layer),
                        "steering_position": steering_position,
                        "generated_text": generated_text,
                    }
                )
        finally:
            handle.remove()
    return generations


def _steering_hook(
    *,
    torch: Any,
    direction: Any,
    strength: float,
    position: SteeringPosition,
) -> Any:
    steer = direction.view(1, 1, -1) * strength

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if position == "last":
            adjusted = hidden.clone()
            adjusted[:, -1:, :] = adjusted[:, -1:, :] + steer.to(adjusted.dtype)
        elif position == "all":
            adjusted = hidden + steer.to(hidden.dtype)
        else:
            raise ValueError(f"unknown steering position: {position}")
        if isinstance(output, tuple):
            return (adjusted, *output[1:])
        return adjusted

    return hook


def _transformer_layers(model: Any) -> Sequence[Any]:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        current = model
        for part in path:
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None:
            return current
    raise ValueError("Could not locate transformer block list on model.")


def _resolve_layer(layer: int, layer_count: int) -> int:
    resolved = layer_count + layer if layer < 0 else layer
    if not 0 <= resolved < layer_count:
        raise ValueError(f"Layer {layer} is outside model layer range {layer_count}.")
    return resolved


def _format_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


__all__ = ["SteeringPosition", "generate_with_activation_steering"]
