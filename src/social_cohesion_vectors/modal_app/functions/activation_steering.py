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

SteeringHookSite = Literal["pre", "post"]
SteeringPosition = Literal["last", "all"]
SteeringTiming = Literal["always", "prefill", "generate"]


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
    hook_site: SteeringHookSite = "post",
    steering_position: SteeringPosition = "last",
    steering_timing: SteeringTiming = "always",
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
        state = {"forward_calls": 0}
        if hook_site == "pre":
            handle = block.register_forward_pre_hook(
                _pre_steering_hook(
                    direction=direction_tensor,
                    strength=float(strength),
                    position=steering_position,
                    timing=steering_timing,
                    state=state,
                )
            )
        elif hook_site == "post":
            handle = block.register_forward_hook(
                _post_steering_hook(
                    direction=direction_tensor,
                    strength=float(strength),
                    position=steering_position,
                    timing=steering_timing,
                    state=state,
                )
            )
        else:
            raise ValueError(f"unknown hook site: {hook_site}")
        try:
            for record in records:
                state["forward_calls"] = 0
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
                        "hook_site": hook_site,
                        "steering_position": steering_position,
                        "steering_timing": steering_timing,
                        "generated_text": generated_text,
                    }
                )
        finally:
            handle.remove()
    return generations


@app.function(
    image=open_llm_image(),
    secrets=modal_secrets,
    gpu=get_config().modal.default_gpu,
    timeout=1800,
)
def trace_activation_steering(
    records: list[dict[str, Any]],
    direction: list[float],
    strengths: list[float],
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    max_new_tokens: int = 24,
    max_length: int = 512,
    top_k: int = 5,
    hook_site: SteeringHookSite = "post",
    steering_position: SteeringPosition = "last",
    steering_timing: SteeringTiming = "generate",
    use_chat_template: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Greedy-generate while recording projection telemetry at the steering hook."""

    if not records:
        raise ValueError("At least one prompt record is required.")
    if not strengths:
        raise ValueError("At least one steering strength is required.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

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
    direction_tensor = _direction_tensor(
        torch=torch,
        direction=direction,
        device=device,
        dtype=model.dtype,
        hidden_size=int(model.config.hidden_size),
    )

    traces: list[dict[str, Any]] = []
    for strength in strengths:
        telemetry_events: list[dict[str, Any]] = []
        state: dict[str, Any] = {"forward_calls": 0, "generation_step": -1}
        if hook_site == "pre":
            handle = block.register_forward_pre_hook(
                _pre_steering_hook(
                    direction=direction_tensor,
                    strength=float(strength),
                    position=steering_position,
                    timing=steering_timing,
                    state=state,
                    telemetry=telemetry_events,
                )
            )
        elif hook_site == "post":
            handle = block.register_forward_hook(
                _post_steering_hook(
                    direction=direction_tensor,
                    strength=float(strength),
                    position=steering_position,
                    timing=steering_timing,
                    state=state,
                    telemetry=telemetry_events,
                )
            )
        else:
            raise ValueError(f"unknown hook site: {hook_site}")
        try:
            for record in records:
                prompt_events_start = len(telemetry_events)
                state["forward_calls"] = 0
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
                generated_ids, step_records = _greedy_generate_with_trace(
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    encoded=encoded,
                    state=state,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                )
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
                traces.append(
                    {
                        "prompt_id": str(record.get("prompt_id", record.get("id", ""))),
                        "mechanism": str(record.get("mechanism", "")),
                        "prompt": prompt,
                        "strength": float(strength),
                        "model_id": model_id,
                        "layer": int(layer),
                        "hook_site": hook_site,
                        "steering_position": steering_position,
                        "steering_timing": steering_timing,
                        "max_new_tokens": int(max_new_tokens),
                        "generated_text": generated_text,
                        "steps": step_records,
                        "events": telemetry_events[prompt_events_start:],
                    }
                )
        finally:
            handle.remove()
    return traces


def _pre_steering_hook(
    *,
    direction: Any,
    strength: float,
    position: SteeringPosition,
    timing: SteeringTiming,
    state: dict[str, Any],
    telemetry: list[dict[str, Any]] | None = None,
) -> Any:
    steer = direction.view(1, 1, -1) * strength

    def hook(_module: Any, inputs: Any) -> Any:
        if not inputs:
            return inputs
        hidden = inputs[0]
        state["forward_calls"] += 1
        if not _should_steer(hidden, timing=timing, state=state):
            return inputs
        adjusted = _apply_steer(hidden, steer=steer, position=position)
        _record_telemetry(
            telemetry,
            direction=direction,
            hidden=hidden,
            adjusted=adjusted,
            state=state,
            strength=strength,
            position=position,
        )
        return (adjusted, *inputs[1:])

    return hook


def _post_steering_hook(
    *,
    direction: Any,
    strength: float,
    position: SteeringPosition,
    timing: SteeringTiming,
    state: dict[str, Any],
    telemetry: list[dict[str, Any]] | None = None,
) -> Any:
    steer = direction.view(1, 1, -1) * strength

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        state["forward_calls"] += 1
        if not _should_steer(hidden, timing=timing, state=state):
            return output
        adjusted = _apply_steer(hidden, steer=steer, position=position)
        _record_telemetry(
            telemetry,
            direction=direction,
            hidden=hidden,
            adjusted=adjusted,
            state=state,
            strength=strength,
            position=position,
        )
        if isinstance(output, tuple):
            return (adjusted, *output[1:])
        return adjusted

    return hook


def _should_steer(
    hidden: Any,
    *,
    timing: SteeringTiming,
    state: dict[str, Any],
) -> bool:
    if timing == "always":
        return True
    if timing == "prefill":
        return state["forward_calls"] == 1
    if timing == "generate":
        return state["forward_calls"] > 1 and int(hidden.shape[1]) == 1
    raise ValueError(f"unknown steering timing: {timing}")


def _apply_steer(
    hidden: Any,
    *,
    steer: Any,
    position: SteeringPosition,
) -> Any:
    if position == "last":
        adjusted = hidden.clone()
        adjusted[:, -1:, :] = adjusted[:, -1:, :] + steer.to(adjusted.dtype)
        return adjusted
    if position == "all":
        return hidden + steer.to(hidden.dtype)
    raise ValueError(f"unknown steering position: {position}")


def _record_telemetry(
    telemetry: list[dict[str, Any]] | None,
    *,
    direction: Any,
    hidden: Any,
    adjusted: Any,
    state: dict[str, Any],
    strength: float,
    position: SteeringPosition,
) -> None:
    if telemetry is None:
        return
    before = _target_hidden(hidden, position=position).float()
    after = _target_hidden(adjusted, position=position).float()
    direction_float = direction.float()
    before_projection = before @ direction_float
    after_projection = after @ direction_float
    telemetry.append(
        {
            "forward_call": int(state["forward_calls"]),
            "generation_step": int(state.get("generation_step", -1)),
            "seq_len": int(hidden.shape[1]),
            "position": position,
            "strength": float(strength),
            "tokens_steered": int(before.shape[1]),
            "mean_projection_before": float(before_projection.mean().item()),
            "mean_projection_after": float(after_projection.mean().item()),
            "mean_projection_delta": float(
                (after_projection - before_projection).mean().item()
            ),
            "mean_norm_before": float(before.norm(dim=-1).mean().item()),
            "mean_norm_after": float(after.norm(dim=-1).mean().item()),
        }
    )


def _target_hidden(hidden: Any, *, position: SteeringPosition) -> Any:
    if position == "last":
        return hidden[:, -1:, :]
    if position == "all":
        return hidden
    raise ValueError(f"unknown steering position: {position}")


def _direction_tensor(
    *,
    torch: Any,
    direction: list[float],
    device: Any,
    dtype: Any,
    hidden_size: int,
) -> Any:
    direction_tensor = torch.tensor(direction, device=device, dtype=dtype)
    if direction_tensor.ndim != 1:
        raise ValueError("direction must be one-dimensional.")
    if int(direction_tensor.numel()) != hidden_size:
        raise ValueError(
            f"direction dim {int(direction_tensor.numel())} does not match "
            f"model hidden size {hidden_size}."
        )
    return direction_tensor / direction_tensor.norm().clamp(min=1e-12)


def _greedy_generate_with_trace(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    encoded: dict[str, Any],
    state: dict[str, Any],
    max_new_tokens: int,
    top_k: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    generated_ids: list[int] = []
    step_records: list[dict[str, Any]] = []
    attention_mask = encoded["attention_mask"]
    past_key_values = None
    next_input_ids = encoded["input_ids"]
    for step in range(max_new_tokens):
        state["generation_step"] = step
        with torch.inference_mode():
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        step_records.append(
            _step_record(
                torch=torch,
                tokenizer=tokenizer,
                logits=logits,
                token_id=token_id,
                step=step,
                top_k=top_k,
            )
        )
        if token_id == tokenizer.eos_token_id:
            break
        next_input_ids = next_token.view(1, 1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
    return generated_ids, step_records


def _step_record(
    *,
    torch: Any,
    tokenizer: Any,
    logits: Any,
    token_id: int,
    step: int,
    top_k: int,
) -> dict[str, Any]:
    k = min(top_k, int(logits.shape[-1]))
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=-1)
    return {
        "step": int(step),
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id]),
        "top_tokens": [
            {
                "token_id": int(index),
                "token_text": tokenizer.decode([int(index)]),
                "probability": float(probability),
            }
            for index, probability in zip(
                top_indices[0].tolist(),
                top_probabilities[0].tolist(),
                strict=True,
            )
        ],
    }


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


__all__ = [
    "SteeringHookSite",
    "SteeringPosition",
    "SteeringTiming",
    "generate_with_activation_steering",
    "trace_activation_steering",
]
