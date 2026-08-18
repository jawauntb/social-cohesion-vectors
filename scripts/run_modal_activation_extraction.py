"""Run Modal GPU activation extraction for local prompt records."""

from __future__ import annotations

import argparse
import base64
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    extract_activation_payload,
    load_activation_npz_bytes,
    read_prompt_records,
    write_activation_npz,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Send local activation prompt records to Modal and save NPZ."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.paths.training / "activation_prompts.jsonl",
        help="Local activation prompt JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults under data/features/open_llm.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id to run remotely.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Hidden-state layer index to pool.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Prompt records per Modal function call.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="GPU tokenizer/model batch size inside Modal.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokenized prompt length.",
    )
    return parser.parse_args(argv)


def model_slug(model_id: str) -> str:
    """Create a filesystem-safe model slug."""

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id.strip()).strip("-")
    return slug.lower() or "model"


def default_output_path(
    input_path: str | Path,
    *,
    model_id: str,
    layer: int,
    features_root: str | Path | None = None,
) -> Path:
    config = get_config()
    root = Path(features_root) if features_root is not None else config.paths.features
    return (
        root
        / "open_llm"
        / f"{Path(input_path).stem}__{model_slug(model_id)}__layer{layer}.npz"
    )


def iter_chunks(
    records: Sequence[Mapping[str, Any]],
    chunk_size: int,
) -> list[list[dict[str, Any]]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    return [
        [dict(record) for record in records[start : start + chunk_size]]
        for start in range(0, len(records), chunk_size)
    ]


def npz_bytes_from_payload(payload: Any) -> bytes:
    """Extract NPZ bytes from a Modal return payload."""

    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected bytes or mapping payload, got {type(payload)!r}.")

    for key in ("npz_bytes", "payload_bytes", "bytes"):
        value = payload.get(key)
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)

    encoded = payload.get("npz_base64")
    if isinstance(encoded, str):
        return base64.b64decode(encoded)

    raise KeyError("Payload is missing NPZ bytes under npz_bytes or npz_base64.")


def collect_remote_activations(
    records: Sequence[Mapping[str, Any]],
    *,
    remote_function: Any = extract_activation_payload,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    chunk_size: int = 64,
    batch_size: int = 4,
    max_length: int = 1024,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    """Call the Modal function in chunks and return concatenated activations."""

    chunks = iter_chunks(records, chunk_size)
    if not chunks:
        raise ValueError("At least one prompt record is required.")

    activation_chunks: list[np.ndarray] = []
    summaries: list[Mapping[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Running Modal activation chunk {index}/{len(chunks)} ({len(chunk)} records)"
        )
        payload = remote_function.remote(
            records=chunk,
            model_id=model_id,
            layer=layer,
            batch_size=batch_size,
            max_length=max_length,
        )
        arrays = load_activation_npz_bytes(npz_bytes_from_payload(payload))
        activations = np.asarray(arrays["activations"], dtype=np.float32)
        if activations.shape[0] != len(chunk):
            raise ValueError(
                f"Remote chunk returned {activations.shape[0]} rows for "
                f"{len(chunk)} records."
            )
        activation_chunks.append(activations)
        if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
            summaries.append(payload["summary"])

    return np.concatenate(activation_chunks, axis=0), summaries


def run_modal_activation_extraction(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    chunk_size: int = 64,
    batch_size: int = 4,
    max_length: int = 1024,
    remote_function: Any = extract_activation_payload,
) -> dict[str, Any]:
    records = read_prompt_records(input_path)
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else default_output_path(input_path, model_id=model_id, layer=layer)
    )
    activations, chunk_summaries = collect_remote_activations(
        records,
        remote_function=remote_function,
        model_id=model_id,
        layer=layer,
        chunk_size=chunk_size,
        batch_size=batch_size,
        max_length=max_length,
    )
    write_activation_npz(
        resolved_output_path,
        records=records,
        activations=activations,
        model_id=model_id,
        layer=layer,
    )
    return {
        "input_path": str(input_path),
        "output_path": str(resolved_output_path),
        "model_id": model_id,
        "layer": int(layer),
        "record_count": int(len(records)),
        "activation_dim": int(activations.shape[1]),
        "chunk_count": int(
            len(chunk_summaries) or len(iter_chunks(records, chunk_size))
        ),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_modal_activation_extraction(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model_id,
        layer=args.layer,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
