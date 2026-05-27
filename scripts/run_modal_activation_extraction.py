"""Send local activation prompts to Modal/GPU and save pooled hidden states."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.modal_app.app import app
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    extract_prompt_records_npz,
    read_prompt_records,
)


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=config.paths.training / "activation_prompts.jsonl",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_prompt_records(args.prompts)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No activation prompt records found.")

    output = args.output or default_output_path(args.model_id, args.layer)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Extracting {len(records)} prompt activations on Modal "
        f"model={args.model_id} layer={args.layer}"
    )
    with app.run():
        payload = extract_prompt_records_npz.remote(
            records=records,
            model_id=args.model_id,
            layer=args.layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    output.write_bytes(payload)
    print(f"Wrote {output} ({len(payload):,} bytes)")
    return 0


def default_output_path(model_id: str, layer: int) -> Path:
    config = get_config()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id).strip("_")
    return (
        config.paths.features
        / "open_llm"
        / f"activation_prompts__{slug}__layer{layer}.npz"
    )


if __name__ == "__main__":
    raise SystemExit(main())
