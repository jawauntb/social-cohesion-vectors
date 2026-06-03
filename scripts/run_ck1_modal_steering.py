"""Run a CK-1-specific Modal activation-steering evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.ck1_steering import (
    default_ck1_steering_prompt_records,
    score_ck1_generations,
    write_ck1_steering_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prompts = _load_prompt_records(args.prompts, args.limit)
    direction = _load_direction(args.direction)
    args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompts, args.prompts_output)

    print(
        "generating CK-1 steered responses on Modal "
        f"model={args.model_id} layer={args.layer} "
        f"prompts={len(prompts)} strengths={args.strengths}"
    )
    from social_cohesion_vectors.modal_app.app import app
    from social_cohesion_vectors.modal_app.functions.activation_steering import (
        generate_with_activation_steering,
    )

    with app.run():
        generations = generate_with_activation_steering.remote(
            records=prompts,
            direction=direction.tolist(),
            strengths=[float(strength) for strength in args.strengths],
            model_id=args.model_id,
            layer=args.layer,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            hook_site=args.hook_site,
            steering_position=args.steering_position,
            steering_timing=args.steering_timing,
            use_chat_template=not args.no_chat_template,
            seed=args.seed,
        )

    generations = _merge_prompt_metadata(generations, prompts)
    scored = score_ck1_generations(generations)
    write_jsonl(scored, args.generations_output)
    report = write_ck1_steering_reports(
        generations,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("CK-1 steering report summary is not a dictionary")
    print(
        "CK-1 steering: "
        f"success={float(summary['positive_vs_negative_ck1_success_rate']):.3f} "
        f"ck1_delta={float(summary['positive_minus_negative_mean_ck1_delta']):+.3f} "
        f"pseudo_delta="
        f"{float(summary['positive_minus_negative_mean_pseudo_risk_delta']):+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    default_model_id = config.model_ids.open_llm
    default_layer = config.model_ids.activation_layer
    default_direction = (
        config.paths.vectors
        / "open_llm"
        / "ck1_cue_balanced__Qwen__Qwen2.5-0.5B-Instruct__layer-1_direction.npz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--direction", type=Path, default=default_direction)
    parser.add_argument("--model-id", default=default_model_id)
    parser.add_argument("--layer", type=int, default=default_layer)
    parser.add_argument("--strengths", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--hook-site",
        choices=("pre", "post"),
        default="post",
    )
    parser.add_argument(
        "--steering-position",
        choices=("last", "all"),
        default="last",
    )
    parser.add_argument(
        "--steering-timing",
        choices=("always", "prefill", "generate"),
        default="always",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=config.paths.training / "ck1_steering_prompts.jsonl",
    )
    parser.add_argument(
        "--generations-output",
        type=Path,
        default=config.paths.processed / "ck1_steering_generations.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "ck1_steering_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "ck1_steering_report.md",
    )
    return parser.parse_args(argv)


def _load_prompt_records(path: Path | None, limit: int | None) -> list[dict[str, str]]:
    if path is None:
        records = default_ck1_steering_prompt_records()
    else:
        records = [
            {
                "prompt_id": str(record.get("prompt_id", record.get("id", index))),
                "phase": str(record.get("phase", "")),
                "mechanism": str(record.get("mechanism", "")),
                "text": str(record["text"]),
            }
            for index, record in enumerate(read_jsonl(path))
        ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SystemExit("No CK-1 steering prompt records found.")
    return records


def _load_direction(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Direction file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        direction = np.asarray(data["direction"], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise SystemExit(f"Direction file has zero norm: {path}")
    return direction / norm


def _merge_prompt_metadata(
    generations: list[dict[str, object]],
    prompts: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    metadata_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    enriched: list[dict[str, object]] = []
    for generation in generations:
        prompt_id = str(generation.get("prompt_id", ""))
        metadata = metadata_by_id.get(prompt_id, {})
        enriched.append(
            {
                **metadata,
                **generation,
                "phase": str(generation.get("phase") or metadata.get("phase", "")),
                "mechanism": str(
                    generation.get("mechanism") or metadata.get("mechanism", "")
                ),
            }
        )
    return enriched


if __name__ == "__main__":
    raise SystemExit(main())
