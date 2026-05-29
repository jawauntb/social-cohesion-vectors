"""Run a Modal causal activation-steering smoke test and score generations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.causal_steering import (
    default_steering_prompt_records,
    score_steered_generations,
    write_steering_reports,
)
from social_cohesion_vectors.modal_app.app import app
from social_cohesion_vectors.modal_app.functions.activation_extractor import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
)
from social_cohesion_vectors.modal_app.functions.activation_steering import (
    generate_with_activation_steering,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prompts = _load_prompt_records(args.prompts, args.limit)
    direction = _load_direction(args.direction)
    args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompts, args.prompts_output)

    print(
        "generating steered responses on Modal "
        f"model={args.model_id} layer={args.layer} "
        f"prompts={len(prompts)} strengths={args.strengths}"
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
            steering_position=args.steering_position,
            use_chat_template=not args.no_chat_template,
            seed=args.seed,
        )

    scored = score_steered_generations(generations)
    write_jsonl(scored, args.generations_output)
    report = write_steering_reports(
        generations,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("steering report summary is not a dictionary")
    print(
        "causal steering smoke: "
        f"positive_vs_negative={float(summary['positive_vs_negative_success_rate']):.3f} "
        f"score_delta={float(summary['positive_minus_negative_mean_score_delta']):+.3f}"
    )
    print(f"wrote {args.markdown_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    default_direction = (
        config.paths.vectors
        / "open_llm"
        / "boundary_prior_cue_balanced_expanded__Qwen__Qwen2.5-0.5B-Instruct__layer-1_direction.npz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--direction", type=Path, default=default_direction)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--strengths", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--steering-position",
        choices=("last", "all"),
        default="last",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=config.paths.training / "causal_steering_prompts.jsonl",
    )
    parser.add_argument(
        "--generations-output",
        type=Path,
        default=config.paths.processed / "causal_steering_generations.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "causal_steering_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "causal_steering_report.md",
    )
    return parser.parse_args(argv)


def _load_prompt_records(path: Path | None, limit: int | None) -> list[dict[str, str]]:
    if path is None:
        records = default_steering_prompt_records()
    else:
        records = [
            {
                "prompt_id": str(record.get("prompt_id", record.get("id", index))),
                "mechanism": str(record.get("mechanism", "")),
                "text": str(record["text"]),
            }
            for index, record in enumerate(read_jsonl(path))
        ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SystemExit("No steering prompt records found.")
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


if __name__ == "__main__":
    raise SystemExit(main())
