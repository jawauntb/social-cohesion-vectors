"""Generate LLM-style social-dilemma trajectories from seed scenarios."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.generation import (
    TRAJECTORY_STYLES,
    TrajectoryPromptRecord,
    TrajectoryStyle,
    build_prompt_records,
    generate_offline_runs,
    normalize_styles,
    run_from_generated_text,
)
from social_cohesion_vectors.scenario_library import (
    filter_scenarios,
    load_seed_scenarios,
)
from social_cohesion_vectors.schemas import Scenario, SimulationRun

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = filter_scenarios(load_seed_scenarios(), limit=args.limit)
    if args.provider == "offline":
        runs = generate_offline_runs(scenarios, styles=args.styles, seed=args.seed)
    else:
        runs = _generate_anthropic_runs(scenarios, styles=args.styles, seed=args.seed)

    _write_jsonl(runs, args.output)
    print(f"Wrote {len(runs)} generated trajectories to {args.output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Generate harder LLM-authored social-dilemma trajectories."
    )
    parser.add_argument(
        "--provider",
        choices=["offline", "anthropic"],
        default="offline",
        help="Trajectory provider. Anthropic requires ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of seed scenarios to use from file order.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=list(TRAJECTORY_STYLES),
        help=(
            "Trajectory styles to generate. Values may be space-separated or "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.processed / "generated_trajectories.jsonl",
        help="Destination JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=config.pipeline.random_seed)
    args = parser.parse_args(argv)
    try:
        args.styles = normalize_styles(args.styles)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _generate_anthropic_runs(
    scenarios: Sequence[Scenario],
    *,
    styles: Iterable[TrajectoryStyle],
    seed: int,
) -> list[SimulationRun]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for --provider anthropic")

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    scenario_index = {scenario.id: scenario for scenario in scenarios}
    prompts = build_prompt_records(scenarios, styles=styles, seed=seed)
    runs: list[SimulationRun] = []
    for prompt in prompts:
        generated_text = _anthropic_message(prompt, api_key=api_key, model=model)
        runs.append(
            run_from_generated_text(
                scenario_index[prompt.scenario_id],
                prompt.style,
                seed=prompt.seed,
                generated_text=generated_text,
                provider="anthropic",
                model=model,
            )
        )
    return runs


def _anthropic_message(
    prompt: TrajectoryPromptRecord,
    *,
    api_key: str,
    model: str,
) -> str:
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": prompt.system_prompt,
        "messages": [{"role": "user", "content": prompt.user_prompt}],
    }
    request = Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc.reason}") from exc

    text_parts = [
        str(block.get("text", ""))
        for block in body.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def _write_jsonl(runs: Iterable[SimulationRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{json.dumps(run.model_dump(mode='json'), sort_keys=True)}\n"
            for run in runs
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
