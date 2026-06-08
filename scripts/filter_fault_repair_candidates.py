"""Filter generated repair candidates through local verifier gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_authorship_tournament import (  # noqa: E402
    CandidateOutputSet,
)
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    API_HARD_NEGATIVE_CONTRACT_VERSION,
    API_PROMPT_CONTRACT_VERSION_CHOICES,
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    filter_prompt_records_for_repair_targets,
    prioritize_prompt_records_for_future_options,
    repair_targets_from_specs,
)
from social_cohesion_vectors.experiments.fault_repair_filter import (  # noqa: E402
    DEFAULT_REPAIR_FILTER_REQUIRED_GATES,
    REPAIR_FILTER_GATE_CHOICES,
    run_fault_repair_candidate_filter,
    save_fault_repair_candidate_filter_report,
)
from social_cohesion_vectors.modal_app.functions.activation_extractor import (  # noqa: E402
    DEFAULT_MODEL_ID,
)

PROVIDER = "modal_hf"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repair_targets = repair_targets_from_specs(args.repair_targets)
    variants = (
        tuple(
            variant
            for variant in DEFAULT_VARIANTS
            if variant.name in set(args.variants)
        )
        if args.variants
        else DEFAULT_VARIANTS
    )
    records = build_fault_prompt_records(
        variants=variants,
        prompt_contract_version=args.prompt_contract_version,
        repair_focus_options_by_contrast=repair_targets,
    )
    if args.availability_priority:
        records = prioritize_prompt_records_for_future_options(records)
    if repair_targets:
        records = filter_prompt_records_for_repair_targets(records, repair_targets)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    candidates = [
        CandidateOutputSet(
            candidate_id=candidate_id,
            output_records=read_jsonl(path),
            source_path=str(path),
        )
        for candidate_id, path in (_candidate_path(value) for value in args.candidates)
    ]
    required_gates = (
        tuple(args.required_gates)
        if args.required_gates
        else DEFAULT_REPAIR_FILTER_REQUIRED_GATES
    )
    result = run_fault_repair_candidate_filter(
        records=records,
        candidates=candidates,
        provider=args.provider,
        model=args.model_id,
        required_gates=required_gates,
        target_word_count_min=args.target_word_count_min,
        target_word_count_max=args.target_word_count_max,
    )
    write_jsonl(result.accepted_output_records, args.accepted_raw_outputs)
    save_fault_repair_candidate_filter_report(
        result.report,
        json_path=args.filter_json_report,
        markdown_path=args.filter_markdown_report,
    )
    summary = result.report["summary"]
    print(
        "fault repair candidate filter: "
        f"candidates={len(candidates)} "
        f"accepted_pairs={summary['accepted_pairs']}/"
        f"{summary['expected_pairs']} "
        f"accepted_raw_outputs={summary['accepted_raw_outputs']} "
        f"rejected_candidate_pairs={summary['rejected_candidate_pairs']}"
    )
    print(f"wrote {args.filter_markdown_report}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-raw-outputs",
        dest="candidates",
        action="append",
        required=True,
        help=(
            "Candidate raw-output JSONL. Use ID=PATH to name the candidate; "
            "otherwise the parent directory name is used for raw_outputs.jsonl."
        ),
    )
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=variant_names, default=None)
    parser.add_argument(
        "--availability-priority",
        action="store_true",
        help="Order prompt pairs to match availability-prioritized generation.",
    )
    parser.add_argument(
        "--prompt-contract-version",
        choices=API_PROMPT_CONTRACT_VERSION_CHOICES,
        default=API_HARD_NEGATIVE_CONTRACT_VERSION,
        help="Prompt contract corresponding to the candidate raw outputs.",
    )
    parser.add_argument(
        "--repair-target",
        dest="repair_targets",
        action="append",
        default=[],
        help=(
            "Filter only one repair-targeted prompt pair for BASE=option,option. "
            "May be repeated, e.g. --repair-target fair_allocation=refusal,repair."
        ),
    )
    parser.add_argument(
        "--required-gate",
        dest="required_gates",
        action="append",
        choices=REPAIR_FILTER_GATE_CHOICES,
        default=[],
        help=(
            "Required verifier gate for accepting a repair candidate pair. "
            "May be repeated. Defaults to score, slack, availability, length, "
            "and formatting gates."
        ),
    )
    parser.add_argument("--target-word-count-min", type=int, default=55)
    parser.add_argument("--target-word-count-max", type=int, default=75)
    parser.add_argument(
        "--accepted-raw-outputs",
        type=Path,
        required=True,
        help="Output JSONL containing only accepted raw prompt rows.",
    )
    parser.add_argument(
        "--filter-json-report",
        type=Path,
        required=True,
        help="Output JSON report for the repair filter.",
    )
    parser.add_argument(
        "--filter-markdown-report",
        type=Path,
        required=True,
        help="Output Markdown report for the repair filter.",
    )
    return parser.parse_args(argv)


def _candidate_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        candidate_id, path = value.split("=", 1)
        return candidate_id.strip(), Path(path)
    path = Path(value)
    candidate_id = path.parent.name if path.name == "raw_outputs.jsonl" else path.stem
    return candidate_id, path


if __name__ == "__main__":
    raise SystemExit(main())
