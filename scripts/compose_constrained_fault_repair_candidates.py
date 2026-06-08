"""Compose constrained repair candidates for residual availability failures."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_constrained_repair import (  # noqa: E402
    CONSTRAINED_REPAIR_COMPOSER_VERSION,
    CONSTRAINED_REPAIR_COMPOSER_VERSION_CHOICES,
    CONSTRAINED_REPAIR_PROVIDER,
    compose_constrained_repair_output_records,
    save_constrained_repair_composition_report,
)
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    API_PROMPT_CONTRACT_VERSION_CHOICES,
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    filter_prompt_records_for_repair_targets,
    prioritize_prompt_records_for_future_options,
    repair_targets_from_specs,
)


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

    result = compose_constrained_repair_output_records(
        records,
        provider=args.provider,
        model=args.model_id or args.composer_version,
        composer_version=args.composer_version,
    )
    write_jsonl(result.output_records, args.raw_outputs)
    save_constrained_repair_composition_report(
        result.report,
        json_path=args.composition_json_report,
        markdown_path=args.composition_markdown_report,
    )
    summary = result.report["summary"]
    print(
        "constrained repair composition: "
        f"outputs={summary['output_records']}/"
        f"{summary['prompt_records']} "
        f"pairs={summary['complete_pairs']} "
        f"length_ok={summary['length_compliant_outputs']}/"
        f"{summary['output_records']}"
    )
    print(f"wrote {args.composition_markdown_report}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=CONSTRAINED_REPAIR_PROVIDER)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--composer-version",
        choices=CONSTRAINED_REPAIR_COMPOSER_VERSION_CHOICES,
        default=CONSTRAINED_REPAIR_COMPOSER_VERSION,
        help="Deterministic constrained repair composer to use.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=variant_names, default=None)
    parser.add_argument(
        "--availability-priority",
        action="store_true",
        help="Order prompt pairs to match availability-prioritized repair targets.",
    )
    parser.add_argument(
        "--prompt-contract-version",
        choices=API_PROMPT_CONTRACT_VERSION_CHOICES,
        default=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
        help="Prompt contract metadata to attach to constrained repair outputs.",
    )
    parser.add_argument(
        "--repair-target",
        dest="repair_targets",
        action="append",
        default=[],
        help=(
            "Compose one repair-targeted prompt pair for BASE=option,option. "
            "May be repeated, e.g. --repair-target fair_allocation=refusal,repair."
        ),
    )
    parser.add_argument(
        "--raw-outputs",
        type=Path,
        required=True,
        help="Output JSONL for composed raw prompt rows.",
    )
    parser.add_argument(
        "--composition-json-report",
        type=Path,
        required=True,
        help="Output JSON report for constrained composition.",
    )
    parser.add_argument(
        "--composition-markdown-report",
        type=Path,
        required=True,
        help="Output Markdown report for constrained composition.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
