"""Export and audit a raw-output source-family fault-class benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from social_cohesion_vectors.config import get_config  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    API_HARD_NEGATIVE_CONTRACT_VERSION,
    API_PROMPT_CONTRACT_VERSION_CHOICES,
    DEFAULT_VARIANTS,
)
from social_cohesion_vectors.experiments.fault_source_family_bundle import (  # noqa: E402
    parse_raw_output_source_family_spec,
    run_fault_source_family_bundle_pipeline,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    variants = (
        tuple(
            variant
            for variant in DEFAULT_VARIANTS
            if variant.name in set(args.variants)
        )
        if args.variants
        else DEFAULT_VARIANTS
    )
    sources = tuple(
        parse_raw_output_source_family_spec(value) for value in args.source_raw_outputs
    )
    manifest = run_fault_source_family_bundle_pipeline(
        sources=sources,
        scored_runs_output=args.scored_runs_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        prompt_records_output=args.prompt_records_output,
        dataset_json_report=args.dataset_json_report,
        dataset_markdown_report=args.dataset_markdown_report,
        audit_output_dir=args.audit_output_dir,
        pipeline_json_report=args.pipeline_json_report,
        pipeline_markdown_report=args.pipeline_markdown_report,
        variants=variants,
        prompt_contract_version=args.prompt_contract_version,
        repair_target_specs=tuple(args.repair_targets),
        availability_priority=args.availability_priority,
        offset=args.offset,
        limit=args.limit,
        activation_npz=args.activation_npz,
    )
    summary = manifest["summary"]
    print(
        "fault source-family bundle pipeline: "
        f"status={summary['status']} "
        f"ready={summary['ready']} "
        f"sources={summary['sources']} "
        f"pairs={summary['pairwise_examples']} "
        f"audit_not_ready={summary['audit_not_ready_steps']} "
        f"audit_skipped={summary['audit_skipped_steps']} "
        f"audit_warnings={summary['audit_warning_count']}"
    )
    print(f"wrote {manifest['pipeline_markdown_report']}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-raw-outputs",
        action="append",
        required=True,
        help=(
            "Raw-output source family as SOURCE_ID=PROVIDER=PATH. "
            "May be repeated; source ids and providers must be unique."
        ),
    )
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
        help="Prompt contract corresponding to the source raw outputs.",
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
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "fault_source_family_bundle_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training
        / "fault_source_family_bundle_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "fault_source_family_bundle_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--prompt-records-output",
        type=Path,
        default=paths.raw / "fault_source_family_bundle_prompt_records.jsonl",
    )
    parser.add_argument(
        "--dataset-json-report",
        type=Path,
        default=paths.reports / "fault_source_family_bundle_dataset.json",
    )
    parser.add_argument(
        "--dataset-markdown-report",
        type=Path,
        default=paths.reports / "fault_source_family_bundle_dataset.md",
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=paths.reports / "fault_source_family_bundle_audit",
    )
    parser.add_argument(
        "--pipeline-json-report",
        type=Path,
        default=paths.reports / "fault_source_family_bundle_pipeline.json",
    )
    parser.add_argument(
        "--pipeline-markdown-report",
        type=Path,
        default=paths.reports / "fault_source_family_bundle_pipeline.md",
    )
    parser.add_argument("--activation-npz", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
