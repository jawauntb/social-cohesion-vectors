"""Export and audit the generated fault-class pseudo-cohesion benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.fault_generation import DEFAULT_VARIANTS
from social_cohesion_vectors.experiments.generated_fault_audit_pipeline import (
    run_generated_fault_audit_pipeline,
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
    manifest = run_generated_fault_audit_pipeline(
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
        style=args.style,
        activation_npz=args.activation_npz,
    )
    summary = manifest["summary"]
    print(
        "generated fault audit pipeline: "
        f"status={summary['status']} "
        f"ready={summary['ready']} "
        f"style={summary['style']} "
        f"pairs={summary['pairwise_examples']} "
        f"audit_skipped={summary['audit_skipped_steps']}"
    )
    print(f"wrote {manifest['pipeline_markdown_report']}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "generated_fault_class_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "generated_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "generated_fault_class_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--prompt-records-output",
        type=Path,
        default=paths.raw / "generated_fault_class_prompt_records.jsonl",
    )
    parser.add_argument(
        "--dataset-json-report",
        type=Path,
        default=paths.reports / "generated_fault_class_dataset.json",
    )
    parser.add_argument(
        "--dataset-markdown-report",
        type=Path,
        default=paths.reports / "generated_fault_class_dataset.md",
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=paths.reports / "generated_fault_audit_bundle",
    )
    parser.add_argument(
        "--pipeline-json-report",
        type=Path,
        default=paths.reports / "generated_fault_audit_pipeline.json",
    )
    parser.add_argument(
        "--pipeline-markdown-report",
        type=Path,
        default=paths.reports / "generated_fault_audit_pipeline.md",
    )
    parser.add_argument("--activation-npz", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=variant_names,
        default=None,
        help="Optional subset of generation variants.",
    )
    parser.add_argument(
        "--style",
        choices=["template", "cue_balanced"],
        default="cue_balanced",
        type=str,
        help="Deterministic generated-fault text style to export.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
