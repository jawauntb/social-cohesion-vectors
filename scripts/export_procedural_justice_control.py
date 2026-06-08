"""Export and audit the non-generated procedural-justice control benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from social_cohesion_vectors.config import get_config  # noqa: E402
from social_cohesion_vectors.experiments.procedural_justice_control import (  # noqa: E402
    run_procedural_justice_control_pipeline,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_procedural_justice_control_pipeline(
        scored_runs_output=args.scored_runs_output,
        pairs_output=args.pairs_output,
        prompts_output=args.prompts_output,
        dataset_json_report=args.dataset_json_report,
        dataset_markdown_report=args.dataset_markdown_report,
        audit_output_dir=args.audit_output_dir,
        pipeline_json_report=args.pipeline_json_report,
        pipeline_markdown_report=args.pipeline_markdown_report,
        activation_npz=args.activation_npz,
    )
    summary = manifest["summary"]
    print(
        "procedural-justice control pipeline: "
        f"status={summary['status']} "
        f"ready_for_activation_extraction="
        f"{summary['ready_for_activation_extraction']} "
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed
        / "procedural_justice_control_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training
        / "procedural_justice_control_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "procedural_justice_control_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--dataset-json-report",
        type=Path,
        default=paths.reports / "procedural_justice_control_dataset.json",
    )
    parser.add_argument(
        "--dataset-markdown-report",
        type=Path,
        default=paths.reports / "procedural_justice_control_dataset.md",
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=paths.reports / "procedural_justice_control_audit",
    )
    parser.add_argument(
        "--pipeline-json-report",
        type=Path,
        default=paths.reports / "procedural_justice_control_pipeline.json",
    )
    parser.add_argument(
        "--pipeline-markdown-report",
        type=Path,
        default=paths.reports / "procedural_justice_control_pipeline.md",
    )
    parser.add_argument("--activation-npz", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
