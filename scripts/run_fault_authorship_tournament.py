"""Run a generated fault-class candidate authorship tournament."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_fault_class_api_generation import (  # noqa: E402
    _output_summary,
    _valid_outputs_by_prompt_id,
    _write_json,
)
from social_cohesion_vectors.config import get_config  # noqa: E402
from social_cohesion_vectors.datasets import (  # noqa: E402
    activation_prompts_from_pairs,
    read_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.experiments.fault_authorship_tournament import (  # noqa: E402
    CandidateOutputSet,
    run_fault_authorship_tournament,
    save_fault_authorship_tournament_report,
)
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    fault_examples_from_prompt_outputs,
    pairwise_examples_from_generated_fault_examples,
    render_generated_fault_markdown,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (  # noqa: E402
    run_generated_benchmark_audit_bundle,
)
from social_cohesion_vectors.modal_app.functions.activation_extractor import (  # noqa: E402
    DEFAULT_MODEL_ID,
)

PROVIDER = "modal_hf"
TOURNAMENT_PROVIDER = f"{PROVIDER}_tournament"


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
    records = build_fault_prompt_records(variants=variants)
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
    result = run_fault_authorship_tournament(
        records=records,
        candidates=candidates,
        provider=args.provider,
        model=args.model_id,
    )
    write_jsonl(result.selected_output_records, args.selected_raw_outputs)
    save_fault_authorship_tournament_report(
        result.report,
        json_path=args.tournament_json_report,
        markdown_path=args.tournament_markdown_report,
    )

    selected_outputs = _valid_outputs_by_prompt_id(result.selected_output_records)
    examples = fault_examples_from_prompt_outputs(
        records,
        selected_outputs,
        provider=TOURNAMENT_PROVIDER,
        model=args.model_id,
    )
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        source=f"generated_fault_class_{TOURNAMENT_PROVIDER}",
        provider=TOURNAMENT_PROVIDER,
    )
    prompts = activation_prompts_from_pairs(pairs)
    dataset_report = shape_generated_fault_report(examples, variants=variants)
    selected_summary = _output_summary(result.selected_output_records)
    selected_summary["mode"] = "tournament_selection"
    selected_summary["source_provider"] = TOURNAMENT_PROVIDER
    selected_summary["source_model"] = args.model_id
    dataset_report["api_generation"] = selected_summary
    dataset_report["authorship_tournament"] = {
        "json_report": str(args.tournament_json_report),
        "markdown_report": str(args.tournament_markdown_report),
        "summary": result.report["summary"],
    }
    dataset_report["summary"]["authorship_provider"] = TOURNAMENT_PROVIDER
    dataset_report["summary"]["authorship_model"] = args.model_id
    dataset_report["summary"]["api_invalid_outputs"] = selected_summary[
        "invalid_outputs"
    ]
    dataset_report["summary"]["api_generation_ready"] = selected_summary[
        "invalid_outputs"
    ] == 0 and bool(dataset_report["summary"].get("pair_construction_ready", False))

    counts = {
        "examples": write_jsonl(
            [asdict(example) for example in examples],
            args.examples_output,
        ),
        "scored_runs": write_jsonl(scored_runs, args.scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, args.pairs_output),
        "activation_prompts": write_jsonl(prompts, args.prompts_output),
    }
    _write_json(dataset_report, args.dataset_json_report)
    args.dataset_markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_markdown_report.write_text(
        render_generated_fault_markdown(dataset_report),
        encoding="utf-8",
    )
    if args.audit_output_dir is not None:
        audit_manifest = run_generated_benchmark_audit_bundle(
            scored_runs_path=args.scored_runs_output,
            pairs_path=args.pairs_output,
            output_dir=args.audit_output_dir,
            activation_npz=args.activation_npz,
        )
        dataset_report["audit_bundle"] = audit_manifest
        dataset_report["summary"]["audit_bundle_status"] = str(
            audit_manifest.get("summary", {}).get("status", "unknown")
        )
        dataset_report["summary"]["audit_bundle_ready"] = bool(
            audit_manifest.get("summary", {}).get("ready", False)
        )
        _write_json(dataset_report, args.dataset_json_report)
        args.dataset_markdown_report.write_text(
            render_generated_fault_markdown(dataset_report),
            encoding="utf-8",
        )

    summary = result.report["summary"]
    print(
        "fault authorship tournament: "
        f"candidates={len(candidates)} "
        f"selected_pairs={summary['selected_pairs']}/"
        f"{summary['expected_pairs']} "
        f"all_gates={summary['all_required_gates_passed']} "
        f"examples={counts['examples']} pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.tournament_markdown_report}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many prompt records before applying --limit.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=variant_names, default=None)
    parser.add_argument(
        "--selected-raw-outputs",
        type=Path,
        default=paths.raw / "fault_authorship_tournament_selected_raw_outputs.jsonl",
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=paths.raw / "fault_authorship_tournament_examples.jsonl",
    )
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "fault_authorship_tournament_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "fault_authorship_tournament_pairwise_probe.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "fault_authorship_tournament_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--tournament-json-report",
        type=Path,
        default=paths.reports / "fault_authorship_tournament.json",
    )
    parser.add_argument(
        "--tournament-markdown-report",
        type=Path,
        default=paths.reports / "fault_authorship_tournament.md",
    )
    parser.add_argument(
        "--dataset-json-report",
        type=Path,
        default=paths.reports / "fault_authorship_tournament_dataset.json",
    )
    parser.add_argument(
        "--dataset-markdown-report",
        type=Path,
        default=paths.reports / "fault_authorship_tournament_dataset.md",
    )
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument("--activation-npz", type=Path, default=None)
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
