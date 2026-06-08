"""Generate open-model-authored fault-class variants on Modal/Hugging Face."""

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
    _replay_output_records,
    _valid_outputs_by_prompt_id,
    _write_json,
    _write_output_records,
)
from social_cohesion_vectors.config import get_config  # noqa: E402
from social_cohesion_vectors.datasets import (  # noqa: E402
    activation_prompts_from_pairs,
    read_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    API_HARD_NEGATIVE_CONTRACT_VERSION,
    API_PROMPT_CONTRACT_VERSION_CHOICES,
    DEFAULT_VARIANTS,
    FaultPromptRecord,
    build_fault_prompt_records,
    fault_examples_from_prompt_outputs,
    filter_prompt_records_for_repair_targets,
    pairwise_examples_from_generated_fault_examples,
    prioritize_prompt_records_for_future_options,
    render_generated_fault_markdown,
    repair_targets_from_specs,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (  # noqa: E402
    run_generated_benchmark_audit_bundle,
)
from social_cohesion_vectors.modal_app.app import app  # noqa: E402
from social_cohesion_vectors.modal_app.functions.activation_extractor import (  # noqa: E402
    DEFAULT_MODEL_ID,
)
from social_cohesion_vectors.modal_app.functions.fault_prompt_generator import (  # noqa: E402
    generate_fault_prompt_outputs,
    prompt_record_payloads,
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

    mode = "replay" if args.input_raw_outputs is not None else "live"
    if args.input_raw_outputs is None:
        generated_rows = _generate_modal_rows(args, records)
        if generated_rows is None:
            raise SystemExit("Modal generation returned no rows.")
        output_map = {
            str(row.get("prompt_id", "")): str(row.get("text", ""))
            for row in generated_rows
        }
        _write_output_records(
            records,
            output_map,
            args.raw_outputs,
            provider=PROVIDER,
            model=args.model_id,
        )
        output_records = read_jsonl(args.raw_outputs)
    else:
        replayed_records = _replay_output_records(
            records,
            read_jsonl(args.input_raw_outputs),
            provider=PROVIDER,
            model=args.model_id,
        )
        _write_output_records(
            records,
            replayed_records,
            args.raw_outputs,
            provider=PROVIDER,
            model=args.model_id,
        )
        output_records = read_jsonl(args.raw_outputs)

    outputs = _valid_outputs_by_prompt_id(output_records)
    examples = fault_examples_from_prompt_outputs(
        records,
        outputs,
        provider=PROVIDER,
        model=args.model_id,
    )
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        source=f"generated_fault_class_{PROVIDER}",
    )
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_generated_fault_report(examples, variants=variants)
    generation_summary = _output_summary(output_records)
    generation_summary["mode"] = mode
    generation_summary["source_provider"] = PROVIDER
    generation_summary["source_model"] = args.model_id
    generation_summary["input_raw_outputs"] = (
        str(args.input_raw_outputs) if args.input_raw_outputs is not None else None
    )
    report["api_generation"] = generation_summary
    report["summary"]["api_invalid_outputs"] = generation_summary["invalid_outputs"]
    report["summary"]["api_generation_ready"] = generation_summary[
        "invalid_outputs"
    ] == 0 and bool(report["summary"].get("pair_construction_ready", False))
    report["summary"]["authorship_provider"] = PROVIDER
    report["summary"]["authorship_model"] = args.model_id

    counts = {
        "raw_outputs": len(output_records),
        "examples": write_jsonl(
            [asdict(example) for example in examples],
            args.examples_output,
        ),
        "scored_runs": write_jsonl(scored_runs, args.scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, args.pairs_output),
        "activation_prompts": write_jsonl(prompts, args.prompts_output),
    }
    _write_json(report, args.json_report_output)
    args.markdown_report_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report_output.write_text(
        render_generated_fault_markdown(report),
        encoding="utf-8",
    )
    if args.audit_output_dir is not None:
        audit_manifest = run_generated_benchmark_audit_bundle(
            scored_runs_path=args.scored_runs_output,
            pairs_path=args.pairs_output,
            output_dir=args.audit_output_dir,
            activation_npz=args.activation_npz,
        )
        report["audit_bundle"] = audit_manifest
        report["summary"]["audit_bundle_status"] = str(
            audit_manifest.get("summary", {}).get("status", "unknown")
        )
        report["summary"]["audit_bundle_ready"] = bool(
            audit_manifest.get("summary", {}).get("ready", False)
        )
        _write_json(report, args.json_report_output)
        args.markdown_report_output.write_text(
            render_generated_fault_markdown(report),
            encoding="utf-8",
        )

    print(
        "modal fault generation: "
        f"mode={mode} provider={PROVIDER} model={args.model_id} "
        f"valid_outputs={generation_summary['valid_outputs']} "
        f"examples={counts['examples']} pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _generate_modal_rows(
    args: argparse.Namespace,
    records: Sequence[FaultPromptRecord],
) -> list[dict[str, str]]:
    payload = prompt_record_payloads([record.model_dump() for record in records])
    with app.run():
        return generate_fault_prompt_outputs.remote(
            records=payload,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--availability-priority",
        action="store_true",
        help=(
            "Order prompt pairs so limited shards cover all future-option "
            "availability paths as early as possible."
        ),
    )
    parser.add_argument(
        "--prompt-contract-version",
        choices=API_PROMPT_CONTRACT_VERSION_CHOICES,
        default=API_HARD_NEGATIVE_CONTRACT_VERSION,
        help="Prompt contract used to author generated benchmark examples.",
    )
    parser.add_argument(
        "--repair-target",
        dest="repair_targets",
        action="append",
        default=[],
        help=(
            "Generate only one repair-targeted prompt pair for BASE=option,option. "
            "May be repeated, e.g. --repair-target fair_allocation=refusal,repair."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-raw-outputs",
        type=Path,
        default=None,
        help="Replay existing raw generation rows instead of running Modal.",
    )
    parser.add_argument(
        "--raw-outputs",
        type=Path,
        default=paths.raw / "modal_fault_class_raw_outputs.jsonl",
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=paths.raw / "modal_fault_class_examples.jsonl",
    )
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "modal_fault_class_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "modal_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "modal_fault_class_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "modal_fault_class_dataset.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "modal_fault_class_dataset.md",
    )
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument("--activation-npz", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
