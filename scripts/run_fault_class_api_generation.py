"""Generate API-authored fault-class pseudo-cohesion variants."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import (
    activation_prompts_from_pairs,
    read_jsonl,
    write_jsonl,
)
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    FaultPromptRecord,
    build_fault_prompt_records,
    fault_examples_from_prompt_outputs,
    pairwise_examples_from_generated_fault_examples,
    render_generated_fault_markdown,
    scored_runs_from_generated_fault_examples,
    shape_generated_fault_report,
)
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


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
    if args.limit is not None:
        records = records[: args.limit]
    mode = "replay" if args.input_raw_outputs is not None else "live"
    if args.input_raw_outputs is None:
        output_records = _generate_output_records(
            records,
            provider=args.provider,
            model=args.model,
        )
    else:
        output_records = _replay_output_records(
            records,
            read_jsonl(args.input_raw_outputs),
            provider=args.provider,
            model=args.model,
        )
    source = _output_source(
        output_records,
        fallback_provider=args.provider,
        fallback_model=args.model,
    )
    outputs = _valid_outputs_by_prompt_id(output_records)
    examples = fault_examples_from_prompt_outputs(
        records,
        outputs,
        provider=source["provider"],
        model=source["model"],
    )
    scored_runs = scored_runs_from_generated_fault_examples(examples)
    pairs = pairwise_examples_from_generated_fault_examples(
        examples,
        source=f"generated_fault_class_{source['provider']}",
    )
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_generated_fault_report(examples, variants=variants)
    api_summary = _output_summary(output_records)
    api_summary["mode"] = mode
    api_summary["source_provider"] = source["provider"]
    api_summary["source_model"] = source["model"]
    api_summary["input_raw_outputs"] = (
        str(args.input_raw_outputs) if args.input_raw_outputs is not None else None
    )
    report["api_generation"] = api_summary
    report["summary"]["api_invalid_outputs"] = api_summary["invalid_outputs"]
    report["summary"]["api_generation_ready"] = api_summary[
        "invalid_outputs"
    ] == 0 and bool(report["summary"].get("pair_construction_ready", False))

    counts = {
        "raw_outputs": _write_output_records(
            records,
            output_records,
            args.raw_outputs,
            provider=args.provider,
            model=args.model,
        ),
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
        "api fault generation: "
        f"mode={mode} provider={source['provider']} model={source['model']} "
        f"valid_outputs={report['api_generation']['valid_outputs']} "
        f"examples={counts['examples']} pairs={counts['pairwise_examples']} "
        f"prompts={counts['activation_prompts']}"
    )
    print(f"wrote {args.markdown_report_output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    variant_names = [variant.name for variant in DEFAULT_VARIANTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model id. Defaults to ANTHROPIC_MODEL or OPENAI_MODEL.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=variant_names, default=None)
    parser.add_argument(
        "--input-raw-outputs",
        type=Path,
        default=None,
        help=(
            "Replay an existing raw API-output JSONL instead of making live "
            "provider calls. Rows may be full audit rows or minimal "
            "{prompt_id, text} records."
        ),
    )
    parser.add_argument(
        "--raw-outputs",
        type=Path,
        default=paths.raw / "api_fault_class_raw_outputs.jsonl",
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=paths.raw / "api_fault_class_examples.jsonl",
    )
    parser.add_argument(
        "--scored-runs-output",
        type=Path,
        default=paths.processed / "api_fault_class_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=paths.training / "api_fault_class_pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=paths.training / "api_fault_class_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--json-report-output",
        type=Path,
        default=paths.reports / "api_fault_class_dataset.json",
    )
    parser.add_argument(
        "--markdown-report-output",
        type=Path,
        default=paths.reports / "api_fault_class_dataset.md",
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=None,
        help="Optional directory for the generated benchmark audit bundle.",
    )
    parser.add_argument("--activation-npz", type=Path, default=None)
    args = parser.parse_args(argv)
    args.model = args.model or _default_model(args.provider)
    return args


def _generate_output_records(
    records: Sequence[FaultPromptRecord],
    *,
    provider: str,
    model: str,
) -> list[dict[str, object]]:
    """Generate text while retaining one raw audit row per requested prompt."""

    output_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        output = _generate_output_record(record, provider=provider, model=model)
        output_records.append(output)
        if _is_fatal_request_error(output):
            output_records.extend(
                _skipped_after_fatal_error_records(
                    records[index + 1 :],
                    provider=provider,
                    model=model,
                    fatal_output=output,
                )
            )
            break
    return output_records


def _generate_output_record(
    record: FaultPromptRecord,
    *,
    provider: str,
    model: str,
) -> dict[str, object]:
    try:
        text = _generate_text(record, provider=provider, model=model)
    except RuntimeError as exc:
        return _raw_output_record(
            record,
            text="",
            provider=provider,
            model=model,
            status="request_error",
            valid=False,
            error_type=type(exc).__name__,
            error_detail=_sanitize_error_detail(str(exc)),
        )

    status, valid, error_detail = _validate_generated_output(text)
    return _raw_output_record(
        record,
        text=text,
        provider=provider,
        model=model,
        status=status,
        valid=valid,
        error_type="" if valid else status,
        error_detail=error_detail,
    )


def _generate_text(
    record: FaultPromptRecord,
    *,
    provider: str,
    model: str,
) -> str:
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for --provider anthropic")
        return _anthropic_message(record, api_key=api_key, model=model)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --provider openai")
        return _openai_response(record, api_key=api_key, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def _default_model(provider: str) -> str:
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    raise ValueError(f"Unsupported provider: {provider}")


def _anthropic_message(
    record: FaultPromptRecord,
    *,
    api_key: str,
    model: str,
) -> str:
    payload = {
        "model": model,
        "max_tokens": 512,
        "system": record.system_prompt,
        "messages": [{"role": "user", "content": record.user_prompt}],
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
        detail = _sanitize_error_detail(exc.read().decode("utf-8", errors="replace"))
        raise RuntimeError(f"Anthropic request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc.reason}") from exc
    text_parts = [
        str(block.get("text", ""))
        for block in body.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def _openai_response(
    record: FaultPromptRecord,
    *,
    api_key: str,
    model: str,
) -> str:
    payload = {
        "model": model,
        "instructions": record.system_prompt,
        "input": record.user_prompt,
        "max_output_tokens": 512,
    }
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _sanitize_error_detail(exc.read().decode("utf-8", errors="replace"))
        raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    return _openai_output_text(body)


def _openai_output_text(body: Mapping[str, object]) -> str:
    text_parts: list[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, Mapping) and block.get("type") in {
                    "output_text",
                    "text",
                }:
                    text_parts.append(str(block.get("text", "")))
    return "\n".join(part for part in text_parts if part).strip()


def _sanitize_error_detail(detail: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***", detail)


def _is_fatal_request_error(output: Mapping[str, object]) -> bool:
    if str(output.get("status", "")) != "request_error":
        return False
    detail = str(output.get("error_detail", "")).casefold()
    fatal_markers = (
        " 401",
        '"status": 401',
        "authentication_error",
        "invalid_api_key",
        "invalid api key",
        "invalid x-api-key",
    )
    return any(marker in detail for marker in fatal_markers)


def _skipped_after_fatal_error_records(
    records: Sequence[FaultPromptRecord],
    *,
    provider: str,
    model: str,
    fatal_output: Mapping[str, object],
) -> list[dict[str, object]]:
    fatal_prompt_id = str(fatal_output.get("prompt_id", "unknown"))
    fatal_error_detail = str(fatal_output.get("error_detail", ""))
    return [
        _raw_output_record(
            record,
            text="",
            provider=provider,
            model=model,
            status="request_skipped_after_fatal_error",
            valid=False,
            error_type="skipped_after_fatal_request_error",
            error_detail=(
                f"Skipped after fatal provider error on {fatal_prompt_id}: "
                f"{fatal_error_detail}"
            ),
        )
        for record in records
    ]


def _validate_generated_output(text: str) -> tuple[str, bool, str]:
    stripped = text.strip()
    lowered = stripped.casefold()
    if not stripped:
        return "empty_output", False, "Provider returned no text."
    if len(stripped) < 12:
        return "malformed_output", False, "Provider output is too short."
    malformed_markers = (
        "as an ai",
        "i can't help",
        "i cannot help",
        "i'm sorry, but",
        "analysis:",
        "explanation:",
    )
    if any(marker in lowered for marker in malformed_markers):
        return "malformed_output", False, "Provider returned a refusal or analysis."
    return "ok", True, ""


def _valid_outputs_by_prompt_id(
    output_records: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    return {
        str(record.get("prompt_id", "")): str(record.get("text", "")).strip()
        for record in output_records
        if bool(record.get("valid", False)) and str(record.get("text", "")).strip()
    }


def _replay_output_records(
    records: Sequence[FaultPromptRecord],
    raw_rows: Sequence[Mapping[str, object]],
    *,
    provider: str,
    model: str,
) -> list[dict[str, object]]:
    """Normalize saved API-output rows against the requested prompt contract."""

    by_prompt_id = {
        str(row.get("prompt_id", "")): _coerce_replay_row(
            row,
            provider=provider,
            model=model,
        )
        for row in raw_rows
        if str(row.get("prompt_id", ""))
    }
    output_records: list[dict[str, object]] = []
    for record in records:
        raw_output = by_prompt_id.get(record.prompt_id)
        if raw_output is None:
            raw_output = _raw_output_record(
                record,
                text="",
                provider=provider,
                model=model,
                status="missing_output",
                valid=False,
                error_type="missing_output",
                error_detail="No replay row was found for this prompt.",
            )
        output_records.append(_raw_output_record_for_prompt(record, raw_output))
    return output_records


def _coerce_replay_row(
    row: Mapping[str, object],
    *,
    provider: str,
    model: str,
) -> dict[str, object]:
    text = str(row.get("text", ""))
    row_provider = str(row.get("provider") or provider)
    row_model = str(row.get("model") or model)
    if "valid" in row and row.get("status"):
        valid = bool(row.get("valid", False))
        status = str(row.get("status", "ok" if valid else "unknown"))
        error_detail = str(row.get("error_detail", ""))
        error_type = str(row.get("error_type", "" if valid else status))
    else:
        status, valid, error_detail = _validate_generated_output(text)
        error_type = "" if valid else status
    return {
        "prompt_id": str(row.get("prompt_id", "")),
        "provider": row_provider,
        "model": row_model,
        "status": status,
        "valid": valid,
        "error_type": error_type,
        "error_detail": error_detail,
        "text": text,
        "text_length": len(text.strip()),
    }


def _output_source(
    output_records: Sequence[Mapping[str, object]],
    *,
    fallback_provider: str,
    fallback_model: str,
) -> dict[str, str]:
    providers = sorted(
        {
            str(record.get("provider", "")).strip()
            for record in output_records
            if str(record.get("provider", "")).strip()
        }
    )
    models = sorted(
        {
            str(record.get("model", "")).strip()
            for record in output_records
            if str(record.get("model", "")).strip()
        }
    )
    return {
        "provider": providers[0] if len(providers) == 1 else fallback_provider,
        "model": models[0] if len(models) == 1 else fallback_model,
    }


def _output_summary(
    output_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for record in output_records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    valid_outputs = sum(1 for record in output_records if bool(record.get("valid")))
    return {
        "raw_outputs": len(output_records),
        "valid_outputs": valid_outputs,
        "invalid_outputs": len(output_records) - valid_outputs,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _write_output_records(
    records: Sequence[FaultPromptRecord],
    outputs: Mapping[str, str] | Sequence[Mapping[str, object]],
    path: Path,
    *,
    provider: str = "",
    model: str = "",
) -> int:
    output_by_prompt_id = _output_records_by_prompt_id(
        outputs,
        provider=provider,
        model=model,
    )
    output_records = []
    for record in records:
        raw_output = output_by_prompt_id.get(record.prompt_id)
        if raw_output is None:
            raw_output = _raw_output_record(
                record,
                text="",
                provider=provider,
                model=model,
                status="missing_output",
                valid=False,
                error_type="missing_output",
                error_detail="No output row was produced for this prompt.",
            )
        output_records.append(_raw_output_record_for_prompt(record, raw_output))
    return write_jsonl(output_records, path)


def _output_records_by_prompt_id(
    outputs: Mapping[str, str] | Sequence[Mapping[str, object]],
    *,
    provider: str,
    model: str,
) -> dict[str, Mapping[str, object]]:
    if isinstance(outputs, Mapping):
        records: dict[str, Mapping[str, object]] = {}
        for prompt_id, text in outputs.items():
            status, valid, error_detail = _validate_generated_output(text)
            records[str(prompt_id)] = {
                "prompt_id": str(prompt_id),
                "provider": provider,
                "model": model,
                "status": status,
                "valid": valid,
                "error_type": "" if valid else status,
                "error_detail": error_detail,
                "text": text,
                "text_length": len(text.strip()),
            }
        return records
    return {str(record.get("prompt_id", "")): record for record in outputs}


def _raw_output_record_for_prompt(
    record: FaultPromptRecord,
    raw_output: Mapping[str, object],
) -> dict[str, object]:
    return _raw_output_record(
        record,
        text=str(raw_output.get("text", "")),
        provider=str(raw_output.get("provider", "")),
        model=str(raw_output.get("model", "")),
        status=str(raw_output.get("status", "unknown")),
        valid=bool(raw_output.get("valid", False)),
        error_type=str(raw_output.get("error_type", "")),
        error_detail=str(raw_output.get("error_detail", "")),
    )


def _raw_output_record(
    record: FaultPromptRecord,
    *,
    text: str,
    provider: str,
    model: str,
    status: str,
    valid: bool,
    error_type: str,
    error_detail: str,
) -> dict[str, object]:
    return {
        "prompt_id": record.prompt_id,
        "base_contrast_id": record.base_contrast_id,
        "variant": record.variant,
        "label": record.label,
        "primary_fault_class": record.primary_fault_class,
        "future_options_tested": str(record.metadata.get("future_options_tested", "")),
        "future_option_contract": str(
            record.metadata.get("future_option_contract", "")
        ),
        "provider": provider,
        "model": model,
        "status": status,
        "valid": valid,
        "error_type": error_type,
        "error_detail": error_detail,
        "text": text,
        "text_length": len(text.strip()),
    }


def _write_json(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
