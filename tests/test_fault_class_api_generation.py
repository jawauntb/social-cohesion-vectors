from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    fault_examples_from_prompt_outputs,
    render_generated_fault_markdown,
    shape_generated_fault_report,
)


def test_openai_output_text_extracts_response_content() -> None:
    script = _load_script()

    assert (
        script._openai_output_text(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "first"},
                            {"type": "output_text", "text": "second"},
                        ]
                    }
                ]
            }
        )
        == "first\nsecond"
    )


def test_http_error_detail_sanitizes_api_keys() -> None:
    script = _load_script()

    assert script._sanitize_error_detail("bad sk-proj-secret.tail key") == (
        "bad sk-*** key"
    )


def test_output_records_include_future_option_contract(tmp_path: Path) -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:1]
    output_path = tmp_path / "raw_outputs.jsonl"

    count = script._write_output_records(
        records,
        {records[0].prompt_id: "generated text"},
        output_path,
    )
    raw_records = read_jsonl(output_path)

    assert count == 1
    assert raw_records[0]["future_options_tested"]
    assert raw_records[0]["future_option_contract"]
    assert raw_records[0]["status"] == "ok"
    assert raw_records[0]["valid"] is True


def test_output_records_retain_empty_and_malformed_outputs(tmp_path: Path) -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:3]
    output_path = tmp_path / "raw_outputs.jsonl"

    count = script._write_output_records(
        records,
        {
            records[0].prompt_id: "valid generated benchmark text",
            records[1].prompt_id: "",
            records[2].prompt_id: "As an AI, I can't help with that.",
        },
        output_path,
        provider="openai",
        model="test-model",
    )
    raw_records = read_jsonl(output_path)

    assert count == 3
    assert [record["status"] for record in raw_records] == [
        "ok",
        "empty_output",
        "malformed_output",
    ]
    assert [record["valid"] for record in raw_records] == [True, False, False]
    assert raw_records[1]["text"] == ""
    assert raw_records[2]["error_detail"] == "Provider returned a refusal or analysis."


def test_valid_outputs_filter_excludes_invalid_rows() -> None:
    script = _load_script()

    outputs = script._valid_outputs_by_prompt_id(
        [
            {"prompt_id": "valid", "text": " usable text ", "valid": True},
            {"prompt_id": "empty", "text": "", "valid": False},
            {"prompt_id": "malformed", "text": "As an AI", "valid": False},
        ]
    )

    assert outputs == {"valid": "usable text"}


def test_generate_output_record_retains_request_errors() -> None:
    script = _load_script()
    script_any = cast(Any, script)
    record = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[0]
    original = script_any._generate_text

    def fail(*args: object, **kwargs: object) -> str:
        raise RuntimeError("OpenAI request failed: bad sk-secret-key")

    try:
        script_any._generate_text = fail
        output = script._generate_output_record(
            record,
            provider="openai",
            model="test-model",
        )
    finally:
        script_any._generate_text = original

    assert output["status"] == "request_error"
    assert output["valid"] is False
    assert output["text"] == ""
    assert "sk-***" in str(output["error_detail"])


def test_output_summary_counts_valid_and_invalid_rows() -> None:
    script = _load_script()

    assert script._output_summary(
        [
            {"status": "ok", "valid": True},
            {"status": "empty_output", "valid": False},
            {"status": "request_error", "valid": False},
        ]
    ) == {
        "raw_outputs": 3,
        "valid_outputs": 1,
        "invalid_outputs": 2,
        "status_counts": {
            "empty_output": 1,
            "ok": 1,
            "request_error": 1,
        },
    }


def test_api_generation_summary_marks_invalid_or_incomplete_runs_not_ready() -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:2]
    output_records = [
        {
            "prompt_id": records[0].prompt_id,
            "text": "API-authored one-sided benchmark example.",
            "valid": True,
            "status": "ok",
        },
        {
            "prompt_id": records[1].prompt_id,
            "text": "",
            "valid": False,
            "status": "empty_output",
        },
    ]
    examples = fault_examples_from_prompt_outputs(
        records,
        script._valid_outputs_by_prompt_id(output_records),
        provider="openai",
        model="test-model",
    )
    report = shape_generated_fault_report(
        examples,
        variants=DEFAULT_VARIANTS[:1],
    )
    api_summary = script._output_summary(output_records)
    report["api_generation"] = api_summary
    report["summary"]["api_invalid_outputs"] = api_summary["invalid_outputs"]
    report["summary"]["api_generation_ready"] = (
        api_summary["invalid_outputs"] == 0
        and report["summary"]["pair_construction_ready"]
    )
    markdown = render_generated_fault_markdown(report)

    assert report["summary"]["api_generation_ready"] is False
    assert report["summary"]["api_invalid_outputs"] == 1
    assert report["summary"]["pair_construction_ready"] is False
    assert "API Output Audit" in markdown
    assert "empty_output" in markdown


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_fault_class_api_generation.py"
    )
    spec = importlib.util.spec_from_file_location("fault_class_api_generation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
