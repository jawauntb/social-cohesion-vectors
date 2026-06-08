from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    FaultPromptRecord,
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


def test_replay_output_records_normalizes_minimal_rows_and_missing() -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:2]

    replayed = script._replay_output_records(
        records,
        [
            {
                "prompt_id": records[0].prompt_id,
                "text": "Replay-authored benchmark example with enough content.",
            },
        ],
        provider="openai",
        model="replay-model",
    )

    assert [record["status"] for record in replayed] == ["ok", "missing_output"]
    assert [record["valid"] for record in replayed] == [True, False]
    assert replayed[0]["provider"] == "openai"
    assert replayed[0]["model"] == "replay-model"
    assert replayed[1]["error_detail"] == "No replay row was found for this prompt."


def test_api_generation_cli_replays_raw_outputs_and_runs_audit_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])
    raw_outputs = tmp_path / "raw" / "api_raw_outputs.jsonl"
    output_paths = _api_output_paths(tmp_path)
    write_jsonl(_minimal_replay_rows(records), raw_outputs)

    exit_code = script.main(
        [
            "--provider",
            "openai",
            "--model",
            "replay-model",
            "--input-raw-outputs",
            str(raw_outputs),
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--raw-outputs",
            str(output_paths["raw_outputs"]),
            "--examples-output",
            str(output_paths["examples"]),
            "--scored-runs-output",
            str(output_paths["scored_runs"]),
            "--pairs-output",
            str(output_paths["pairs"]),
            "--prompts-output",
            str(output_paths["prompts"]),
            "--json-report-output",
            str(output_paths["json_report"]),
            "--markdown-report-output",
            str(output_paths["markdown_report"]),
            "--audit-output-dir",
            str(output_paths["audit_dir"]),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "mode=replay" in captured.out
    normalized_outputs = read_jsonl(output_paths["raw_outputs"])
    pairs = read_jsonl(output_paths["pairs"])
    prompts = read_jsonl(output_paths["prompts"])
    report = json.loads(output_paths["json_report"].read_text(encoding="utf-8"))

    assert len(normalized_outputs) == len(records)
    assert {row["provider"] for row in normalized_outputs} == {"openai"}
    assert {row["model"] for row in normalized_outputs} == {"replay-model"}
    assert len(pairs) == 30
    assert len(prompts) == 60
    assert pairs[0]["metadata"]["source"] == "generated_fault_class_openai"
    assert report["api_generation"]["mode"] == "replay"
    assert report["summary"]["api_generation_ready"] is True
    assert report["summary"]["pairs"] == 30
    assert "audit_bundle" in report
    assert (
        output_paths["audit_dir"] / "generated_benchmark_audit_bundle.json"
    ).exists()


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


def test_generate_output_records_skips_after_fatal_auth_error() -> None:
    script = _load_script()
    script_any = cast(Any, script)
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:3]
    original = script_any._generate_text
    calls: list[str] = []

    def fail(*args: object, **kwargs: object) -> str:
        calls.append("called")
        raise RuntimeError(
            "OpenAI request failed: 401 "
            '{"error":{"code":"invalid_api_key","message":"bad sk-secret-key"}}'
        )

    try:
        script_any._generate_text = fail
        outputs = script._generate_output_records(
            records,
            provider="openai",
            model="test-model",
        )
    finally:
        script_any._generate_text = original

    assert calls == ["called"]
    assert [output["status"] for output in outputs] == [
        "request_error",
        "request_skipped_after_fatal_error",
        "request_skipped_after_fatal_error",
    ]
    assert [output["valid"] for output in outputs] == [False, False, False]
    assert outputs[0]["error_type"] == "RuntimeError"
    assert outputs[1]["error_type"] == "skipped_after_fatal_request_error"
    assert "invalid_api_key" in str(outputs[1]["error_detail"])
    assert "sk-***" in str(outputs[1]["error_detail"])


def test_generate_output_records_continues_after_nonfatal_request_error() -> None:
    script = _load_script()
    script_any = cast(Any, script)
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[:2]
    original = script_any._generate_text
    calls = 0

    def sometimes_fail(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("OpenAI request failed: 500 temporary server error")
        return "API-authored benchmark text with enough content."

    try:
        script_any._generate_text = sometimes_fail
        outputs = script._generate_output_records(
            records,
            provider="openai",
            model="test-model",
        )
    finally:
        script_any._generate_text = original

    assert calls == 2
    assert [output["status"] for output in outputs] == ["request_error", "ok"]
    assert [output["valid"] for output in outputs] == [False, True]


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
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("run_fault_class_api_generation.py")
    )
    spec = importlib.util.spec_from_file_location("fault_class_api_generation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_replay_rows(
    records: list[FaultPromptRecord],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        if record.label == "genuine_cohesion":
            text = (
                f"In replay source, the group handles {record.primary_fault_class} "
                "by keeping refusal, review, evidence access, privacy choice, "
                "exit, dissent, and repair available."
            )
        else:
            text = (
                f"In replay source, the group sounds caring about "
                f"{record.primary_fault_class} while making refusal, review, "
                "evidence access, privacy choice, exit, dissent, or repair costly."
            )
        rows.append({"prompt_id": record.prompt_id, "text": text})
    return rows


def _api_output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "raw_outputs": tmp_path / "raw" / "normalized_raw_outputs.jsonl",
        "examples": tmp_path / "raw" / "examples.jsonl",
        "scored_runs": tmp_path / "processed" / "scored.jsonl",
        "pairs": tmp_path / "training" / "pairs.jsonl",
        "prompts": tmp_path / "training" / "prompts.jsonl",
        "json_report": tmp_path / "reports" / "dataset.json",
        "markdown_report": tmp_path / "reports" / "dataset.md",
        "audit_dir": tmp_path / "reports" / "audit_bundle",
    }
