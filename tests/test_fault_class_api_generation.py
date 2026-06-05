from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
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
