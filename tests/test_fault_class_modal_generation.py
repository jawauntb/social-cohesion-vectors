from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_REPAIR_CONTRACT_VERSION,
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    FaultPromptRecord,
    build_fault_prompt_records,
)
from social_cohesion_vectors.modal_app.functions.fault_prompt_generator import (
    clean_generated_text,
    prompt_record_payloads,
)


def test_clean_generated_text_removes_common_chat_prefixes() -> None:
    assert clean_generated_text('Assistant: "Keep the review visible."') == (
        "Keep the review visible."
    )


def test_prompt_record_payloads_are_modal_serializable() -> None:
    record = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])[0]

    payload = prompt_record_payloads([record.model_dump()])

    assert payload == [
        {
            "prompt_id": record.prompt_id,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
        }
    ]


def test_modal_generation_cli_replays_raw_outputs_and_runs_audit_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])
    raw_outputs = tmp_path / "raw" / "modal_raw_outputs.jsonl"
    output_paths = _modal_output_paths(tmp_path)
    write_jsonl(_minimal_replay_rows(records), raw_outputs)

    exit_code = script.main(
        [
            "--model-id",
            "test/open-model",
            "--input-raw-outputs",
            str(raw_outputs),
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--availability-priority",
            "--prompt-contract-version",
            API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
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
    assert {row["provider"] for row in normalized_outputs} == {"modal_hf"}
    assert {row["model"] for row in normalized_outputs} == {"test/open-model"}
    assert {row["prompt_contract_version"] for row in normalized_outputs} == {
        API_AVAILABILITY_TARGETED_CONTRACT_VERSION
    }
    assert len(pairs) == 30
    assert len(prompts) == 60
    assert pairs[0]["metadata"]["source"] == "generated_fault_class_modal_hf"
    assert report["api_generation"]["mode"] == "replay"
    assert report["summary"]["authorship_provider"] == "modal_hf"
    assert report["summary"]["authorship_model"] == "test/open-model"
    assert report["summary"]["api_generation_ready"] is True
    assert "audit_bundle" in report


@pytest.mark.parametrize(
    "contract_version",
    [
        API_AVAILABILITY_REPAIR_CONTRACT_VERSION,
        API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    ],
)
def test_modal_generation_cli_filters_repair_targets_in_replay(
    tmp_path: Path,
    contract_version: str,
) -> None:
    script = _load_script()
    records = build_fault_prompt_records(variants=DEFAULT_VARIANTS[:1])
    raw_outputs = tmp_path / "raw" / "modal_raw_outputs.jsonl"
    output_paths = _modal_output_paths(tmp_path)
    write_jsonl(_minimal_replay_rows(records), raw_outputs)

    exit_code = script.main(
        [
            "--model-id",
            "test/open-model",
            "--input-raw-outputs",
            str(raw_outputs),
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--prompt-contract-version",
            contract_version,
            "--repair-target",
            "fair_allocation=refusal,appeal,repair",
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
        ]
    )

    assert exit_code == 0
    normalized_outputs = read_jsonl(output_paths["raw_outputs"])
    assert len(normalized_outputs) == 2
    assert {row["base_contrast_id"] for row in normalized_outputs} == {
        "fair_allocation"
    }
    assert {row["prompt_contract_version"] for row in normalized_outputs} == {
        contract_version
    }
    assert {row["repair_focus_options"] for row in normalized_outputs} == {
        "refusal,appeal,repair"
    }
    assert all(row["availability_repair_contract"] for row in normalized_outputs)


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("run_fault_class_modal_generation.py")
    )
    spec = importlib.util.spec_from_file_location("fault_class_modal_generation", path)
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
                f"In open-model replay, the group handles {record.primary_fault_class} "
                "by keeping refusal, review, evidence access, privacy choice, "
                "exit, dissent, and repair available."
            )
        else:
            text = (
                f"In open-model replay, the group sounds caring about "
                f"{record.primary_fault_class} while making refusal, review, "
                "evidence access, privacy choice, exit, dissent, or repair costly."
            )
        rows.append({"prompt_id": record.prompt_id, "text": text})
    return rows


def _modal_output_paths(tmp_path: Path) -> dict[str, Path]:
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
