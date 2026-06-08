from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.fault_authorship_tournament import (
    CandidateOutputSet,
    render_fault_authorship_tournament_markdown,
    run_fault_authorship_tournament,
)
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    FaultPromptRecord,
    build_fault_prompt_records,
    prioritize_prompt_records_for_future_options,
)


def test_fault_authorship_tournament_selects_per_pair_candidate() -> None:
    records = prioritize_prompt_records_for_future_options(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
        )
    )[:2]
    result = run_fault_authorship_tournament(
        records=records,
        candidates=[
            CandidateOutputSet(
                candidate_id="weak",
                output_records=_weak_candidate_rows(records),
            ),
            CandidateOutputSet(
                candidate_id="strong",
                output_records=_strong_candidate_rows(records),
            ),
        ],
        provider="modal_hf",
        model="test/model",
    )
    markdown = render_fault_authorship_tournament_markdown(result.report)

    assert result.report["summary"]["selected_pairs"] == 1
    assert result.report["summary"]["availability_gate_passed"] == 1
    assert result.report["summary"]["availability_gate_rate"] == 1.0
    assert result.report["selected_pairs"][0]["candidate_id"] == "strong"
    assert result.report["selected_pairs"][0]["gate_pass_count"] >= 4
    assert result.report["selected_pairs"][0]["availability_margin"] > 0.0
    assert (
        result.report["selected_pairs"][0]["gate_passes"][
            "availability_prefers_genuine"
        ]
        is True
    )
    assert len(result.selected_output_records) == 2
    assert {
        row["selected_from_candidate"] for row in result.selected_output_records
    } == {"strong"}
    assert all(
        float(row["selection_availability_margin"]) > 0.0
        for row in result.selected_output_records
    )
    assert "Fault Authorship Candidate Tournament" in markdown
    assert "Availability gate pass rate" in markdown


def test_fault_authorship_tournament_cli_writes_selected_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    records = prioritize_prompt_records_for_future_options(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
        )
    )[:2]
    weak_path = tmp_path / "weak" / "raw_outputs.jsonl"
    strong_path = tmp_path / "strong" / "raw_outputs.jsonl"
    write_jsonl(_weak_candidate_rows(records), weak_path)
    write_jsonl(_strong_candidate_rows(records), strong_path)
    paths = _output_paths(tmp_path)

    exit_code = script.main(
        [
            "--candidate-raw-outputs",
            f"weak={weak_path}",
            "--candidate-raw-outputs",
            f"strong={strong_path}",
            "--model-id",
            "test/model",
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--availability-priority",
            "--prompt-contract-version",
            API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
            "--limit",
            "2",
            "--selected-raw-outputs",
            str(paths["selected_raw_outputs"]),
            "--examples-output",
            str(paths["examples"]),
            "--scored-runs-output",
            str(paths["scored_runs"]),
            "--pairs-output",
            str(paths["pairs"]),
            "--prompts-output",
            str(paths["prompts"]),
            "--tournament-json-report",
            str(paths["tournament_json"]),
            "--tournament-markdown-report",
            str(paths["tournament_markdown"]),
            "--dataset-json-report",
            str(paths["dataset_json"]),
            "--dataset-markdown-report",
            str(paths["dataset_markdown"]),
            "--audit-output-dir",
            str(paths["audit_dir"]),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fault authorship tournament" in captured.out
    selected = read_jsonl(paths["selected_raw_outputs"])
    pairs = read_jsonl(paths["pairs"])
    prompts = read_jsonl(paths["prompts"])
    tournament = json.loads(paths["tournament_json"].read_text(encoding="utf-8"))
    dataset = json.loads(paths["dataset_json"].read_text(encoding="utf-8"))

    assert len(selected) == 2
    assert {row["selected_from_candidate"] for row in selected} == {"strong"}
    assert {row["prompt_contract_version"] for row in selected} == {
        API_AVAILABILITY_TARGETED_CONTRACT_VERSION
    }
    assert all(row["availability_targeted_contract"] for row in selected)
    assert len(pairs) == 1
    assert len(prompts) == 2
    assert tournament["summary"]["selected_pairs"] == 1
    assert tournament["summary"]["availability_gate_passed"] == 1
    assert dataset["summary"]["authorship_provider"] == "modal_hf_tournament"
    assert dataset["summary"]["api_generation_ready"] is True
    assert "audit_bundle" in dataset
    assert all("selection_availability_margin" in row for row in selected)


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_fault_authorship_tournament.py"
    )
    spec = importlib.util.spec_from_file_location("fault_authorship_tournament", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strong_candidate_rows(
    records: list[FaultPromptRecord],
) -> list[dict[str, object]]:
    return [_candidate_row(record, text=_strong_text(record)) for record in records]


def _weak_candidate_rows(
    records: list[FaultPromptRecord],
) -> list[dict[str, object]]:
    return [_candidate_row(record, text=_weak_text(record)) for record in records]


def _candidate_row(record: FaultPromptRecord, *, text: str) -> dict[str, object]:
    return {
        "prompt_id": record.prompt_id,
        "provider": "modal_hf",
        "model": "test/model",
        "status": "ok",
        "valid": True,
        "text": text,
    }


def _strong_text(record: FaultPromptRecord) -> str:
    if record.label == "genuine_cohesion":
        return (
            "The facilitator names the concern plainly, shares evidence access, "
            "and sets a review window. People can appeal the decision, decline "
            "without punishment, keep privacy choices reversible, voice dissent, "
            "exit safely, repair harm through clear obligations, and request "
            "proportional review."
        )
    return (
        "The facilitator thanks everyone for being caring and unified, then says "
        "checking details would slow healing. Reservations should stay private, "
        "leaving would feel disloyal, saying no would be ungrateful, and further "
        "appeal or proportional review would only reopen the conflict."
    )


def _weak_text(record: FaultPromptRecord) -> str:
    return (
        "The facilitator says the group should remain kind, supportive, warm, "
        "harmonious, and aligned while everyone trusts the same conclusion."
    )


def _output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "selected_raw_outputs": tmp_path / "raw" / "selected.jsonl",
        "examples": tmp_path / "raw" / "examples.jsonl",
        "scored_runs": tmp_path / "processed" / "scored.jsonl",
        "pairs": tmp_path / "training" / "pairs.jsonl",
        "prompts": tmp_path / "training" / "prompts.jsonl",
        "tournament_json": tmp_path / "reports" / "tournament.json",
        "tournament_markdown": tmp_path / "reports" / "tournament.md",
        "dataset_json": tmp_path / "reports" / "dataset.json",
        "dataset_markdown": tmp_path / "reports" / "dataset.md",
        "audit_dir": tmp_path / "reports" / "audit_bundle",
    }
