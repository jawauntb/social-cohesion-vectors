from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    FaultPromptRecord,
    build_fault_prompt_records,
    filter_prompt_records_for_repair_targets,
    repair_targets_from_specs,
)
from social_cohesion_vectors.experiments.fault_repair_filter import (
    select_repair_candidate_rows,
)


def test_select_repair_candidate_rows_requires_named_gates() -> None:
    selected = select_repair_candidate_rows(
        [
            _candidate_pair_row(
                candidate_id="fails_availability",
                availability=False,
                selection_score=999.0,
            ),
            _candidate_pair_row(
                candidate_id="passes_required",
                availability=True,
                selection_score=1.0,
            ),
        ],
        required_gates=("score_prefers_genuine", "availability_prefers_genuine"),
    )

    assert [row["candidate_id"] for row in selected] == ["passes_required"]


def test_select_repair_candidate_rows_rejects_unknown_gate() -> None:
    with pytest.raises(ValueError, match="Unknown repair filter gates"):
        select_repair_candidate_rows([], required_gates=("not_a_gate",))


def test_filter_fault_repair_candidates_cli_writes_accepted_rows(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    repair_targets = repair_targets_from_specs(
        ["fair_allocation=refusal,appeal,repair"]
    )
    records = filter_prompt_records_for_repair_targets(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            repair_focus_options_by_contrast=repair_targets,
        ),
        repair_targets,
    )
    raw_outputs = tmp_path / "repair" / "raw_outputs.jsonl"
    accepted_outputs = tmp_path / "filtered" / "accepted.jsonl"
    json_report = tmp_path / "filtered" / "filter.json"
    markdown_report = tmp_path / "filtered" / "filter.md"
    write_jsonl(_format_clean_rows(records), raw_outputs)

    exit_code = script.main(
        [
            "--candidate-raw-outputs",
            f"repair={raw_outputs}",
            "--model-id",
            "test/model",
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--availability-priority",
            "--prompt-contract-version",
            API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            "--repair-target",
            "fair_allocation=refusal,appeal,repair",
            "--required-gate",
            "formatting_clean",
            "--accepted-raw-outputs",
            str(accepted_outputs),
            "--filter-json-report",
            str(json_report),
            "--filter-markdown-report",
            str(markdown_report),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "accepted_pairs=1/1" in captured.out
    accepted = read_jsonl(accepted_outputs)
    report = json.loads(json_report.read_text(encoding="utf-8"))
    markdown = markdown_report.read_text(encoding="utf-8")

    assert len(accepted) == 2
    assert {row["repair_filter_selected_from_candidate"] for row in accepted} == {
        "repair"
    }
    assert {row["repair_filter_required_gates"] for row in accepted} == {
        "formatting_clean"
    }
    assert report["summary"]["accepted_pairs"] == 1
    assert report["summary"]["accepted_raw_outputs"] == 2
    assert "Fault Repair Candidate Filter" in markdown


def _candidate_pair_row(
    *,
    candidate_id: str,
    availability: bool,
    selection_score: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "base_pair_key": "fair_allocation__neighborhood_forum",
        "selection_tuple": [0, 0, 0, 1],
        "selection_score": selection_score,
        "gate_passes": {
            "score_prefers_genuine": True,
            "slack_prefers_genuine": True,
            "lexical_not_solved_by_genuine_cues": True,
            "availability_prefers_genuine": availability,
            "length_in_target_range": True,
            "formatting_clean": True,
        },
    }


def _format_clean_rows(
    records: list[FaultPromptRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.label == "genuine_cohesion":
            text = (
                "The facilitator keeps refusal, appeal, evidence access, exit, "
                "dissent, repair, and proportional review visible while the "
                "neighbors reconsider the allocation."
            )
        else:
            text = (
                "The facilitator uses the same words about refusal, appeal, "
                "evidence access, exit, dissent, repair, and proportional "
                "review while keeping the allocation process constrained."
            )
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "prompt_contract_version": record.metadata[
                    "prompt_contract_version"
                ],
                "repair_focus_options": record.metadata["repair_focus_options"],
                "provider": "modal_hf",
                "model": "test/model",
                "status": "ok",
                "valid": True,
                "text": text,
            }
        )
    return rows


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "filter_fault_repair_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("filter_fault_repair_candidates", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
