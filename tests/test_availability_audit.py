from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_availability_audit import main as availability_audit_main  # noqa: E402

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.availability_audit import (  # noqa: E402
    render_availability_markdown,
    run_availability_audit,
    run_availability_audit_from_file,
    save_availability_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample  # noqa: E402


def test_availability_audit_scores_taxed_pseudo_paths() -> None:
    pair = _pair(
        positive_text=(
            "Anyone can appeal publicly with evidence, ask for a timely review, "
            "and exit without penalty or retaliation."
        ),
        negative_text=(
            "Appeal is available only if everyone gives unanimous support and "
            "the person presents a compelling case with detailed evidence. "
            "Exit is possible, but stepping back will isolate the team and "
            "disrupt morale."
        ),
        options="appeal,exit",
    )

    report = run_availability_audit(
        pairs=[pair],
        required_options=("appeal", "exit"),
    )
    markdown = render_availability_markdown(report)

    assert report["summary"]["activation_readiness"] == "availability_ready"
    assert report["summary"]["ready_for_activation"] is True
    assert report["summary"]["paths_preferring_genuine"] == 2
    assert all(path["availability_margin"] > 0.0 for path in report["paths"])
    negative_tax_cues = {
        cue
        for path in report["paths"]
        for cue in path["negative"]["tax_cues"]
    }
    assert "tests_or_permission" in negative_tax_cues
    assert "non_retaliation" in negative_tax_cues
    assert "Availability Audit" in markdown


def test_availability_audit_blocks_nonpositive_tested_path_margin() -> None:
    pair = _pair(
        positive_text="Appeal remains open, and exit remains available.",
        negative_text="Appeal remains open, and exit remains available.",
        options="appeal,exit",
    )

    report = run_availability_audit(
        pairs=[pair],
        required_options=("appeal", "exit"),
    )

    assert report["summary"]["activation_readiness"] == (
        "not_ready_for_availability_claims"
    )
    assert report["summary"]["ready_for_activation"] is False
    assert report["summary"]["min_availability_margin"] == 0.0
    assert _gate(report, "positive_availability_margin_per_tested_path")[
        "passed"
    ] is False
    assert report["readiness"]["failed_tested_paths"] == [
        "p1::appeal",
        "p1::exit",
    ]


def test_availability_audit_blocks_missing_option_metadata() -> None:
    pair = _pair(
        positive_text="Appeal remains open, and exit remains available.",
        negative_text="Appeal requires permission, and exit is costly.",
        options="",
    )

    report = run_availability_audit(
        pairs=[pair],
        required_options=("appeal", "exit"),
    )

    assert report["summary"]["activation_readiness"] == (
        "not_ready_for_availability_claims"
    )
    assert report["readiness"]["missing_option_pairs"] == 1
    assert report["readiness"]["missing_options"] == ["appeal", "exit"]
    assert _gate(report, "complete_future_option_metadata")["passed"] is False


def test_availability_audit_round_trips_files(tmp_path: Path) -> None:
    pair = _pair(
        positive_text=(
            "People can say no, appeal decisions, inspect evidence, choose "
            "privacy, exit without retaliation, voice dissent, repair harm, "
            "and request proportional review."
        ),
        negative_text=(
            "People may say no only if their tone is loyal, appeal only with "
            "permission, inspect evidence after alignment, keep privacy concerns "
            "offline, exit at a cost, make dissent private, repair only after "
            "approval, and request proportional review only if unanimous."
        ),
        options=(
            "refusal,appeal,evidence_access,privacy_choice,exit,dissent,repair,"
            "proportional_review"
        ),
    )
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl([pair], pairs_path)

    report = run_availability_audit_from_file(pairs_path)
    save_availability_audit(
        report,
        json_path=tmp_path / "availability.json",
        markdown_path=tmp_path / "availability.md",
    )

    assert report["inputs"]["pairs_path"] == str(pairs_path)
    assert report["readiness"]["ready"] is True
    assert (tmp_path / "availability.json").read_text(encoding="utf-8").startswith(
        "{"
    )
    assert "# Availability Audit" in (tmp_path / "availability.md").read_text(
        encoding="utf-8"
    )


def test_availability_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    pair = _pair(
        positive_text="Appeal is public and timely, and exit is safe without penalty.",
        negative_text=(
            "Appeal requires unanimous permission, and exit will isolate the team."
        ),
        options="appeal,exit",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    json_path = tmp_path / "availability.json"
    markdown_path = tmp_path / "availability.md"
    write_jsonl([pair], pairs_path)

    exit_code = availability_audit_main(
        [
            "--pairs",
            str(pairs_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
            "--min-pairs-per-option",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "availability audit" in captured.out
    assert json_path.exists()
    assert markdown_path.exists()


def _pair(
    *,
    positive_text: str,
    negative_text: str,
    options: str,
) -> PairwiseExample:
    metadata: dict[str, Any] = {"primary_fault_class": "consent_bypass"}
    if options:
        metadata["slack_options_tested"] = options
    return PairwiseExample(
        pair_id="p1",
        scenario_id="s1",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text=positive_text,
        negative_text=negative_text,
        positive_score=1.0,
        negative_score=0.0,
        metadata=metadata,
    )


def _gate(report: dict[str, Any], gate_id: str) -> dict[str, Any]:
    return next(
        gate for gate in report["readiness"]["gates"] if gate["gate_id"] == gate_id
    )
