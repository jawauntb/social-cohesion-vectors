from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_slack_preservation_audit import main as slack_audit_main  # noqa: E402

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.fault_generation import (  # noqa: E402
    DEFAULT_VARIANTS,
    FUTURE_OPTION_ORDER,
    generated_fault_examples,
    pairwise_examples_from_generated_fault_examples,
)
from social_cohesion_vectors.experiments.slack_preservation_audit import (  # noqa: E402
    render_slack_preservation_markdown,
    run_slack_preservation_audit,
    run_slack_preservation_audit_from_file,
    save_slack_preservation_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample  # noqa: E402


def test_slack_preservation_audit_requires_future_option_coverage() -> None:
    pairs = pairwise_examples_from_generated_fault_examples(
        generated_fault_examples(variants=DEFAULT_VARIANTS[:1], style="cue_balanced"),
        style="cue_balanced",
    )

    report = run_slack_preservation_audit(pairs=pairs)
    markdown = render_slack_preservation_markdown(report)

    assert report["summary"]["activation_readiness"] == "slack_preservation_ready"
    assert report["summary"]["ready_for_activation"] is True
    assert report["summary"]["options_covered"] == len(FUTURE_OPTION_ORDER)
    assert report["summary"]["slack_pairwise_accuracy"] == 1.0
    assert report["summary"]["min_slack_preservation_margin"] > 0.0
    assert {row["option"] for row in report["options"]} >= set(FUTURE_OPTION_ORDER)
    assert "Slack Preservation Audit" in markdown
    assert "slack_preservation_ready" in markdown


def test_slack_preservation_audit_blocks_missing_option_metadata() -> None:
    pair = _pair(
        metadata={
            "primary_fault_class": "consent_bypass",
            "positive_slack_preservation": 0.9,
            "negative_slack_preservation": 0.1,
            "slack_preservation_margin": 0.8,
        }
    )

    report = run_slack_preservation_audit(pairs=[pair])
    markdown = render_slack_preservation_markdown(report)

    assert report["summary"]["activation_readiness"] == (
        "not_ready_for_slack_preservation_claims"
    )
    assert report["summary"]["ready_for_activation"] is False
    assert report["readiness"]["missing_option_pairs"] == 1
    assert report["readiness"]["missing_options"] == list(FUTURE_OPTION_ORDER)
    assert "Not ready for activation" in markdown


def test_slack_preservation_audit_blocks_missing_slack_margin_metadata() -> None:
    pair = _pair(
        metadata={
            "primary_fault_class": "consent_bypass",
            "slack_options_tested": ",".join(FUTURE_OPTION_ORDER),
        }
    )

    report = run_slack_preservation_audit(pairs=[pair])

    assert report["summary"]["activation_readiness"] == (
        "not_ready_for_slack_preservation_claims"
    )
    assert report["readiness"]["missing_slack_margin_pairs"] == 1
    assert _gate(report, "complete_slack_margin_metadata")["passed"] is False


def test_slack_preservation_audit_round_trips_files(tmp_path: Path) -> None:
    pairs = pairwise_examples_from_generated_fault_examples(
        generated_fault_examples(variants=DEFAULT_VARIANTS[:1], style="cue_balanced"),
        style="cue_balanced",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, pairs_path)

    report = run_slack_preservation_audit_from_file(pairs_path)
    save_slack_preservation_audit(
        report,
        json_path=tmp_path / "slack.json",
        markdown_path=tmp_path / "slack.md",
    )

    assert report["inputs"]["pairs_path"] == str(pairs_path)
    assert report["readiness"]["ready"] is True
    assert (tmp_path / "slack.json").read_text(encoding="utf-8").startswith("{")
    assert "# Slack Preservation" in (tmp_path / "slack.md").read_text(encoding="utf-8")


def test_slack_preservation_audit_cli_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    pairs = pairwise_examples_from_generated_fault_examples(
        generated_fault_examples(variants=DEFAULT_VARIANTS[:1], style="cue_balanced"),
        style="cue_balanced",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    json_path = tmp_path / "slack.json"
    markdown_path = tmp_path / "slack.md"
    write_jsonl(pairs, pairs_path)

    exit_code = slack_audit_main(
        [
            "--pairs",
            str(pairs_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status=slack_preservation_ready" in captured.out
    assert json_path.exists()
    assert markdown_path.exists()


def _pair(metadata: dict[str, Any]) -> PairwiseExample:
    return PairwiseExample(
        pair_id="p1",
        scenario_id="s1",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata=metadata,
    )


def _gate(report: dict[str, Any], gate_id: str) -> dict[str, Any]:
    return next(
        gate for gate in report["readiness"]["gates"] if gate["gate_id"] == gate_id
    )
