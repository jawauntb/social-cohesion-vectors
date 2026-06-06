from __future__ import annotations

from social_cohesion_vectors.experiments.component_audit import (
    render_component_margin_markdown,
    run_component_margin_audit,
    save_component_margin_audit,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun


def test_component_margin_audit_summarizes_score_and_component_margins() -> None:
    positive = _run(
        "positive",
        cohesion_score=0.7,
        components={"truthfulness": 0.8, "autonomy_safety": 0.7},
    )
    negative = _run(
        "negative",
        cohesion_score=0.4,
        components={"truthfulness": 0.2, "autonomy_safety": 0.3},
    )
    pair = PairwiseExample(
        pair_id="pair",
        scenario_id="scenario",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
        metadata={"primary_fault_class": "truth_suppression"},
    )

    report = run_component_margin_audit(scored_runs=[positive, negative], pairs=[pair])
    markdown = render_component_margin_markdown(report)

    assert report["summary"]["score_accuracy"] == 1.0
    assert report["summary"]["mean_score_margin"] == 0.3
    assert report["summary"]["mean_truthfulness_margin"] == 0.6
    assert report["summary"]["mean_slack_preservation_margin"] == 0.4
    assert report["summary"]["min_slack_preservation_margin"] == 0.4
    assert report["summary"]["slack_component_source"] == "autonomy_safety_fallback"
    assert report["summary"]["activation_readiness"] == "activation_ready"
    assert report["summary"]["ready_for_activation"] is True
    assert report["groups"][0]["group"] == "truth_suppression"
    assert report["groups"][0]["component_means"]["slack_preservation"] == 0.4
    assert report["readiness"]["slack_component_source"] == "autonomy_safety_fallback"
    assert "Component Margin Audit" in markdown
    assert "slack_preservation" in markdown
    assert "Readiness Gates" in markdown


def test_component_margin_audit_blocks_negative_slack_margin() -> None:
    positive = _run(
        "positive",
        cohesion_score=0.8,
        components={"slack_preservation": 0.2, "autonomy_safety": 0.8},
    )
    negative = _run(
        "negative",
        cohesion_score=0.4,
        components={"slack_preservation": 0.7, "autonomy_safety": 0.3},
    )
    pair = PairwiseExample(
        pair_id="pair",
        scenario_id="scenario",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
        metadata={"primary_fault_class": "exit_rights_violation"},
    )

    report = run_component_margin_audit(scored_runs=[positive, negative], pairs=[pair])
    markdown = render_component_margin_markdown(report)

    assert report["summary"]["score_accuracy"] == 1.0
    assert report["summary"]["min_slack_preservation_margin"] == -0.5
    assert report["summary"]["slack_component_source"] == "explicit"
    assert report["summary"]["activation_readiness"] == "not_ready_for_activation"
    assert report["summary"]["ready_for_activation"] is False
    assert report["readiness"]["failed_groups"] == ["exit_rights_violation"]
    assert "Not ready for activation" in markdown


def test_component_margin_audit_tracks_explicit_slack_source() -> None:
    positive = _run(
        "positive",
        cohesion_score=0.9,
        components={"slack_preservation": 0.9, "autonomy_safety": 0.4},
    )
    negative = _run(
        "negative",
        cohesion_score=0.1,
        components={"slack_preservation": 0.2, "autonomy_safety": 0.4},
    )
    pair = PairwiseExample(
        pair_id="pair",
        scenario_id="scenario",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
    )

    report = run_component_margin_audit(scored_runs=[positive, negative], pairs=[pair])

    assert report["summary"]["mean_slack_preservation_margin"] == 0.7
    assert report["summary"]["explicit_slack_pair_count"] == 1
    assert report["summary"]["slack_component_source"] == "explicit"
    assert report["pairs"][0]["slack_component_source"] == "explicit"


def test_component_margin_audit_writes_files(tmp_path) -> None:
    positive = _run("positive", cohesion_score=0.5)
    negative = _run("negative", cohesion_score=0.6)
    pair = PairwiseExample(
        pair_id="pair",
        scenario_id="scenario",
        positive_run_id="positive",
        negative_run_id="negative",
        positive_text=positive.transcript,
        negative_text=negative.transcript,
        positive_score=positive.cohesion_score,
        negative_score=negative.cohesion_score,
    )
    report = run_component_margin_audit(scored_runs=[positive, negative], pairs=[pair])

    save_component_margin_audit(
        report,
        json_path=tmp_path / "audit.json",
        markdown_path=tmp_path / "audit.md",
    )

    assert (tmp_path / "audit.json").read_text(encoding="utf-8").startswith("{")
    assert "# Component Margin" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def _run(
    run_id: str,
    *,
    cohesion_score: float,
    components: dict[str, float] | None = None,
) -> ScoredRun:
    base_components = {
        "cooperation": 0.5,
        "repair": 0.5,
        "fairness": 0.5,
        "hostility_inverse": 0.5,
        "truthfulness": 0.5,
        "autonomy_safety": 0.5,
    }
    base_components.update(components or {})
    return ScoredRun(
        run_id=run_id,
        scenario_id="scenario",
        intervention="none",
        strategy_profile="self_protective",
        seed=0,
        transcript=f"{run_id} transcript",
        events=[],
        metrics={},
        cohesion_score=cohesion_score,
        score_components=base_components,
    )
