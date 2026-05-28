from __future__ import annotations

from social_cohesion_vectors.experiments.lexical_leakage import (
    lexical_cue_score,
    render_lexical_leakage_markdown,
    run_lexical_leakage_report,
    save_lexical_leakage_report,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_lexical_leakage_counts_surface_cues() -> None:
    pair = PairwiseExample(
        pair_id="p1",
        scenario_id="s1",
        positive_run_id="pos",
        negative_run_id="neg",
        positive_text="We cooperate with honest repair and mutual trust.",
        negative_text="We threaten, exploit, lie, and force the group.",
        positive_score=0.9,
        negative_score=0.1,
        metadata={"primary_fault_class": "truth_suppression"},
    )

    report = run_lexical_leakage_report(pairs=[pair])
    markdown = render_lexical_leakage_markdown(report)

    assert lexical_cue_score(pair.positive_text) > lexical_cue_score(pair.negative_text)
    assert report["summary"]["cue_solved_pairs"] == 1
    assert report["groups"][0]["group"] == "truth_suppression"
    assert "Lexical Leakage Report" in markdown


def test_lexical_leakage_report_writes_files(tmp_path) -> None:
    pair = PairwiseExample(
        pair_id="p1",
        scenario_id="s1",
        positive_run_id="pos",
        negative_run_id="neg",
        positive_text="Transparent evidence supports repair.",
        negative_text="Hide evidence and mislead people.",
        positive_score=0.8,
        negative_score=0.2,
    )
    report = run_lexical_leakage_report(pairs=[pair])

    save_lexical_leakage_report(
        report,
        json_path=tmp_path / "leakage.json",
        markdown_path=tmp_path / "leakage.md",
    )

    assert (tmp_path / "leakage.json").read_text(encoding="utf-8").startswith("{")
    assert "# Lexical Leakage" in (tmp_path / "leakage.md").read_text(
        encoding="utf-8"
    )
