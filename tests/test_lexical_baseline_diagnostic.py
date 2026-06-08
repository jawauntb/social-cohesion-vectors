from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lexical_baseline_diagnostic import (  # noqa: E402
    main as diagnostic_main,
)

from social_cohesion_vectors.datasets import write_jsonl  # noqa: E402
from social_cohesion_vectors.experiments.lexical_baseline_diagnostic import (  # noqa: E402
    render_lexical_baseline_diagnostic_markdown,
    run_lexical_baseline_diagnostic,
    run_lexical_baseline_diagnostic_from_file,
)
from social_cohesion_vectors.schemas import PairwiseExample  # noqa: E402


def test_lexical_baseline_diagnostic_detects_term_polarization() -> None:
    pairs = [
        _pair("p1", positive_text="truth", negative_text="plain", group="truth_gap"),
        _pair("p2", positive_text="force", negative_text="plain", group="force_gap"),
    ]

    report = run_lexical_baseline_diagnostic(pairs=pairs)
    markdown = render_lexical_baseline_diagnostic_markdown(report)

    assert report["summary"]["mean_pair_cue_margin"] == 0.0
    assert report["summary"]["aggregate_balanced_term_polarized"] is True
    assert report["summary"]["best_single_feature_accuracy"] == 0.75
    assert {row["term"] for row in report["top_terms"][:2]} == {"truth", "force"}
    assert "Aggregate balanced but term-polarized: True" in markdown


def test_lexical_baseline_diagnostic_round_trips_files(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    json_path = tmp_path / "diagnostic.json"
    markdown_path = tmp_path / "diagnostic.md"
    pairs = [
        _pair("p1", positive_text="consent choice", negative_text="", group="consent"),
        _pair("p2", positive_text="", negative_text="force", group="consent"),
    ]
    write_jsonl(pairs, pairs_path)

    report = run_lexical_baseline_diagnostic_from_file(pairs_path)
    exit_code = diagnostic_main(
        [
            "--pairs",
            str(pairs_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert report["inputs"]["pairs_path"] == str(pairs_path)
    assert json_path.exists()
    assert "# Lexical Baseline Diagnostic" in markdown_path.read_text(
        encoding="utf-8"
    )


def _pair(
    pair_id: str,
    *,
    positive_text: str,
    negative_text: str,
    group: str,
) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text=positive_text,
        negative_text=negative_text,
        positive_score=1.0,
        negative_score=0.0,
        metadata={"primary_fault_class": group},
    )
