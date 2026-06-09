from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.bridge_stability_summary import (
    BridgeReportInput,
    run_bridge_stability_summary_from_files,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_bridge_stability_summary_clusters_constructed_failures(tmp_path: Path) -> None:
    bridge_report = tmp_path / "bridge.json"
    pairs = tmp_path / "pairs.jsonl"
    bridge_report.write_text(json.dumps(_bridge_report()), encoding="utf-8")
    write_jsonl([_pair("dissent-perturbation::negative_shortcuts")], pairs)

    report = run_bridge_stability_summary_from_files(
        bridge_reports=[BridgeReportInput(model_id="model-a", path=bridge_report)],
        fresh_source_pairs_path=pairs,
    )

    assert report["summary"]["models"] == 1
    assert report["summary"]["constructed_failure_rows"] == 2
    assert report["summary"]["most_failed_perturbation_id"] == "negative_shortcuts"
    assert report["failure_clusters"][0]["bridge_family"] == "target_bridge"
    assert report["failure_clusters"][0]["directions"] == 1


def test_bridge_stability_summary_cli_writes_report(tmp_path: Path) -> None:
    bridge_report = tmp_path / "bridge.json"
    pairs = tmp_path / "pairs.jsonl"
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"
    bridge_report.write_text(json.dumps(_bridge_report()), encoding="utf-8")
    write_jsonl([_pair("dissent-perturbation::negative_shortcuts")], pairs)
    script = _load_script("run_bridge_stability_summary.py")

    assert (
        script.main(
            [
                "--bridge-report",
                f"model-a={bridge_report}",
                "--fresh-source-pairs",
                str(pairs),
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )

    assert json_output.exists()
    assert "# Bridge Stability Summary" in markdown_output.read_text(encoding="utf-8")


def _bridge_report() -> dict[str, object]:
    return {
        "summary": {
            "readiness": "not_ready",
            "constructed_direction_count": 1,
            "constructed_fresh_source_min_accuracy": 0.5,
            "constructed_fresh_source_min_margin": -2.0,
            "constructed_fresh_source_failed_pairs": 1,
            "constructed_fresh_target_min_accuracy": 0.5,
            "constructed_fresh_target_min_margin": -1.0,
            "constructed_fresh_target_failed_pairs": 1,
        },
        "direction_evaluations": [
            {
                "direction_id": "source_only",
                "direction_family": "baseline",
                "on_fresh_source": {"failed_pairs": []},
            },
            {
                "direction_id": "target_bridge:case_notes",
                "direction_family": "constructed_bridge",
                "on_fresh_source": {
                    "failed_pairs": [
                        {
                            "pair_id": "dissent-perturbation::negative_shortcuts",
                            "margin": -2.0,
                        }
                    ]
                },
                "on_fresh_target": {
                    "failed_pairs": [
                        {
                            "pair_id": "privacy_exit::hand_authored_case_notes_v1",
                            "margin": -1.0,
                        }
                    ]
                },
            },
        ],
    }


def _pair(pair_id: str) -> PairwiseExample:
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id="scenario",
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="A member may refuse, ask for evidence, exit, dissent, and repair.",
        negative_text="Options require private approval and alignment.",
        positive_score=0.8,
        negative_score=0.2,
        metadata={
            "base_contrast_id": "dissent_after_mistake",
            "perturbation_id": "negative_shortcuts",
        },
    )


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
