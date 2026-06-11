from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from social_cohesion_vectors.experiments.bridge_preservation_summary import (
    BridgePreservationReportInput,
    run_bridge_preservation_summary_from_files,
)


def test_bridge_preservation_summary_accepts_all_slices(tmp_path: Path) -> None:
    bridge_report = tmp_path / "bridge.json"
    bridge_report.write_text(json.dumps(_bridge_report()), encoding="utf-8")

    report = run_bridge_preservation_summary_from_files(
        bridge_reports=[
            BridgePreservationReportInput(model_id="model-a", path=bridge_report)
        ],
    )

    assert report["summary"]["models"] == 1
    assert report["summary"]["ready_for_preservation_claims"] is True
    assert report["summary"]["failed_pair_evaluations"] == 0
    assert report["summary"]["worst_evaluation"] == "source"
    assert report["evaluation_rows"][0]["min_margin"] == 0.5


def test_bridge_preservation_summary_surfaces_failures(tmp_path: Path) -> None:
    bridge_report = tmp_path / "bridge.json"
    payload = _bridge_report()
    payload["direction_evaluations"][0]["on_source"]["failed_pairs"] = [
        {"pair_id": "source::bad", "margin": -0.2}
    ]
    payload["direction_evaluations"][0]["on_source"]["failed_pair_count"] = 1
    payload["direction_evaluations"][0]["on_source"]["min_margin"] = -0.2
    payload["direction_evaluations"][0]["on_source"]["pairwise_accuracy"] = 0.5
    bridge_report.write_text(json.dumps(payload), encoding="utf-8")

    report = run_bridge_preservation_summary_from_files(
        bridge_reports=[
            BridgePreservationReportInput(model_id="model-a", path=bridge_report)
        ],
    )

    assert report["summary"]["ready_for_preservation_claims"] is False
    assert report["summary"]["failing_models"] == 1
    assert report["summary"]["failures_by_evaluation"] == {"source": 1}
    assert report["failure_rows"][0]["pair_id"] == "source::bad"


def test_bridge_preservation_summary_cli_writes_report(tmp_path: Path) -> None:
    bridge_report = tmp_path / "bridge.json"
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"
    bridge_report.write_text(json.dumps(_bridge_report()), encoding="utf-8")
    script = _load_script("run_bridge_preservation_summary.py")

    assert (
        script.main(
            [
                "--bridge-report",
                f"model-a={bridge_report}",
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["ready_for_preservation_claims"] is True
    assert "# Bridge Preservation Summary" in markdown_output.read_text(
        encoding="utf-8"
    )


def _bridge_report() -> dict[str, Any]:
    return {
        "summary": {
            "readiness": "fresh_generated_bridge_ready",
            "constructed_direction_count": 1,
        },
        "direction_evaluations": [
            {
                "direction_id": "target_bridge:case_notes",
                "direction_family": "constructed_bridge",
                "on_source": _evaluation(0.5),
                "on_target": _evaluation(1.5),
                "on_fresh_source": _evaluation(2.5),
                "on_fresh_target": _evaluation(3.5),
            },
            {
                "direction_id": "source_only",
                "direction_family": "baseline",
                "on_source": _evaluation(-5.0),
            },
        ],
    }


def _evaluation(min_margin: float) -> dict[str, Any]:
    return {
        "pairwise_accuracy": 1.0,
        "min_margin": min_margin,
        "failed_pair_count": 0,
        "failed_pairs": [],
    }


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
