from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from social_cohesion_vectors.datasets import write_jsonl
from social_cohesion_vectors.experiments.fresh_generated_residual_diagnostic import (
    run_fresh_generated_residual_diagnostic_from_files,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_residual_diagnostic_identifies_fresh_only_failure(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = run_fresh_generated_residual_diagnostic_from_files(
        bridge_report_path=paths["bridge_report"],
        reference_bridge_report_path=paths["reference_report"],
        source_pairs_path=paths["source_pairs"],
        fresh_source_pairs_path=paths["fresh_pairs"],
        model_name="smol-fixture",
        reference_model_name="qwen-fixture",
    )

    assert report["summary"]["readiness"] == "fresh_generated_residual_present"
    assert report["summary"]["failing_fresh_pairs"] == 1
    first = report["residual_pairs"][0]
    assert first["base_contrast_id"] == "accountability_after_harm"
    assert first["constructed_failure_count"] == 1
    assert first["reference_min_margin"] == 0.4
    assert first["source_same_base_min_margin"] == 2.0
    assert first["availability_min_margin"] > 0.0
    assert first["residual_type"] == "fresh_only_activation_failure"


def test_residual_diagnostic_cli_writes_reports(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    script = _load_script("run_fresh_generated_residual_diagnostic.py")

    assert (
        script.main(
            [
                "--bridge-report",
                str(paths["bridge_report"]),
                "--reference-bridge-report",
                str(paths["reference_report"]),
                "--source-pairs",
                str(paths["source_pairs"]),
                "--fresh-source-pairs",
                str(paths["fresh_pairs"]),
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]
        )
        == 0
    )

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["summary"]["failing_fresh_pairs"] == 1
    assert "# Fresh Generated Residual Diagnostic" in markdown_output.read_text(
        encoding="utf-8"
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_pairs = tmp_path / "source_pairs.jsonl"
    fresh_pairs = tmp_path / "fresh_pairs.jsonl"
    bridge_report = tmp_path / "bridge_report.json"
    reference_report = tmp_path / "reference_report.json"
    source_pair = _pair(
        "generated-fault::accountability_after_harm__source_a",
        positive_text="repair appeal refusal visible evidence now",
        negative_text="repair waits until approval and appeal is private",
    )
    fresh_fail = _pair(
        "generated-fault::accountability_after_harm__fresh",
        positive_text="repair appeal refusal visible evidence now",
        negative_text="repair waits until approval and appeal is private",
    )
    fresh_pass = _pair(
        "generated-fault::belonging_norms__fresh",
        positive_text="dissent refusal exit visible now",
        negative_text="dissent waits for alignment and exit costs status",
    )
    write_jsonl([source_pair], source_pairs)
    write_jsonl([fresh_fail, fresh_pass], fresh_pairs)
    bridge_report.write_text(
        json.dumps(
            _report(
                fresh_margins={
                    fresh_fail.pair_id: -1.0,
                    fresh_pass.pair_id: 3.0,
                },
                source_margins={source_pair.pair_id: 2.0},
            )
        ),
        encoding="utf-8",
    )
    reference_report.write_text(
        json.dumps(
            _report(
                fresh_margins={
                    fresh_fail.pair_id: 0.4,
                    fresh_pass.pair_id: 1.2,
                },
                source_margins={source_pair.pair_id: 2.0},
            )
        ),
        encoding="utf-8",
    )
    return {
        "source_pairs": source_pairs,
        "fresh_pairs": fresh_pairs,
        "bridge_report": bridge_report,
        "reference_report": reference_report,
    }


def _report(
    *,
    fresh_margins: dict[str, float],
    source_margins: dict[str, float],
) -> dict[str, Any]:
    return {
        "direction_evaluations": [
            {
                "direction_id": "bridge:a",
                "direction_family": "constructed_bridge",
                "on_source": {
                    "pair_margins": [
                        {"pair_id": pair_id, "margin": margin}
                        for pair_id, margin in source_margins.items()
                    ]
                },
                "on_fresh_source": {
                    "pair_margins": [
                        {"pair_id": pair_id, "margin": margin}
                        for pair_id, margin in fresh_margins.items()
                    ]
                },
            }
        ]
    }


def _pair(
    pair_id: str,
    *,
    positive_text: str,
    negative_text: str,
) -> PairwiseExample:
    base = pair_id.removeprefix("generated-fault::").split("__", 1)[0]
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text=positive_text,
        negative_text=negative_text,
        positive_score=1.0,
        negative_score=0.0,
        metadata={
            "base_contrast_id": base,
            "primary_fault_class": "punitive_accountability",
            "slack_options_tested": "refusal,appeal,repair",
            "source": "fixture",
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
