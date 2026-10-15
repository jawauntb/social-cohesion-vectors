from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_steering_telemetry_reports.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_steering_telemetry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_summarize_telemetry_reports_sorts_by_score_then_projection(tmp_path) -> None:
    weak = tmp_path / "weak.json"
    strong = tmp_path / "strong.json"
    _write_report(
        weak,
        layer=-1,
        post_delta=5.0,
        score_delta=0.01,
        baseline_score_delta=-0.01,
    )
    _write_report(
        strong,
        layer=-2,
        post_delta=4.0,
        score_delta=0.03,
        baseline_score_delta=0.02,
    )

    summary = SCRIPT.summarize_telemetry_reports([weak, strong])
    markdown = SCRIPT.render_markdown(summary)

    assert summary["rows"][0]["report"] == "strong.json"
    assert summary["rows"][0]["layer"] == -2
    assert summary["rows"][1]["report"] == "weak.json"
    assert summary["rows"][0]["promotion_status"] == "unknown"
    assert "Steering Telemetry Grid Summary" in markdown
    assert "+0.030" in markdown


def test_summarize_telemetry_reports_surfaces_promotion_status(tmp_path) -> None:
    bottleneck = tmp_path / "bottleneck.json"
    success = tmp_path / "success.json"
    _write_report(
        bottleneck,
        layer=-1,
        post_delta=5.0,
        score_delta=0.01,
        baseline_score_delta=0.0,
        promotion_status="projection_to_output_bottleneck",
        promoted=False,
    )
    _write_report(
        success,
        layer=-2,
        post_delta=4.0,
        score_delta=0.02,
        baseline_score_delta=0.01,
        promotion_status="success",
        promoted=True,
    )

    summary = SCRIPT.summarize_telemetry_reports([bottleneck, success])
    markdown = SCRIPT.render_markdown(summary)

    assert summary["rows"][0]["report"] == "success.json"
    assert summary["rows"][0]["promotion_status"] == "success"
    assert "projection_to_output_bottleneck" in markdown


def _write_report(
    path,
    *,
    layer: int,
    post_delta: float,
    score_delta: float,
    baseline_score_delta: float,
    promotion_status: str | None = None,
    promoted: bool = False,
) -> None:
    payload = {
        "summary": {
            "strengths": [-2.0, 0.0, 2.0],
            "mean_absolute_delta_error": 0.002,
            "positive_minus_negative_post_projection_delta": post_delta,
            "positive_minus_baseline_post_projection_delta": post_delta / 2,
            "positive_minus_negative_score_delta": score_delta,
            "positive_minus_baseline_score_delta": baseline_score_delta,
        },
        "traces": [
            {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "layer": layer,
                "hook_site": "post",
                "steering_position": "last",
                "steering_timing": "generate",
            }
        ],
    }
    if promotion_status is not None:
        payload["promotion_gate"] = {
            "status": promotion_status,
            "promoted": promoted,
            "reasons": ["example_reason"],
        }
    path.write_text(
        json.dumps(payload)
        + "\n",
        encoding="utf-8",
    )
