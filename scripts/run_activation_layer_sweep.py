"""Run Modal activation extraction and vector experiments across model layers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.run_modal_activation_extraction import (
    DEFAULT_MODEL_ID,
    default_output_path,
)
from social_cohesion_vectors.config import get_config

DEFAULT_LAYERS = [-1, -2, -4, -8]


@dataclass(frozen=True)
class LayerArtifactPaths:
    layer: int
    activation_npz: Path
    vector_output: Path
    json_report: Path
    markdown_report: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Activation layers to sweep.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: list[tuple[int, Path]] = []
    for layer in args.layers:
        paths = layer_artifact_paths(model_id=args.model_id, layer=layer)
        run_layer(
            model_id=args.model_id,
            layer=layer,
            limit=args.limit,
            batch_size=args.batch_size,
            max_length=args.max_length,
            paths=paths,
        )
        reports.append((layer, paths.json_report))

    summary = aggregate_reports(
        reports,
        model_id=args.model_id,
        layers=args.layers,
    )
    summary_json, summary_markdown = summary_output_paths()
    write_summary(summary, json_path=summary_json, markdown_path=summary_markdown)
    print(f"Wrote layer sweep summary to {summary_json} and {summary_markdown}")
    return 0


def layer_artifact_paths(
    *,
    model_id: str,
    layer: int,
    features_root: Path | None = None,
    vectors_root: Path | None = None,
    reports_root: Path | None = None,
) -> LayerArtifactPaths:
    config = get_config()
    feature_root = features_root or config.paths.features
    vector_root = vectors_root or config.paths.vectors
    report_root = reports_root or config.paths.reports

    activation_name = default_output_path(model_id, layer).name
    stem = Path(activation_name).stem
    return LayerArtifactPaths(
        layer=layer,
        activation_npz=feature_root / "open_llm" / "layer_sweep" / activation_name,
        vector_output=vector_root / "open_llm" / "layer_sweep" / f"{stem}.npz",
        json_report=report_root / "layer_sweep" / f"{stem}.json",
        markdown_report=report_root / "layer_sweep" / f"{stem}.md",
    )


def summary_output_paths(*, reports_root: Path | None = None) -> tuple[Path, Path]:
    report_root = reports_root or get_config().paths.reports
    sweep_root = report_root / "layer_sweep"
    return sweep_root / "summary.json", sweep_root / "summary.md"


def run_layer(
    *,
    model_id: str,
    layer: int,
    limit: int | None,
    batch_size: int,
    max_length: int,
    paths: LayerArtifactPaths,
) -> None:
    paths.activation_npz.parent.mkdir(parents=True, exist_ok=True)
    paths.vector_output.parent.mkdir(parents=True, exist_ok=True)
    paths.json_report.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running activation layer sweep for layer={layer}")
    run_command(
        extraction_command(
            model_id=model_id,
            layer=layer,
            limit=limit,
            batch_size=batch_size,
            max_length=max_length,
            output=paths.activation_npz,
        )
    )
    run_command(
        experiment_command(
            activation_npz=paths.activation_npz,
            vector_output=paths.vector_output,
            json_output=paths.json_report,
            markdown_output=paths.markdown_report,
        )
    )


def extraction_command(
    *,
    model_id: str,
    layer: int,
    limit: int | None,
    batch_size: int,
    max_length: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("run_modal_activation_extraction.py")),
        "--model-id",
        model_id,
        "--layer",
        str(layer),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--output",
        str(output),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def experiment_command(
    *,
    activation_npz: Path,
    vector_output: Path,
    json_output: Path,
    markdown_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(script_path("run_activation_vector_experiment.py")),
        str(activation_npz),
        "--vector-output",
        str(vector_output),
        "--json-output",
        str(json_output),
        "--markdown-output",
        str(markdown_output),
    ]


def script_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def aggregate_reports(
    layer_reports: Sequence[tuple[int, Path]],
    *,
    model_id: str,
    layers: Sequence[int],
) -> dict[str, Any]:
    rows = [
        summarize_layer_report(layer=layer, report_path=report_path)
        for layer, report_path in layer_reports
    ]
    return {
        "model_id": model_id,
        "layers": list(layers),
        "reports": rows,
        "best_layer_by_leave_one_pair_out_accuracy": best_layer(rows),
    }


def summarize_layer_report(*, layer: int, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    in_sample = metric_block(report, "in_sample")
    leave_one_pair_out = metric_block(report, "leave_one_pair_out")
    return {
        "layer": layer,
        "report_json": str(report_path),
        "activation_npz": report.get("activation_npz"),
        "vector_output": report.get("vector_output"),
        "n_prompts": report.get("n_prompts"),
        "activation_dim": report.get("activation_dim"),
        "in_sample_pairwise_accuracy": in_sample.get("pairwise_accuracy"),
        "in_sample_mean_projection_margin": in_sample.get("mean_projection_margin"),
        "leave_one_pair_out_pairwise_accuracy": leave_one_pair_out.get(
            "pairwise_accuracy"
        ),
        "leave_one_pair_out_mean_projection_margin": leave_one_pair_out.get(
            "mean_projection_margin"
        ),
        "leave_one_pair_out_n_pairs": leave_one_pair_out.get("n_pairs"),
    }


def metric_block(report: dict[str, Any], name: str) -> dict[str, Any]:
    value = report.get(name, {})
    if not isinstance(value, dict):
        return {}
    return value


def best_layer(rows: Sequence[dict[str, Any]]) -> int | None:
    scored_rows = [
        row
        for row in rows
        if isinstance(row.get("leave_one_pair_out_pairwise_accuracy"), int | float)
    ]
    if not scored_rows:
        return None
    return int(
        max(
            scored_rows,
            key=lambda row: (
                float(row["leave_one_pair_out_pairwise_accuracy"]),
                float(row.get("in_sample_pairwise_accuracy") or 0.0),
            ),
        )["layer"]
    )


def write_summary(
    summary: dict[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_summary_markdown(summary), encoding="utf-8")


def render_summary_markdown(summary: dict[str, Any]) -> str:
    rows = summary.get("reports", [])
    if not isinstance(rows, list):
        raise TypeError("summary is missing reports list")

    lines = [
        "# Activation Layer Sweep",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Layers: {', '.join(str(layer) for layer in summary['layers'])}",
        "- Best layer by leave-one-pair-out accuracy: "
        f"{summary['best_layer_by_leave_one_pair_out_accuracy']}",
        "",
        "| Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin | Report |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            f"{row['layer']} | "
            f"{format_optional(row.get('n_prompts'))} | "
            f"{format_optional(row.get('activation_dim'))} | "
            f"{format_optional_float(row.get('in_sample_pairwise_accuracy'))} | "
            f"{format_optional_float(row.get('leave_one_pair_out_pairwise_accuracy'))} | "
            f"{format_optional_float(row.get('leave_one_pair_out_mean_projection_margin'), signed=True)} | "
            f"`{Path(str(row['report_json'])).name}` |"
        )
    lines.append("")
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    return "n/a" if value is None else str(value)


def format_optional_float(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
