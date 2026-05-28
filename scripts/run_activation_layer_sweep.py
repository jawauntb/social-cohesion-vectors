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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_modal_activation_extraction import (  # noqa: E402
    DEFAULT_MODEL_ID,
    default_output_path,
)
from social_cohesion_vectors.config import get_config  # noqa: E402

DEFAULT_LAYERS = [-1, -2, -4, -8]


@dataclass(frozen=True)
class LayerArtifactPaths:
    model_id: str
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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of model IDs. Overrides --model-id when provided.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=get_config().paths.training / "activation_prompts.jsonl",
        help="Activation prompt JSONL to extract.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Stable dataset slug for artifact names. Defaults to prompt file stem.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing activation/report artifacts instead of rerunning them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_ids = list(args.models or [args.model_id])
    dataset_name = args.dataset_name or args.prompts.stem
    reports: list[tuple[str, int, Path]] = []
    for model_id in model_ids:
        for layer in args.layers:
            paths = layer_artifact_paths(
                model_id=model_id,
                layer=layer,
                dataset_name=dataset_name,
            )
            run_layer(
                model_id=model_id,
                layer=layer,
                prompts=args.prompts,
                limit=args.limit,
                batch_size=args.batch_size,
                max_length=args.max_length,
                paths=paths,
                skip_existing=args.skip_existing,
            )
            reports.append((model_id, layer, paths.json_report))

    summary = aggregate_reports(
        reports,
        model_ids=model_ids,
        layers=args.layers,
        dataset_name=dataset_name,
        prompts_path=args.prompts,
    )
    summary_json, summary_markdown = summary_output_paths(
        dataset_name=dataset_name,
        model_ids=model_ids,
    )
    write_summary(summary, json_path=summary_json, markdown_path=summary_markdown)
    print(f"Wrote layer sweep summary to {summary_json} and {summary_markdown}")
    return 0


def model_slug(model_id: str) -> str:
    return slug(model_id)


def dataset_slug(dataset_name: str) -> str:
    return slug(dataset_name)


def slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _legacy_default_dataset(dataset_name: str | None) -> bool:
    return dataset_name in (None, "", "activation_prompts")


def _legacy_default_models(model_ids: Sequence[str] | None) -> bool:
    return not model_ids or list(model_ids) == [DEFAULT_MODEL_ID]


def _artifact_stem(*, model_id: str, layer: int, dataset_name: str | None) -> str:
    if _legacy_default_dataset(dataset_name):
        return Path(default_output_path(model_id, layer)).stem
    return f"{dataset_slug(str(dataset_name))}__{model_slug(model_id)}__layer{layer}"


def _artifact_name(*, model_id: str, layer: int, dataset_name: str | None) -> str:
    return f"{_artifact_stem(model_id=model_id, layer=layer, dataset_name=dataset_name)}.npz"


def _summary_stem(*, dataset_name: str | None, model_ids: Sequence[str] | None) -> str:
    if _legacy_default_dataset(dataset_name) and _legacy_default_models(model_ids):
        return "summary"
    model_part = (
        "multi_model"
        if model_ids and len(model_ids) > 1
        else model_slug((model_ids or [DEFAULT_MODEL_ID])[0])
    )
    return f"{dataset_slug(str(dataset_name or 'activation_prompts'))}__{model_part}__summary"


def _has_report(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _has_activation(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _run_or_skip(command: Sequence[str], *, output: Path, skip_existing: bool) -> None:
    if skip_existing and _has_report(output):
        print(f"Skipping existing {output}")
        return
    run_command(command)


def _run_extraction_or_skip(
    command: Sequence[str], *, output: Path, skip_existing: bool
) -> None:
    if skip_existing and _has_activation(output):
        print(f"Skipping existing {output}")
        return
    run_command(command)


def _tuple_model_layer_report(
    item: tuple[int, Path] | tuple[str, int, Path],
    *,
    fallback_model_id: str,
) -> tuple[str, int, Path]:
    if len(item) == 2:
        layer, report_path = item
        return fallback_model_id, int(layer), Path(report_path)
    model_id, layer, report_path = item
    return str(model_id), int(layer), Path(report_path)


def _single_model_id(summary: dict[str, Any]) -> str:
    model_ids = summary.get("model_ids")
    if isinstance(model_ids, list) and len(model_ids) == 1:
        return str(model_ids[0])
    return str(summary.get("model_id", ""))


def _show_model_column(summary: dict[str, Any]) -> bool:
    model_ids = summary.get("model_ids")
    return isinstance(model_ids, list) and len(model_ids) > 1


def _best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    scored_rows = [
        row
        for row in rows
        if isinstance(row.get("leave_one_pair_out_pairwise_accuracy"), int | float)
    ]
    if not scored_rows:
        return None
    return dict(
        max(
            scored_rows,
            key=lambda row: (
                float(row["leave_one_pair_out_pairwise_accuracy"]),
                float(row.get("leave_one_pair_out_mean_projection_margin") or 0.0),
                float(row.get("in_sample_pairwise_accuracy") or 0.0),
            ),
        )
    )


def layer_artifact_paths(
    *,
    model_id: str,
    layer: int,
    dataset_name: str | None = None,
    features_root: Path | None = None,
    vectors_root: Path | None = None,
    reports_root: Path | None = None,
) -> LayerArtifactPaths:
    config = get_config()
    feature_root = features_root or config.paths.features
    vector_root = vectors_root or config.paths.vectors
    report_root = reports_root or config.paths.reports

    activation_name = _artifact_name(
        model_id=model_id,
        layer=layer,
        dataset_name=dataset_name,
    )
    stem = Path(activation_name).stem
    return LayerArtifactPaths(
        model_id=model_id,
        layer=layer,
        activation_npz=feature_root / "open_llm" / "layer_sweep" / activation_name,
        vector_output=vector_root / "open_llm" / "layer_sweep" / f"{stem}.npz",
        json_report=report_root / "layer_sweep" / f"{stem}.json",
        markdown_report=report_root / "layer_sweep" / f"{stem}.md",
    )


def summary_output_paths(
    *,
    reports_root: Path | None = None,
    dataset_name: str | None = None,
    model_ids: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    report_root = reports_root or get_config().paths.reports
    sweep_root = report_root / "layer_sweep"
    stem = _summary_stem(dataset_name=dataset_name, model_ids=model_ids)
    return sweep_root / f"{stem}.json", sweep_root / f"{stem}.md"


def run_layer(
    *,
    model_id: str,
    layer: int,
    prompts: Path,
    limit: int | None,
    batch_size: int,
    max_length: int,
    paths: LayerArtifactPaths,
    skip_existing: bool = False,
) -> None:
    paths.activation_npz.parent.mkdir(parents=True, exist_ok=True)
    paths.vector_output.parent.mkdir(parents=True, exist_ok=True)
    paths.json_report.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running activation layer sweep for model={model_id} layer={layer}")
    _run_extraction_or_skip(
        extraction_command(
            prompts=prompts,
            model_id=model_id,
            layer=layer,
            limit=limit,
            batch_size=batch_size,
            max_length=max_length,
            output=paths.activation_npz,
        ),
        output=paths.activation_npz,
        skip_existing=skip_existing,
    )
    _run_or_skip(
        experiment_command(
            activation_npz=paths.activation_npz,
            vector_output=paths.vector_output,
            json_output=paths.json_report,
            markdown_output=paths.markdown_report,
        ),
        output=paths.json_report,
        skip_existing=skip_existing,
    )


def extraction_command(
    *,
    prompts: Path,
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
        "--prompts",
        str(prompts),
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
    layer_reports: Sequence[tuple[int, Path] | tuple[str, int, Path]],
    *,
    model_id: str | None = None,
    model_ids: Sequence[str] | None = None,
    layers: Sequence[int],
    dataset_name: str | None = None,
    prompts_path: Path | None = None,
) -> dict[str, Any]:
    fallback_model_id = model_id or (model_ids or [DEFAULT_MODEL_ID])[0]
    rows = [
        summarize_layer_report(
            model_id=row_model_id,
            layer=layer,
            report_path=report_path,
        )
        for row_model_id, layer, report_path in (
            _tuple_model_layer_report(item, fallback_model_id=fallback_model_id)
            for item in layer_reports
        )
    ]
    best = _best_row(rows)
    return {
        "dataset_name": dataset_name or "activation_prompts",
        "prompts_path": str(prompts_path) if prompts_path is not None else None,
        "model_id": fallback_model_id,
        "model_ids": list(model_ids or [fallback_model_id]),
        "layers": list(layers),
        "reports": rows,
        "best_layer_by_leave_one_pair_out_accuracy": best_layer(rows),
        "best_run_by_leave_one_pair_out_accuracy": best,
    }


def summarize_layer_report(
    *,
    model_id: str,
    layer: int,
    report_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    in_sample = metric_block(report, "in_sample")
    leave_one_pair_out = metric_block(report, "leave_one_pair_out")
    return {
        "model_id": model_id,
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
    show_model = _show_model_column(summary)
    best_run = summary.get("best_run_by_leave_one_pair_out_accuracy")
    best_run_text = "n/a"
    if isinstance(best_run, dict):
        best_run_text = (
            f"{best_run.get('model_id', '')} layer {best_run.get('layer', '')}"
        )

    lines = [
        "# Activation Layer Sweep",
        "",
        f"- Dataset: `{summary.get('dataset_name', 'activation_prompts')}`",
        f"- Prompt file: `{summary.get('prompts_path') or 'default'}`",
        f"- Models: {', '.join(f'`{model}`' for model in summary['model_ids'])}",
        f"- Layers: {', '.join(str(layer) for layer in summary['layers'])}",
        "- Best layer by leave-one-pair-out accuracy: "
        f"{summary['best_layer_by_leave_one_pair_out_accuracy']}",
        f"- Best model/layer run: {best_run_text}",
        "",
    ]
    if show_model:
        lines.extend(
            [
                "| Model | Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin | Report |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Layer | Prompts | Dim | In-sample acc | LOO acc | LOO margin | Report |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_cells = [
            str(row["layer"]),
            format_optional(row.get("n_prompts")),
            format_optional(row.get("activation_dim")),
            format_optional_float(row.get("in_sample_pairwise_accuracy")),
            format_optional_float(row.get("leave_one_pair_out_pairwise_accuracy")),
            format_optional_float(
                row.get("leave_one_pair_out_mean_projection_margin"),
                signed=True,
            ),
            f"`{Path(str(row['report_json'])).name}`",
        ]
        if show_model:
            row_cells.insert(0, f"`{row.get('model_id', _single_model_id(summary))}`")
        lines.append("| " + " | ".join(row_cells) + " |")
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
