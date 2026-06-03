from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.drosophila_substrate import (
    canonical_drosophila_toy_graph,
    export_drosophila_toy_artifacts,
    export_drosophila_transmitter_matrix_artifacts,
    rank_transmitter_matrix_results,
    render_drosophila_toy_markdown,
    render_drosophila_transmitter_matrix_markdown,
    run_edge_scaling_sweep,
    run_transmitter_coefficient_matrix,
    shape_drosophila_toy_report,
    shape_drosophila_transmitter_matrix_report,
    toy_transmitter_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_drosophila_toy_graph_is_well_formed() -> None:
    graph = canonical_drosophila_toy_graph()
    node_ids = {node.node_id for node in graph.nodes}

    assert graph.graph_id == "drosophila_toy_substrate_v0"
    assert len(graph.nodes) == 8
    assert len(graph.edges) == 12
    assert set(graph.target_nodes) == {"descending_output", "motor_proxy"}
    assert set(graph.seed_activity) <= node_ids
    assert {edge.transmitter for edge in graph.edges} == {
        "acetylcholine",
        "dopamine",
        "gaba",
        "glutamate",
        "octopamine",
        "serotonin",
    }
    assert all(
        edge.source in node_ids and edge.target in node_ids for edge in graph.edges
    )


def test_edge_scaling_sweep_reports_target_off_target_and_washout() -> None:
    results = run_edge_scaling_sweep(coefficients=(0.0, 0.5, 1.0, 1.5, 2.0))
    by_coefficient = {result.coefficient: result for result in results}

    assert len(results) == 5
    assert all(result.scaled_edges == 4 for result in results)
    assert by_coefficient[1.0].target_movement == 0.0
    assert by_coefficient[1.0].off_target_movement == 0.0
    assert by_coefficient[1.0].washout == 0.0
    assert by_coefficient[2.0].target_movement > 0.0
    assert by_coefficient[0.0].target_movement < 0.0
    assert by_coefficient[2.0].off_target_movement >= 0.0
    assert by_coefficient[2.0].washout >= 0.0
    assert all(not result.instability_flag for result in results)


def test_source_class_gate_scales_local_matching_edges() -> None:
    results = run_edge_scaling_sweep(
        transmitter="acetylcholine",
        coefficients=(0.0, 1.0, 2.0),
        source_class="interneuron",
    )

    assert {result.scaled_edges for result in results} == {2}
    assert all(result.source_class == "interneuron" for result in results)
    assert results[1].target_movement == 0.0
    assert results[2].target_movement > 0.0


def test_drosophila_toy_report_and_markdown_shape() -> None:
    report = shape_drosophila_toy_report()
    markdown = render_drosophila_toy_markdown(report)

    assert report["experiment"] == "drosophila_toy_substrate_edge_scaling"
    assert "Toy-only graph perturbation fixture" in report["claim_boundary"]
    assert report["kickoff"] == (
        "docs/research/2026-06-03-drosophila-substrate-kickoff.md"
    )
    assert report["summary"]["runs"] == 5
    assert report["summary"]["target_nodes"] == [
        "descending_output",
        "motor_proxy",
    ]
    assert "Drosophila-Inspired Toy Substrate Sweep" in markdown
    assert "2026-06-03-drosophila-substrate-kickoff.md" in markdown
    assert "Target movement" in markdown
    assert "Washout" in markdown


def test_transmitter_matrix_has_expected_shape() -> None:
    coefficients = (0.0, 1.0, 2.0)
    results = run_transmitter_coefficient_matrix(coefficients=coefficients)
    transmitters = toy_transmitter_labels()

    assert len(results) == len(transmitters) * len(coefficients)
    assert {result.transmitter for result in results} == set(transmitters)
    assert {result.coefficient for result in results} == set(coefficients)
    assert all(result.graph_id == "drosophila_toy_substrate_v0" for result in results)

    by_transmitter = {
        transmitter: [result for result in results if result.transmitter == transmitter]
        for transmitter in transmitters
    }

    assert all(len(rows) == len(coefficients) for rows in by_transmitter.values())
    assert all(
        any(
            result.coefficient == 1.0 and result.target_movement == 0.0
            for result in rows
        )
        for rows in by_transmitter.values()
    )


def test_transmitter_matrix_report_and_rankings_are_rankable() -> None:
    results = run_transmitter_coefficient_matrix(coefficients=(0.0, 1.0, 2.0))
    report = shape_drosophila_transmitter_matrix_report(
        results,
        coefficients=(0.0, 1.0, 2.0),
    )
    markdown = render_drosophila_transmitter_matrix_markdown(report)
    rankings = rank_transmitter_matrix_results(results, limit=3)

    assert report["experiment"] == "drosophila_toy_substrate_transmitter_matrix"
    assert report["summary"]["matrix_shape"] == [6, 3]
    assert report["summary"]["runs"] == 18
    assert len(report["rankings"]["target"]) == 12
    assert len(report["by_transmitter"]) == 6
    assert (
        rankings["target"][0]["abs_target_movement"]
        >= rankings["target"][1]["abs_target_movement"]
    )
    assert rankings["washout"][0]["washout"] <= rankings["washout"][1]["washout"]
    assert report["rankings"]["target"][0]["coefficient"] != 1.0
    assert "Toy-only transmitter-by-coefficient" in report["claim_boundary"]
    assert "Drosophila-Inspired Toy Transmitter Matrix" in markdown
    assert "Top Target Movement" in markdown
    assert "Lowest Washout Rows" in markdown


def test_export_drosophila_toy_artifacts(tmp_path: Path) -> None:
    counts = export_drosophila_toy_artifacts(
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
        jsonl_output=tmp_path / "sweep.jsonl",
        coefficients=(0.5, 1.0, 1.5),
    )

    assert counts == {
        "results": 3,
        "markdown_reports": 1,
        "jsonl_results": 3,
    }
    records = read_jsonl(tmp_path / "sweep.jsonl")
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert len(records) == 3
    assert records[1]["coefficient"] == 1.0
    assert records[1]["target_movement"] == 0.0
    assert (tmp_path / "report.json").read_text(encoding="utf-8").startswith("{")
    assert "Toy-only graph perturbation fixture" in markdown


def test_export_drosophila_transmitter_matrix_artifacts(tmp_path: Path) -> None:
    counts = export_drosophila_transmitter_matrix_artifacts(
        json_report_output=tmp_path / "matrix.json",
        markdown_report_output=tmp_path / "matrix.md",
        jsonl_output=tmp_path / "matrix.jsonl",
        coefficients=(0.0, 1.0),
    )

    assert counts == {
        "results": 12,
        "markdown_reports": 1,
        "jsonl_results": 12,
    }
    records = read_jsonl(tmp_path / "matrix.jsonl")
    report_text = (tmp_path / "matrix.json").read_text(encoding="utf-8")
    markdown = (tmp_path / "matrix.md").read_text(encoding="utf-8")

    assert len(records) == 12
    assert {record["transmitter"] for record in records} == set(
        toy_transmitter_labels()
    )
    assert '"matrix_shape": [' in report_text
    assert "Per-Transmitter Summary" in markdown


def test_cli_writes_json_markdown_and_jsonl(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    json_report = tmp_path / "drosophila_toy_substrate_sweep.json"
    markdown_report = tmp_path / "drosophila_toy_substrate_sweep.md"
    jsonl_output = tmp_path / "drosophila_toy_substrate_sweep.jsonl"

    assert (
        cli_module.main(
            [
                "--json-report-output",
                str(json_report),
                "--markdown-report-output",
                str(markdown_report),
                "--jsonl-output",
                str(jsonl_output),
                "--transmitter",
                "acetylcholine",
                "--coefficient",
                "1.0",
                "--coefficient",
                "2.0",
            ]
        )
        == 0
    )

    records = read_jsonl(jsonl_output)

    assert len(records) == 2
    assert records[0]["coefficient"] == 1.0
    assert records[0]["target_movement"] == 0.0
    assert "Drosophila-Inspired Toy Substrate Sweep" in markdown_report.read_text(
        encoding="utf-8"
    )
    assert '"runs": 2' in json_report.read_text(encoding="utf-8")


def test_matrix_cli_writes_json_markdown_and_jsonl(tmp_path: Path) -> None:
    cli_module = _load_matrix_cli_module()
    json_report = tmp_path / "drosophila_transmitter_matrix.json"
    markdown_report = tmp_path / "drosophila_transmitter_matrix.md"
    jsonl_output = tmp_path / "drosophila_transmitter_matrix.jsonl"

    assert (
        cli_module.main(
            [
                "--json-report-output",
                str(json_report),
                "--markdown-report-output",
                str(markdown_report),
                "--jsonl-output",
                str(jsonl_output),
                "--transmitter",
                "acetylcholine",
                "--transmitter",
                "gaba",
                "--coefficient",
                "1.0",
                "--coefficient",
                "2.0",
            ]
        )
        == 0
    )

    records = read_jsonl(jsonl_output)
    markdown = markdown_report.read_text(encoding="utf-8")

    assert len(records) == 4
    assert {record["transmitter"] for record in records} == {
        "acetylcholine",
        "gaba",
    }
    assert '"matrix_shape": [' in json_report.read_text(encoding="utf-8")
    assert "Drosophila-Inspired Toy Transmitter Matrix" in markdown


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "export_drosophila_toy_fixture.py"
    spec = importlib.util.spec_from_file_location(
        "export_drosophila_toy_fixture",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_matrix_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "run_drosophila_transmitter_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "run_drosophila_transmitter_matrix",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
