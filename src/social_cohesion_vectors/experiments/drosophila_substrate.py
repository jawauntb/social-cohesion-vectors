"""Tiny Drosophila-inspired graph perturbation toy.

This module deliberately models a synthetic graph substrate, not fly biology.
Nodes, edges, transmitter labels, and perturbations are hand-authored fixtures
for testing graph-intervention reporting discipline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from social_cohesion_vectors.datasets import write_jsonl

TransmitterLabel = Literal[
    "acetylcholine",
    "gaba",
    "glutamate",
    "dopamine",
    "serotonin",
    "octopamine",
]

TOY_TRANSMITTER_LABELS: tuple[TransmitterLabel, ...] = (
    "acetylcholine",
    "gaba",
    "glutamate",
    "dopamine",
    "serotonin",
    "octopamine",
)


@dataclass(frozen=True)
class ToyNode:
    """One synthetic node in the Drosophila-inspired fixture graph."""

    node_id: str
    label: str
    node_class: str
    role: str


@dataclass(frozen=True)
class ToyEdge:
    """One directed weighted synthetic edge with a transmitter label."""

    source: str
    target: str
    transmitter: TransmitterLabel
    weight: float


@dataclass(frozen=True)
class ToyGraph:
    """Small directed graph plus declared target and off-target readouts."""

    graph_id: str
    description: str
    nodes: tuple[ToyNode, ...]
    edges: tuple[ToyEdge, ...]
    seed_activity: Mapping[str, float]
    target_nodes: tuple[str, ...]


@dataclass(frozen=True)
class EdgeScalingIntervention:
    """Deterministic toy operation for transmitter-class edge scaling."""

    intervention_id: str
    transmitter: TransmitterLabel
    coefficient: float
    source_class: str | None = None


@dataclass(frozen=True)
class PerturbationResult:
    """JSON-ready metrics from one toy edge-scaling run."""

    run_id: str
    graph_id: str
    intervention_id: str
    transmitter: TransmitterLabel
    coefficient: float
    source_class: str | None
    target_movement: float
    off_target_movement: float
    selectivity_ratio: float
    washout: float
    instability_flag: bool
    target_nodes: tuple[str, ...]
    scaled_edges: int
    baseline_target: float
    perturbed_target: float


def canonical_drosophila_toy_graph() -> ToyGraph:
    """Return the canonical synthetic graph fixture.

    The fixture borrows Drosophila-style vocabulary for node and transmitter
    labels, but the values are toy metadata with no organism-level claim.
    """

    return ToyGraph(
        graph_id="drosophila_toy_substrate_v0",
        description=(
            "Synthetic Drosophila-inspired directed graph for deterministic "
            "transmitter-label edge-scaling sweeps."
        ),
        nodes=(
            ToyNode("odor_input", "odor input proxy", "sensory", "input"),
            ToyNode("visual_input", "visual input proxy", "sensory", "input"),
            ToyNode("integration_hub", "integration hub", "interneuron", "hub"),
            ToyNode("memory_proxy", "memory proxy", "interneuron", "off_target"),
            ToyNode("state_gate", "state gate", "modulatory", "gate"),
            ToyNode("descending_output", "descending output", "output", "target"),
            ToyNode("motor_proxy", "motor proxy", "output", "target"),
            ToyNode("sleep_context", "sleep context", "context", "off_target"),
        ),
        edges=(
            ToyEdge("odor_input", "integration_hub", "acetylcholine", 0.62),
            ToyEdge("visual_input", "integration_hub", "acetylcholine", 0.38),
            ToyEdge("odor_input", "memory_proxy", "glutamate", 0.24),
            ToyEdge("integration_hub", "descending_output", "acetylcholine", 0.68),
            ToyEdge("integration_hub", "motor_proxy", "acetylcholine", 0.44),
            ToyEdge("integration_hub", "memory_proxy", "dopamine", 0.26),
            ToyEdge("state_gate", "integration_hub", "octopamine", 0.30),
            ToyEdge("state_gate", "sleep_context", "serotonin", 0.35),
            ToyEdge("sleep_context", "integration_hub", "gaba", -0.22),
            ToyEdge("memory_proxy", "descending_output", "gaba", -0.18),
            ToyEdge("memory_proxy", "sleep_context", "serotonin", 0.27),
            ToyEdge("descending_output", "motor_proxy", "glutamate", 0.16),
        ),
        seed_activity={
            "odor_input": 1.0,
            "visual_input": 0.55,
            "state_gate": 0.45,
        },
        target_nodes=("descending_output", "motor_proxy"),
    )


def default_coefficients() -> tuple[float, ...]:
    """Coefficient grid for the canonical toy sweep."""

    return (0.0, 0.5, 1.0, 1.5, 2.0)


def toy_transmitter_labels() -> tuple[TransmitterLabel, ...]:
    """Return every transmitter label used by the synthetic fixture."""

    return TOY_TRANSMITTER_LABELS


def run_edge_scaling_sweep(
    graph: ToyGraph | None = None,
    *,
    transmitter: TransmitterLabel = "acetylcholine",
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
    steps: int = 4,
    washout_steps: int = 3,
) -> list[PerturbationResult]:
    """Run a deterministic transmitter-class edge-scaling coefficient sweep."""

    graph = graph or canonical_drosophila_toy_graph()
    coefficients = coefficients or default_coefficients()
    _validate_graph(graph)
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if washout_steps < 1:
        raise ValueError("washout_steps must be at least 1.")

    return [
        run_edge_scaling_intervention(
            graph,
            EdgeScalingIntervention(
                intervention_id=_intervention_id(
                    transmitter,
                    coefficient,
                    source_class,
                ),
                transmitter=transmitter,
                coefficient=coefficient,
                source_class=source_class,
            ),
            steps=steps,
            washout_steps=washout_steps,
        )
        for coefficient in coefficients
    ]


def run_transmitter_coefficient_matrix(
    graph: ToyGraph | None = None,
    *,
    transmitters: Sequence[TransmitterLabel] | None = None,
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
    steps: int = 4,
    washout_steps: int = 3,
) -> list[PerturbationResult]:
    """Run all transmitter-by-coefficient toy edge-scaling combinations."""

    graph = graph or canonical_drosophila_toy_graph()
    transmitters = tuple(transmitters or toy_transmitter_labels())
    coefficients = tuple(coefficients or default_coefficients())

    return [
        result
        for transmitter in transmitters
        for result in run_edge_scaling_sweep(
            graph,
            transmitter=transmitter,
            coefficients=coefficients,
            source_class=source_class,
            steps=steps,
            washout_steps=washout_steps,
        )
    ]


def run_edge_scaling_intervention(
    graph: ToyGraph,
    intervention: EdgeScalingIntervention,
    *,
    steps: int = 4,
    washout_steps: int = 3,
) -> PerturbationResult:
    """Run one deterministic edge-scaling intervention and report metrics."""

    _validate_graph(graph)
    baseline_at_window = propagate_activity(
        graph,
        graph.seed_activity,
        steps=steps,
    )
    scaled_edges = _scale_edges(graph, intervention)
    perturbed = propagate_activity(
        graph,
        graph.seed_activity,
        edges=scaled_edges,
        steps=steps,
    )
    baseline_after_washout = propagate_activity(
        graph,
        graph.seed_activity,
        steps=steps + washout_steps,
    )
    washout_state = propagate_activity(
        graph,
        perturbed,
        steps=washout_steps,
    )

    target_nodes = graph.target_nodes
    off_target_nodes = tuple(
        node.node_id for node in graph.nodes if node.node_id not in target_nodes
    )
    baseline_target = _mean_state(baseline_at_window, target_nodes)
    perturbed_target = _mean_state(perturbed, target_nodes)
    target_movement = perturbed_target - baseline_target
    off_target_movement = _mean_abs_delta(
        perturbed,
        baseline_at_window,
        off_target_nodes,
    )
    washout = _mean_abs_delta(washout_state, baseline_after_washout, _node_ids(graph))
    selectivity_ratio = abs(target_movement) / max(off_target_movement, 1e-9)
    matching_edges = _count_matching_edges(graph, intervention)

    return PerturbationResult(
        run_id=f"{graph.graph_id}::{intervention.intervention_id}",
        graph_id=graph.graph_id,
        intervention_id=intervention.intervention_id,
        transmitter=intervention.transmitter,
        coefficient=round(intervention.coefficient, 6),
        source_class=intervention.source_class,
        target_movement=round(target_movement, 6),
        off_target_movement=round(off_target_movement, 6),
        selectivity_ratio=round(selectivity_ratio, 6),
        washout=round(washout, 6),
        instability_flag=_has_instability(perturbed),
        target_nodes=target_nodes,
        scaled_edges=matching_edges,
        baseline_target=round(baseline_target, 6),
        perturbed_target=round(perturbed_target, 6),
    )


def propagate_activity(
    graph: ToyGraph,
    initial_activity: Mapping[str, float],
    *,
    edges: Sequence[ToyEdge] | None = None,
    steps: int = 4,
    retention: float = 0.28,
    input_gain: float = 0.72,
) -> dict[str, float]:
    """Propagate toy activity through signed weighted edges."""

    if steps < 0:
        raise ValueError("steps cannot be negative.")
    node_ids = _node_ids(graph)
    state = {node_id: float(initial_activity.get(node_id, 0.0)) for node_id in node_ids}
    active_edges = tuple(edges or graph.edges)

    for _ in range(steps):
        incoming = {node_id: 0.0 for node_id in node_ids}
        for edge in active_edges:
            incoming[edge.target] += state[edge.source] * edge.weight
        state = {
            node_id: _squash(
                retention * state[node_id] + input_gain * incoming[node_id]
            )
            for node_id in node_ids
        }
    return state


def shape_drosophila_toy_report(
    results: Sequence[PerturbationResult] | None = None,
    *,
    graph: ToyGraph | None = None,
    transmitter: TransmitterLabel = "acetylcholine",
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report for the toy perturbation sweep."""

    graph = graph or canonical_drosophila_toy_graph()
    result_list = list(
        results
        or run_edge_scaling_sweep(
            graph,
            transmitter=transmitter,
            coefficients=coefficients,
            source_class=source_class,
        )
    )
    non_baseline = [result for result in result_list if result.coefficient != 1.0]
    best = max(
        non_baseline or result_list,
        key=lambda item: (abs(item.target_movement), item.selectivity_ratio),
    )
    return {
        "experiment": "drosophila_toy_substrate_edge_scaling",
        "claim_boundary": (
            "Toy-only graph perturbation fixture. It does not model real "
            "Drosophila biology, behavior, pharmacology, or social effects."
        ),
        "kickoff": "docs/research/2026-06-03-drosophila-substrate-kickoff.md",
        "graph": _graph_record(graph),
        "summary": {
            "graph_id": graph.graph_id,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "target_nodes": list(graph.target_nodes),
            "transmitter": transmitter,
            "source_class": source_class,
            "runs": len(result_list),
            "unstable_runs": sum(
                1 for result in result_list if result.instability_flag
            ),
            "max_abs_target_movement": round(
                max(abs(result.target_movement) for result in result_list),
                6,
            ),
            "max_off_target_movement": round(
                max(result.off_target_movement for result in result_list),
                6,
            ),
            "min_washout": round(min(result.washout for result in result_list), 6),
            "best_run_id": best.run_id,
            "best_coefficient": best.coefficient,
        },
        "results": [_result_record(result) for result in result_list],
    }


def shape_drosophila_transmitter_matrix_report(
    results: Sequence[PerturbationResult] | None = None,
    *,
    graph: ToyGraph | None = None,
    transmitters: Sequence[TransmitterLabel] | None = None,
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report for the transmitter-by-coefficient matrix."""

    graph = graph or canonical_drosophila_toy_graph()
    transmitter_list = tuple(transmitters or toy_transmitter_labels())
    coefficient_list = tuple(coefficients or default_coefficients())
    result_list = list(
        results
        or run_transmitter_coefficient_matrix(
            graph,
            transmitters=transmitter_list,
            coefficients=coefficient_list,
            source_class=source_class,
        )
    )
    rankings = rank_transmitter_matrix_results(result_list)
    by_transmitter = _summarize_by_transmitter(result_list)

    return {
        "experiment": "drosophila_toy_substrate_transmitter_matrix",
        "claim_boundary": (
            "Toy-only transmitter-by-coefficient graph perturbation matrix. "
            "It does not model real Drosophila biology, behavior, "
            "pharmacology, neural mechanisms, or social effects."
        ),
        "kickoff": "docs/research/2026-06-03-drosophila-substrate-kickoff.md",
        "graph": _graph_record(graph),
        "summary": {
            "graph_id": graph.graph_id,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "target_nodes": list(graph.target_nodes),
            "transmitters": list(transmitter_list),
            "coefficients": [round(coefficient, 6) for coefficient in coefficient_list],
            "source_class": source_class,
            "runs": len(result_list),
            "matrix_shape": [len(transmitter_list), len(coefficient_list)],
            "unstable_runs": sum(
                1 for result in result_list if result.instability_flag
            ),
            "top_target_run_id": _first_run_id(rankings["target"]),
            "top_selective_run_id": _first_run_id(rankings["selective"]),
            "top_washout_run_id": _first_run_id(rankings["washout"]),
        },
        "rankings": rankings,
        "by_transmitter": by_transmitter,
        "results": [_result_record(result) for result in result_list],
    }


def rank_transmitter_matrix_results(
    results: Sequence[PerturbationResult],
    *,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Rank matrix rows by target, selectivity, and washout summaries."""

    candidates = [result for result in results if result.coefficient != 1.0]
    candidates = candidates or list(results)
    target = sorted(
        candidates,
        key=lambda item: (
            abs(item.target_movement),
            item.selectivity_ratio,
            -item.off_target_movement,
            -item.washout,
        ),
        reverse=True,
    )
    selective = sorted(
        candidates,
        key=lambda item: (
            item.selectivity_ratio,
            abs(item.target_movement),
            -item.off_target_movement,
            -item.washout,
        ),
        reverse=True,
    )
    washout = sorted(
        candidates,
        key=lambda item: (
            item.washout,
            item.off_target_movement,
            -abs(item.target_movement),
        ),
    )
    return {
        "target": [_ranking_record(result) for result in _limit(target, limit)],
        "selective": [_ranking_record(result) for result in _limit(selective, limit)],
        "washout": [_ranking_record(result) for result in _limit(washout, limit)],
    }


def render_drosophila_toy_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise markdown report for the toy sweep."""

    summary = _mapping(report.get("summary"))
    graph = _mapping(report.get("graph"))
    lines = [
        "# Drosophila-Inspired Toy Substrate Sweep",
        "",
        str(report.get("claim_boundary", "")),
        "",
        "Related kickoff: "
        "[2026-06-03 Drosophila Substrate Kickoff]"
        "(2026-06-03-drosophila-substrate-kickoff.md).",
        "",
        "## Summary",
        "",
        f"- Graph: {summary.get('graph_id', '')}",
        f"- Nodes: {int(summary.get('nodes', 0))}",
        f"- Edges: {int(summary.get('edges', 0))}",
        f"- Target nodes: {', '.join(_strings(summary.get('target_nodes')))}",
        f"- Transmitter sweep: {summary.get('transmitter', '')}",
        f"- Runs: {int(summary.get('runs', 0))}",
        f"- Unstable runs: {int(summary.get('unstable_runs', 0))}",
        f"- Max absolute target movement: "
        f"{float(summary.get('max_abs_target_movement', 0.0)):.3f}",
        f"- Max off-target movement: "
        f"{float(summary.get('max_off_target_movement', 0.0)):.3f}",
        f"- Minimum washout: {float(summary.get('min_washout', 0.0)):.3f}",
        f"- Best coefficient: {float(summary.get('best_coefficient', 0.0)):.3f}",
        "",
        "## Nodes",
        "",
        "| Node | Class | Role |",
        "| --- | --- | --- |",
    ]
    for node in _sequence(graph.get("nodes")):
        node_map = _mapping(node)
        lines.append(
            "| "
            f"{node_map.get('node_id', '')} | "
            f"{node_map.get('node_class', '')} | "
            f"{node_map.get('role', '')} |"
        )
    lines.extend(
        [
            "",
            "## Sweep Results",
            "",
            "| Coefficient | Scaled edges | Target movement | Off-target movement | "
            "Selectivity | Washout | Instability |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in _sequence(report.get("results")):
        result_map = _mapping(result)
        lines.append(
            "| "
            f"{float(result_map.get('coefficient', 0.0)):.3f} | "
            f"{int(result_map.get('scaled_edges', 0))} | "
            f"{float(result_map.get('target_movement', 0.0)):+.3f} | "
            f"{float(result_map.get('off_target_movement', 0.0)):.3f} | "
            f"{float(result_map.get('selectivity_ratio', 0.0)):.3f} | "
            f"{float(result_map.get('washout', 0.0)):.3f} | "
            f"{bool(result_map.get('instability_flag', False))} |"
        )
    return "\n".join(lines) + "\n"


def render_drosophila_transmitter_matrix_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise markdown report for the toy transmitter matrix."""

    summary = _mapping(report.get("summary"))
    rankings = _mapping(report.get("rankings"))
    lines = [
        "# Drosophila-Inspired Toy Transmitter Matrix",
        "",
        str(report.get("claim_boundary", "")),
        "",
        "Related kickoff: "
        "[2026-06-03 Drosophila Substrate Kickoff]"
        "(2026-06-03-drosophila-substrate-kickoff.md).",
        "",
        "## Summary",
        "",
        f"- Graph: {summary.get('graph_id', '')}",
        f"- Matrix shape: {_format_shape(summary.get('matrix_shape'))}",
        f"- Runs: {int(summary.get('runs', 0))}",
        f"- Transmitters: {', '.join(_strings(summary.get('transmitters')))}",
        f"- Coefficients: {', '.join(_format_floats(summary.get('coefficients')))}",
        f"- Target nodes: {', '.join(_strings(summary.get('target_nodes')))}",
        f"- Unstable runs: {int(summary.get('unstable_runs', 0))}",
        "",
        "## Top Target Movement",
        "",
        *_ranking_table(_sequence(rankings.get("target"))),
        "",
        "## Top Selective Rows",
        "",
        *_ranking_table(_sequence(rankings.get("selective"))),
        "",
        "## Lowest Washout Rows",
        "",
        *_ranking_table(_sequence(rankings.get("washout"))),
        "",
        "## Per-Transmitter Summary",
        "",
        "| Transmitter | Scaled edges | Best coefficient | Target movement | "
        "Off-target movement | Washout |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(report.get("by_transmitter")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('transmitter', '')} | "
            f"{int(row_map.get('scaled_edges', 0))} | "
            f"{float(row_map.get('best_coefficient', 0.0)):.3f} | "
            f"{float(row_map.get('best_abs_target_movement', 0.0)):.3f} | "
            f"{float(row_map.get('off_target_at_best', 0.0)):.3f} | "
            f"{float(row_map.get('washout_at_best', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def export_drosophila_toy_artifacts(
    *,
    json_report_output: str | Path,
    markdown_report_output: str | Path | None = None,
    jsonl_output: str | Path | None = None,
    graph: ToyGraph | None = None,
    transmitter: TransmitterLabel = "acetylcholine",
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
) -> dict[str, int]:
    """Write JSON, Markdown, and optional JSONL artifacts for the toy sweep."""

    graph = graph or canonical_drosophila_toy_graph()
    results = run_edge_scaling_sweep(
        graph,
        transmitter=transmitter,
        coefficients=coefficients,
        source_class=source_class,
    )
    report = shape_drosophila_toy_report(
        results,
        graph=graph,
        transmitter=transmitter,
        source_class=source_class,
    )
    _write_json(report, json_report_output)
    counts = {"results": len(results)}
    if markdown_report_output is not None:
        output = Path(markdown_report_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_drosophila_toy_markdown(report), encoding="utf-8")
        counts["markdown_reports"] = 1
    if jsonl_output is not None:
        counts["jsonl_results"] = write_jsonl(
            (_result_record(result) for result in results),
            jsonl_output,
        )
    return counts


def export_drosophila_transmitter_matrix_artifacts(
    *,
    json_report_output: str | Path,
    markdown_report_output: str | Path | None = None,
    jsonl_output: str | Path | None = None,
    graph: ToyGraph | None = None,
    transmitters: Sequence[TransmitterLabel] | None = None,
    coefficients: Sequence[float] | None = None,
    source_class: str | None = None,
) -> dict[str, int]:
    """Write JSON, Markdown, and optional JSONL artifacts for the matrix."""

    graph = graph or canonical_drosophila_toy_graph()
    results = run_transmitter_coefficient_matrix(
        graph,
        transmitters=transmitters,
        coefficients=coefficients,
        source_class=source_class,
    )
    report = shape_drosophila_transmitter_matrix_report(
        results,
        graph=graph,
        transmitters=transmitters,
        coefficients=coefficients,
        source_class=source_class,
    )
    _write_json(report, json_report_output)
    counts = {"results": len(results)}
    if markdown_report_output is not None:
        output = Path(markdown_report_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_drosophila_transmitter_matrix_markdown(report),
            encoding="utf-8",
        )
        counts["markdown_reports"] = 1
    if jsonl_output is not None:
        counts["jsonl_results"] = write_jsonl(
            (_result_record(result) for result in results),
            jsonl_output,
        )
    return counts


def _scale_edges(
    graph: ToyGraph,
    intervention: EdgeScalingIntervention,
) -> tuple[ToyEdge, ...]:
    node_classes = {node.node_id: node.node_class for node in graph.nodes}
    scaled: list[ToyEdge] = []
    for edge in graph.edges:
        if _edge_matches(edge, intervention, node_classes):
            scaled.append(
                ToyEdge(
                    edge.source,
                    edge.target,
                    edge.transmitter,
                    edge.weight * intervention.coefficient,
                )
            )
        else:
            scaled.append(edge)
    return tuple(scaled)


def _edge_matches(
    edge: ToyEdge,
    intervention: EdgeScalingIntervention,
    node_classes: Mapping[str, str],
) -> bool:
    return edge.transmitter == intervention.transmitter and (
        intervention.source_class is None
        or node_classes[edge.source] == intervention.source_class
    )


def _count_matching_edges(
    graph: ToyGraph,
    intervention: EdgeScalingIntervention,
) -> int:
    node_classes = {node.node_id: node.node_class for node in graph.nodes}
    return sum(
        1 for edge in graph.edges if _edge_matches(edge, intervention, node_classes)
    )


def _intervention_id(
    transmitter: TransmitterLabel,
    coefficient: float,
    source_class: str | None,
) -> str:
    scope = source_class or "global"
    coefficient_label = str(coefficient).replace(".", "p").replace("-", "neg")
    return f"{scope}_{transmitter}_x{coefficient_label}"


def _validate_graph(graph: ToyGraph) -> None:
    node_ids = _node_ids(graph)
    if not graph.nodes:
        raise ValueError("Toy graph must contain at least one node.")
    if not graph.edges:
        raise ValueError("Toy graph must contain at least one edge.")
    missing_targets = set(graph.target_nodes) - node_ids
    if missing_targets:
        raise ValueError(f"Unknown target nodes: {sorted(missing_targets)}")
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            raise ValueError(f"Edge references unknown node: {edge}")


def _node_ids(graph: ToyGraph) -> set[str]:
    return {node.node_id for node in graph.nodes}


def _mean_state(state: Mapping[str, float], nodes: Iterable[str]) -> float:
    values = [state[node] for node in nodes]
    return sum(values) / len(values) if values else 0.0


def _mean_abs_delta(
    left: Mapping[str, float],
    right: Mapping[str, float],
    nodes: Iterable[str],
) -> float:
    deltas = [abs(left[node] - right[node]) for node in nodes]
    return sum(deltas) / len(deltas) if deltas else 0.0


def _squash(value: float) -> float:
    return max(-1.5, min(1.5, value))


def _has_instability(state: Mapping[str, float]) -> bool:
    return any(abs(value) >= 1.45 for value in state.values())


def _graph_record(graph: ToyGraph) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "description": graph.description,
        "target_nodes": list(graph.target_nodes),
        "seed_activity": dict(graph.seed_activity),
        "nodes": [
            {
                "node_id": node.node_id,
                "label": node.label,
                "node_class": node.node_class,
                "role": node.role,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "transmitter": edge.transmitter,
                "weight": edge.weight,
            }
            for edge in graph.edges
        ],
    }


def _result_record(result: PerturbationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "graph_id": result.graph_id,
        "intervention_id": result.intervention_id,
        "transmitter": result.transmitter,
        "coefficient": result.coefficient,
        "source_class": result.source_class,
        "target_movement": result.target_movement,
        "off_target_movement": result.off_target_movement,
        "selectivity_ratio": result.selectivity_ratio,
        "washout": result.washout,
        "instability_flag": result.instability_flag,
        "target_nodes": list(result.target_nodes),
        "scaled_edges": result.scaled_edges,
        "baseline_target": result.baseline_target,
        "perturbed_target": result.perturbed_target,
    }


def _ranking_record(result: PerturbationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "transmitter": result.transmitter,
        "coefficient": result.coefficient,
        "scaled_edges": result.scaled_edges,
        "target_movement": result.target_movement,
        "abs_target_movement": round(abs(result.target_movement), 6),
        "off_target_movement": result.off_target_movement,
        "selectivity_ratio": result.selectivity_ratio,
        "washout": result.washout,
        "instability_flag": result.instability_flag,
    }


def _summarize_by_transmitter(
    results: Sequence[PerturbationResult],
) -> list[dict[str, Any]]:
    transmitters = sorted({result.transmitter for result in results})
    rows: list[dict[str, Any]] = []
    for transmitter in transmitters:
        transmitter_results = [
            result for result in results if result.transmitter == transmitter
        ]
        non_baseline = [
            result for result in transmitter_results if result.coefficient != 1.0
        ]
        best = max(
            non_baseline or transmitter_results,
            key=lambda item: (
                abs(item.target_movement),
                item.selectivity_ratio,
                -item.off_target_movement,
            ),
        )
        rows.append(
            {
                "transmitter": transmitter,
                "runs": len(transmitter_results),
                "scaled_edges": best.scaled_edges,
                "best_run_id": best.run_id,
                "best_coefficient": best.coefficient,
                "best_abs_target_movement": round(abs(best.target_movement), 6),
                "target_movement_at_best": best.target_movement,
                "off_target_at_best": best.off_target_movement,
                "washout_at_best": best.washout,
                "selectivity_at_best": best.selectivity_ratio,
            }
        )
    return rows


def _limit(
    results: Sequence[PerturbationResult],
    limit: int | None,
) -> Sequence[PerturbationResult]:
    return results if limit is None else results[:limit]


def _first_run_id(records: Sequence[Mapping[str, Any]]) -> str | None:
    return str(records[0]["run_id"]) if records else None


def _ranking_table(rows: Sequence[object]) -> list[str]:
    lines = [
        "| Transmitter | Coefficient | Target movement | Off-target movement | "
        "Selectivity | Washout |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:5]:
        row_map = _mapping(row)
        lines.append(
            "| "
            f"{row_map.get('transmitter', '')} | "
            f"{float(row_map.get('coefficient', 0.0)):.3f} | "
            f"{float(row_map.get('target_movement', 0.0)):+.3f} | "
            f"{float(row_map.get('off_target_movement', 0.0)):.3f} | "
            f"{float(row_map.get('selectivity_ratio', 0.0)):.3f} | "
            f"{float(row_map.get('washout', 0.0)):.3f} |"
        )
    return lines


def _format_shape(value: object) -> str:
    shape = _sequence(value)
    if len(shape) != 2:
        return ""
    return f"{_to_int(shape[0])} x {_to_int(shape[1])}"


def _format_floats(value: object) -> list[str]:
    return [f"{_to_float(item):.3f}" for item in _sequence(value)]


def _to_int(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


def _to_float(value: object) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _strings(value: object) -> list[str]:
    return [str(item) for item in _sequence(value)]
