"""Source-diversity gates for generated pseudo-cohesion benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.schemas import PairwiseExample


def run_source_diversity_audit_from_file(
    pairs_path: str | Path,
    *,
    source_metadata_key: str = "source",
    group_metadata_key: str = "primary_fault_class",
    min_sources: int = 2,
    min_pairs_per_source: int = 2,
    min_groups_per_source: int = 2,
    min_shared_groups: int = 2,
) -> dict[str, Any]:
    """Load pairwise examples and audit source/fault coverage."""

    return run_source_diversity_audit(
        pairs=load_pairwise_examples_jsonl(pairs_path),
        source_metadata_key=source_metadata_key,
        group_metadata_key=group_metadata_key,
        min_sources=min_sources,
        min_pairs_per_source=min_pairs_per_source,
        min_groups_per_source=min_groups_per_source,
        min_shared_groups=min_shared_groups,
        input_paths={"pairs": str(pairs_path)},
    )


def run_source_diversity_audit(
    *,
    pairs: Sequence[PairwiseExample],
    source_metadata_key: str = "source",
    group_metadata_key: str = "primary_fault_class",
    min_sources: int = 2,
    min_pairs_per_source: int = 2,
    min_groups_per_source: int = 2,
    min_shared_groups: int = 2,
    input_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Audit whether source-held-out claims have enough source coverage."""

    source_counts = _value_counts(pairs, source_metadata_key)
    groups_by_source = _groups_by_source(
        pairs,
        source_metadata_key=source_metadata_key,
        group_metadata_key=group_metadata_key,
    )
    shared_groups = _shared_groups(groups_by_source)
    failed_pair_sources = sorted(
        source
        for source, count in source_counts.items()
        if count < min_pairs_per_source
    )
    failed_group_sources = sorted(
        source
        for source, groups in groups_by_source.items()
        if len(groups) < min_groups_per_source
    )
    missing_source_pairs = _missing_metadata_pair_count(pairs, source_metadata_key)
    missing_group_pairs = _missing_metadata_pair_count(pairs, group_metadata_key)
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(len(pairs)),
            "threshold": 1.0,
            "passed": len(pairs) > 0,
        },
        {
            "gate_id": "source_count",
            "value": float(len(source_counts)),
            "threshold": float(min_sources),
            "passed": len(source_counts) >= min_sources,
        },
        {
            "gate_id": "min_pairs_per_source",
            "value": float(min(source_counts.values(), default=0)),
            "threshold": float(min_pairs_per_source),
            "passed": not failed_pair_sources,
        },
        {
            "gate_id": "min_groups_per_source",
            "value": float(
                min((len(groups) for groups in groups_by_source.values()), default=0)
            ),
            "threshold": float(min_groups_per_source),
            "passed": not failed_group_sources,
        },
        {
            "gate_id": "shared_group_count",
            "value": float(len(shared_groups)),
            "threshold": float(min_shared_groups),
            "passed": len(shared_groups) >= min_shared_groups,
        },
        {
            "gate_id": "missing_source_pairs",
            "value": float(missing_source_pairs),
            "threshold": 0.0,
            "passed": missing_source_pairs == 0,
        },
        {
            "gate_id": "missing_group_pairs",
            "value": float(missing_group_pairs),
            "threshold": 0.0,
            "passed": missing_group_pairs == 0,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "experiment": "source_diversity_audit",
        "description": (
            "Audits whether generated benchmark pairs have enough independent "
            "source coverage for source-held-out hard-negative claims."
        ),
        "inputs": {
            "paths": dict(input_paths or {}),
            "pairs": len(pairs),
            "source_metadata_key": source_metadata_key,
            "group_metadata_key": group_metadata_key,
            "min_sources": min_sources,
            "min_pairs_per_source": min_pairs_per_source,
            "min_groups_per_source": min_groups_per_source,
            "min_shared_groups": min_shared_groups,
        },
        "summary": {
            "status": "source_diversity_ready" if ready else "not_ready",
            "ready_for_activation": ready,
            "activation_readiness": (
                "source_diversity_ready" if ready else "not_ready"
            ),
            "pairs": len(pairs),
            "sources": len(source_counts),
            "groups": len(_metadata_values(pairs, group_metadata_key)),
            "shared_groups": len(shared_groups),
            "missing_source_pairs": missing_source_pairs,
            "missing_group_pairs": missing_group_pairs,
            "failed_pair_sources": failed_pair_sources,
            "failed_group_sources": failed_group_sources,
            "gates": gates,
        },
        "source_counts": source_counts,
        "groups_by_source": {
            source: sorted(groups)
            for source, groups in sorted(groups_by_source.items())
        },
        "shared_groups": shared_groups,
    }


def save_source_diversity_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write source-diversity audit artifacts."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_source_diversity_markdown(report),
        encoding="utf-8",
    )


def render_source_diversity_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise source-diversity audit report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Source Diversity Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status', 'not_ready')}`",
        f"- Ready for activation claims: "
        f"{bool(summary.get('ready_for_activation', False))}",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Fault groups: {int(summary.get('groups', 0))}",
        f"- Shared fault groups: {int(summary.get('shared_groups', 0))}",
        f"- Missing source pairs: {int(summary.get('missing_source_pairs', 0))}",
        f"- Missing group pairs: {int(summary.get('missing_group_pairs', 0))}",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for raw_gate in _sequence(summary.get("gates")):
        gate = _mapping(raw_gate)
        lines.append(
            "| "
            f"{gate.get('gate_id', '')} | "
            f"{float(gate.get('value', 0.0)):.3f} | "
            f"{float(gate.get('threshold', 0.0)):.3f} | "
            f"{bool(gate.get('passed', False))} |"
        )
    lines.extend(
        [
            "",
            "## Source Coverage",
            "",
            "| Source | Pairs | Fault groups |",
            "| --- | ---: | ---: |",
        ]
    )
    source_counts = _mapping(report.get("source_counts"))
    groups_by_source = _mapping(report.get("groups_by_source"))
    for source, count in sorted(source_counts.items()):
        groups = _sequence(groups_by_source.get(source))
        lines.append(f"| `{source}` | {int(count)} | {len(groups)} |")
    shared_groups = _sequence(report.get("shared_groups"))
    if shared_groups:
        lines.extend(
            [
                "",
                "## Shared Fault Groups",
                "",
                ", ".join(str(group) for group in shared_groups),
            ]
        )
    return "\n".join(lines) + "\n"


def _value_counts(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        value = str(pair.metadata.get(metadata_key, "")).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _groups_by_source(
    pairs: Sequence[PairwiseExample],
    *,
    source_metadata_key: str,
    group_metadata_key: str,
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for pair in pairs:
        source = str(pair.metadata.get(source_metadata_key, "")).strip()
        group = str(pair.metadata.get(group_metadata_key, "")).strip()
        if not source or not group:
            continue
        groups.setdefault(source, set()).add(group)
    return groups


def _shared_groups(groups_by_source: Mapping[str, set[str]]) -> list[str]:
    if not groups_by_source:
        return []
    shared = set.intersection(*groups_by_source.values())
    return sorted(shared)


def _metadata_values(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> set[str]:
    return {
        str(pair.metadata.get(metadata_key, "")).strip()
        for pair in pairs
        if str(pair.metadata.get(metadata_key, "")).strip()
    }


def _missing_metadata_pair_count(
    pairs: Sequence[PairwiseExample],
    metadata_key: str,
) -> int:
    return sum(
        1 for pair in pairs if not str(pair.metadata.get(metadata_key, "")).strip()
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
