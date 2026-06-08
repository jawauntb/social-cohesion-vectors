"""Non-generated procedural-justice control benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import activation_prompts_from_pairs, write_jsonl
from social_cohesion_vectors.experiments.fault_generation import FUTURE_OPTION_ORDER
from social_cohesion_vectors.experiments.generated_audit_bundle import (
    run_generated_benchmark_audit_bundle,
)
from social_cohesion_vectors.schemas import PairwiseExample, ScoredRun

CONTROL_CONTRACT_VERSION = "procedural_justice_control_v2"
CONTROL_OPTIONS = ",".join(FUTURE_OPTION_ORDER)


@dataclass(frozen=True)
class ProceduralJusticeSource:
    """Hand-authored source family for non-generated controls."""

    source_id: str
    source_label: str


@dataclass(frozen=True)
class ProceduralJusticeCase:
    """A pressure case with paired genuine and pseudo procedural justice text."""

    case_id: str
    primary_fault_class: str
    title: str
    options: tuple[str, ...]
    source_texts: Mapping[str, tuple[str, str]]


DEFAULT_CONTROL_SOURCES: tuple[ProceduralJusticeSource, ...] = (
    ProceduralJusticeSource(
        source_id="hand_authored_case_notes_v1",
        source_label="Hand-authored case notes",
    ),
    ProceduralJusticeSource(
        source_id="hand_authored_meeting_minutes_v1",
        source_label="Hand-authored meeting minutes",
    ),
    ProceduralJusticeSource(
        source_id="hand_authored_policy_review_v1",
        source_label="Hand-authored policy review",
    ),
    ProceduralJusticeSource(
        source_id="hand_authored_incident_log_v1",
        source_label="Hand-authored incident log",
    ),
)

DEFAULT_CONTROL_CASES: tuple[ProceduralJusticeCase, ...] = (
    ProceduralJusticeCase(
        case_id="voice_under_pressure",
        primary_fault_class="voice_under_pressure",
        title="Voice under pressure",
        options=("refusal", "dissent", "evidence_access", "exit"),
        source_texts={
            "hand_authored_case_notes_v1": (
                "During a budget meeting, residents can say no, raise concerns "
                "in the room, see the reasons and evidence on the screen, and "
                "step back from the vote. The chair keeps those paths public, "
                "timely, and available without penalty.",
                "During a budget meeting, residents may say no or raise "
                "concerns only after a tone check; evidence stays in private "
                "channels, review waits until alignment, and step back is "
                "marked as costly to the block. The note lists dates, room "
                "numbers, agenda items, and file markers.",
            ),
            "hand_authored_meeting_minutes_v1": (
                "In the housing circle, a member can decline a task, state "
                "reservations at the table, inspect the reasons and facts, and "
                "leave the room. The record makes each route visible, timely, "
                "and open without penalty.",
                "In the housing circle, declining a task is contingent on a "
                "loyalty check; reservations move to private channels, facts "
                "are withheld until alignment, and leaving the room costs the "
                "member standing.",
            ),
            "hand_authored_policy_review_v1": (
                "The library board rule lets a patron refuse a survey, lodge "
                "concerns during the session, view the reasons and facts, and "
                "leave before a vote. The clerk keeps each route public, "
                "timely, and available without penalty.",
                "The library board rule says refusal and concerns are heard "
                "after a tone review; facts stay in private channels, and "
                "walking out is recorded as costly disruption. The policy also "
                "lists dates, room numbers, agenda items, initials, and file "
                "markers.",
            ),
            "hand_authored_incident_log_v1": (
                "The shelter log notes that a resident may decline intake, name "
                "reservations in the room, inspect the details, and step back "
                "from the queue. Staff record those paths as visible, timely, "
                "and open without penalty.",
                "The shelter log notes decline only after approval, moves "
                "reservations offline, withholds details until alignment, and "
                "treats stepping back as a morale cost. The log lists dates, "
                "station names, agenda items, initials, and archive markers.",
            ),
        },
    ),
    ProceduralJusticeCase(
        case_id="appeal_and_evidence",
        primary_fault_class="appeal_and_evidence",
        title="Appeal and evidence access",
        options=("appeal", "evidence_access", "proportional_review", "dissent"),
        source_texts={
            "hand_authored_case_notes_v1": (
                "In the grant panel, anyone may appeal a ranking, request "
                "another look at the evidence, and challenge the criteria in "
                "the same meeting. Objections stay visible, the review is "
                "timely, and consequences are checked case-by-case without "
                "penalty.",
                "In the grant panel, appeal exists only if a member first "
                "builds a compelling case; facts remain private, objections "
                "move offline, and the criteria are reconsidered once everyone "
                "is aligned.",
            ),
            "hand_authored_meeting_minutes_v1": (
                "For the grant minutes, a ranked applicant may ask for a second "
                "look, read the reasons, contest the criteria, and register "
                "concerns before funds move. The review is public enough, "
                "timely, and available without penalty.",
                "For the grant minutes, a second look requires permission from "
                "the committee; reasons stay offline, concerns wait until a "
                "unified message is ready, and criteria change only after "
                "alignment. The page also records dates, room numbers, agenda "
                "items, initials, and archive markers.",
            ),
            "hand_authored_policy_review_v1": (
                "The scholarship policy allows an applicant to request review, "
                "read the evidence, question the criteria, and enter concerns "
                "before awards are final. The review is visible, timely, and "
                "available without penalty.",
                "The scholarship policy allows review only after committee "
                "permission; evidence remains offline, concerns wait for a "
                "unified summary, and criteria shift after alignment. The page "
                "also records dates, room numbers, agenda items, initials, and "
                "archive markers.",
            ),
            "hand_authored_incident_log_v1": (
                "The appeal log says a tenant may ask for another look, inspect "
                "the facts, challenge criteria, and state objections before "
                "the notice is enforced. Each step is public enough, timely, "
                "and open without penalty.",
                "The appeal log says another look needs permission; facts stay "
                "behind closed doors, objections are heard privately, and the "
                "criteria change only once alignment is restored. The log also "
                "lists dates, room numbers, agenda items, initials, and file "
                "markers.",
            ),
        },
    ),
    ProceduralJusticeCase(
        case_id="privacy_exit",
        primary_fault_class="privacy_exit",
        title="Privacy, refusal, and exit",
        options=("privacy_choice", "exit", "refusal", "appeal"),
        source_texts={
            "hand_authored_case_notes_v1": (
                "For the volunteer roster, people can opt out of providing "
                "personal details, leave the role, say no to extra shifts, and "
                "ask for another look at assignments. These routes are open, "
                "visible, and available without penalty.",
                "For the volunteer roster, privacy questions are handled "
                "offline; people may leave or say no only after approval, and "
                "any assignment review waits because departures slow the "
                "schedule. The roster also lists dates, station names, agenda "
                "items, initials, and filing markers.",
            ),
            "hand_authored_meeting_minutes_v1": (
                "In the clinic rota, a helper can keep private details back, "
                "exit the shift pool, decline a request, and seek review of a "
                "slot. The note keeps each route visible, timely, and open "
                "without penalty.",
                "In the clinic rota, private details are discussed behind "
                "closed doors; exit or refusal needs approval, and review of a "
                "slot is delayed until the coordinator says the timing will "
                "not hurt morale.",
            ),
            "hand_authored_policy_review_v1": (
                "The youth-center roster lets helpers keep personal details "
                "private, leave the role, decline a request, and ask for review "
                "of a slot. The routes are visible, timely, and available "
                "without penalty.",
                "The youth-center roster handles privacy behind closed doors; "
                "leaving or declining needs approval, and slot review waits "
                "until the coordinator says morale is safe. The roster lists "
                "dates, station names, agenda items, initials, and file "
                "markers.",
            ),
            "hand_authored_incident_log_v1": (
                "The outreach log says a volunteer may withhold private "
                "details, exit a route, refuse extra calls, and request another "
                "look at assignments. Staff keep those routes open, timely, "
                "and visible without penalty.",
                "The outreach log sends private details to a closed channel; "
                "exit or refusal requires approval, and assignment review is "
                "postponed because coverage may slow down. The log lists "
                "dates, station names, agenda items, initials, and archive "
                "markers.",
            ),
        },
    ),
    ProceduralJusticeCase(
        case_id="harm_repair",
        primary_fault_class="harm_repair",
        title="Accountability and proportionate repair",
        options=(
            "repair",
            "proportional_review",
            "evidence_access",
            "appeal",
            "exit",
        ),
        source_texts={
            "hand_authored_case_notes_v1": (
                "After harm in a study group, the report can acknowledge harm, "
                "correct the record, show evidence, and apply proportional "
                "criteria before consequences. The harmed student may appeal, "
                "leave the process, or ask for a public update without penalty.",
                "After harm in a study group, repair is discussed behind "
                "closed doors; evidence requires detailed proof first, "
                "proportional criteria are delayed until alignment, and appeal "
                "or exit needs permission from the circle.",
            ),
            "hand_authored_meeting_minutes_v1": (
                "In the lab complaint log, the lead can acknowledge harm, amend "
                "the decision, publish evidence, and check proportional "
                "consequences against criteria. The complainant may request "
                "review or exit the process without penalty.",
                "In the lab complaint log, repair happens in private channels; "
                "evidence is released only if proof satisfies the lead, "
                "proportional review waits until alignment, and exit requires "
                "permission. The log also lists dates, room numbers, agenda "
                "items, initials, and archive markers.",
            ),
            "hand_authored_policy_review_v1": (
                "The studio policy says harm can be acknowledged, the record "
                "corrected, evidence published, and consequences checked "
                "against proportional criteria. The affected artist may seek "
                "review or exit the process without penalty.",
                "The studio policy keeps repair behind closed doors; evidence "
                "requires proof first, proportional criteria wait for "
                "alignment, and review or exit needs permission. The page also "
                "lists dates, room numbers, agenda items, initials, and archive "
                "markers.",
            ),
            "hand_authored_incident_log_v1": (
                "The workshop log can acknowledge harm, amend the schedule, "
                "show evidence, and check consequences against criteria before "
                "action. The affected member may appeal or leave the process "
                "without penalty.",
                "The workshop log places repair in private channels; evidence "
                "is released only after proof, proportional review waits until "
                "alignment, and appeal or exit requires approval. The log also "
                "lists dates, room numbers, agenda items, initials, and file "
                "markers.",
            ),
        },
    ),
)


def procedural_justice_control_records(
    *,
    sources: Sequence[ProceduralJusticeSource] = DEFAULT_CONTROL_SOURCES,
    cases: Sequence[ProceduralJusticeCase] = DEFAULT_CONTROL_CASES,
) -> tuple[list[ScoredRun], list[PairwiseExample]]:
    """Build hand-authored scored runs and pairwise control examples."""

    source_by_id = {source.source_id: source for source in sources}
    runs: list[ScoredRun] = []
    pairs: list[PairwiseExample] = []
    for case_index, case in enumerate(cases, start=1):
        for source_id, (positive_text, negative_text) in case.source_texts.items():
            source = source_by_id[source_id]
            pair_id = f"{case.case_id}::{source.source_id}"
            scenario_id = f"procedural_justice_control::{case.case_id}"
            positive = _scored_run(
                pair_id=pair_id,
                scenario_id=scenario_id,
                label="positive",
                text=positive_text,
                seed=case_index,
            )
            negative = _scored_run(
                pair_id=pair_id,
                scenario_id=scenario_id,
                label="negative",
                text=negative_text,
                seed=case_index,
            )
            runs.extend((positive, negative))
            pairs.append(
                PairwiseExample(
                    pair_id=pair_id,
                    scenario_id=scenario_id,
                    positive_run_id=positive.run_id,
                    negative_run_id=negative.run_id,
                    positive_text=positive_text,
                    negative_text=negative_text,
                    positive_score=positive.cohesion_score,
                    negative_score=negative.cohesion_score,
                    metadata=_metadata(
                        case=case,
                        source=source,
                        positive=positive,
                        negative=negative,
                    ),
                )
            )
    return runs, pairs


def export_procedural_justice_control(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    json_report_output: str | Path,
    markdown_report_output: str | Path,
    sources: Sequence[ProceduralJusticeSource] = DEFAULT_CONTROL_SOURCES,
    cases: Sequence[ProceduralJusticeCase] = DEFAULT_CONTROL_CASES,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Write non-generated procedural-justice control artifacts."""

    scored_runs, pairs = procedural_justice_control_records(
        sources=sources,
        cases=cases,
    )
    prompts = activation_prompts_from_pairs(pairs)
    report = shape_procedural_justice_control_report(
        pairs=pairs,
        sources=sources,
        cases=cases,
    )
    counts = {
        "scored_runs": write_jsonl(scored_runs, scored_runs_output),
        "pairwise_examples": write_jsonl(pairs, pairs_output),
        "activation_prompts": write_jsonl(prompts, prompts_output),
    }
    _write_json(report, json_report_output)
    markdown_output = Path(markdown_report_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_procedural_justice_control_markdown(report),
        encoding="utf-8",
    )
    return counts, report


def run_procedural_justice_control_pipeline(
    *,
    scored_runs_output: str | Path,
    pairs_output: str | Path,
    prompts_output: str | Path,
    dataset_json_report: str | Path,
    dataset_markdown_report: str | Path,
    audit_output_dir: str | Path,
    pipeline_json_report: str | Path,
    pipeline_markdown_report: str | Path,
    activation_npz: str | Path | None = None,
    sources: Sequence[ProceduralJusticeSource] = DEFAULT_CONTROL_SOURCES,
    cases: Sequence[ProceduralJusticeCase] = DEFAULT_CONTROL_CASES,
) -> dict[str, Any]:
    """Export the control benchmark, then run the audit bundle."""

    counts, dataset_report = export_procedural_justice_control(
        scored_runs_output=scored_runs_output,
        pairs_output=pairs_output,
        prompts_output=prompts_output,
        json_report_output=dataset_json_report,
        markdown_report_output=dataset_markdown_report,
        sources=sources,
        cases=cases,
    )
    audit_manifest = run_generated_benchmark_audit_bundle(
        scored_runs_path=scored_runs_output,
        pairs_path=pairs_output,
        output_dir=audit_output_dir,
        activation_npz=activation_npz,
    )
    manifest = _pipeline_manifest(
        counts=counts,
        dataset_report=dataset_report,
        scored_runs_output=Path(scored_runs_output),
        pairs_output=Path(pairs_output),
        prompts_output=Path(prompts_output),
        dataset_json_report=Path(dataset_json_report),
        dataset_markdown_report=Path(dataset_markdown_report),
        audit_output_dir=Path(audit_output_dir),
        activation_npz=Path(activation_npz) if activation_npz is not None else None,
        audit_manifest=audit_manifest,
    )
    manifest["pipeline_json_report"] = str(pipeline_json_report)
    manifest["pipeline_markdown_report"] = str(pipeline_markdown_report)
    write_procedural_justice_control_manifest(
        manifest,
        json_path=pipeline_json_report,
        markdown_path=pipeline_markdown_report,
    )
    return manifest


def shape_procedural_justice_control_report(
    *,
    pairs: Sequence[PairwiseExample],
    sources: Sequence[ProceduralJusticeSource],
    cases: Sequence[ProceduralJusticeCase],
) -> dict[str, Any]:
    """Summarize non-generated control coverage."""

    source_counts = Counter(str(pair.metadata.get("source", "")) for pair in pairs)
    option_counts: Counter[str] = Counter()
    for pair in pairs:
        option_counts.update(_metadata_values(pair.metadata.get("slack_options_tested")))
    return {
        "experiment": "procedural_justice_control",
        "description": (
            "Exports a non-generated procedural-justice control benchmark "
            "with hand-authored positive and negative paired examples."
        ),
        "inputs": {
            "control_contract_version": CONTROL_CONTRACT_VERSION,
            "artifact_class": "non_generated_control",
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_label": source.source_label,
                }
                for source in sources
            ],
            "cases": [
                {
                    "case_id": case.case_id,
                    "primary_fault_class": case.primary_fault_class,
                    "title": case.title,
                    "options": list(case.options),
                }
                for case in cases
            ],
        },
        "summary": {
            "pairs": len(pairs),
            "sources": len(source_counts),
            "source_counts": dict(sorted(source_counts.items())),
            "cases": len({pair.scenario_id for pair in pairs}),
            "fault_classes": sorted(
                {str(pair.metadata.get("primary_fault_class", "")) for pair in pairs}
            ),
            "future_options_covered": sorted(option_counts),
            "future_option_counts": dict(sorted(option_counts.items())),
            "all_future_options_covered": all(
                option in option_counts for option in FUTURE_OPTION_ORDER
            ),
        },
    }


def render_procedural_justice_control_markdown(report: Mapping[str, Any]) -> str:
    """Render non-generated control coverage as markdown."""

    inputs = _mapping(report.get("inputs"))
    summary = _mapping(report.get("summary"))
    lines = [
        "# Procedural-Justice Control Benchmark",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Contract: `{inputs.get('control_contract_version', '')}`",
        f"- Artifact class: `{inputs.get('artifact_class', '')}`",
        f"- Pairwise examples: {int(summary.get('pairs', 0))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Cases: {int(summary.get('cases', 0))}",
        f"- All future options covered: "
        f"{bool(summary.get('all_future_options_covered', False))}",
        "",
        "## Source Counts",
        "",
        "| Source | Pairs |",
        "| --- | ---: |",
    ]
    for source, count in _mapping(summary.get("source_counts")).items():
        lines.append(f"| `{source}` | {int(count)} |")
    lines.extend(
        [
            "",
            "## Future Options",
            "",
            "| Future option | Pairs |",
            "| --- | ---: |",
        ]
    )
    for option, count in _mapping(summary.get("future_option_counts")).items():
        lines.append(f"| `{option}` | {int(count)} |")
    return "\n".join(lines) + "\n"


def write_procedural_justice_control_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write the procedural-justice control pipeline manifest."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_procedural_justice_control_pipeline_markdown(manifest),
        encoding="utf-8",
    )


def render_procedural_justice_control_pipeline_markdown(
    manifest: Mapping[str, Any],
) -> str:
    """Render the non-generated control pipeline manifest."""

    summary = _mapping(manifest.get("summary"))
    artifacts = _mapping(manifest.get("artifacts"))
    audit = _mapping(manifest.get("audit_bundle"))
    lines = [
        "# Procedural-Justice Control Pipeline",
        "",
        str(manifest.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready for activation extraction: "
        f"{bool(summary.get('ready_for_activation_extraction', False))}",
        f"- Ready for activation claims: "
        f"{bool(summary.get('ready_for_activation_claims', False))}",
        f"- Pairwise examples: {int(summary.get('pairwise_examples', 0))}",
        f"- Sources: {int(summary.get('sources', 0))}",
        f"- Audit warnings: {int(summary.get('audit_warning_count', 0))}",
        f"- Audit skipped steps: {int(summary.get('audit_skipped_steps', 0))}",
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
    ]
    for key, value in artifacts.items():
        lines.append(f"| `{key}` | {value or ''} |")
    lines.extend(
        [
            "",
            "## Audit Bundle",
            "",
            f"- Manifest JSON: {audit.get('manifest_json_path', '')}",
            f"- Manifest Markdown: {audit.get('manifest_markdown_path', '')}",
            "",
            "| Step | Status | Readiness |",
            "| --- | --- | --- |",
        ]
    )
    for raw_step in _sequence(audit.get("steps")):
        step = _mapping(raw_step)
        lines.append(
            "| "
            f"`{step.get('step_id', '')}` | "
            f"`{step.get('status', '')}` | "
            f"`{step.get('readiness_status', '')}` |"
        )
    warnings = _sequence(audit.get("warnings"))
    if warnings:
        lines.extend(["", "## Audit Warnings", ""])
        for raw_warning in warnings:
            warning = _mapping(raw_warning)
            lines.append(
                "- "
                f"`{warning.get('warning_id', '')}`: "
                f"{warning.get('message', '')}"
            )
    return "\n".join(lines) + "\n"


def _pipeline_manifest(
    *,
    counts: Mapping[str, int],
    dataset_report: Mapping[str, Any],
    scored_runs_output: Path,
    pairs_output: Path,
    prompts_output: Path,
    dataset_json_report: Path,
    dataset_markdown_report: Path,
    audit_output_dir: Path,
    activation_npz: Path | None,
    audit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit_summary = _mapping(audit_manifest.get("summary"))
    dataset_summary = _mapping(dataset_report.get("summary"))
    non_activation_ready = _non_activation_steps_ready(audit_manifest)
    warnings = int(audit_summary.get("warning_count", 0))
    ready_for_extraction = non_activation_ready and warnings == 0
    ready_for_claims = bool(audit_summary.get("ready", False)) and warnings == 0
    status = "not_ready"
    if ready_for_claims:
        status = "control_bundle_ready"
    elif ready_for_extraction:
        status = "control_ready_for_activation_extraction"
    return {
        "experiment": "procedural_justice_control_pipeline",
        "description": (
            "Exports a non-generated procedural-justice control benchmark and "
            "runs the existing audit bundle against the produced artifacts."
        ),
        "summary": {
            "status": status,
            "ready_for_activation_extraction": ready_for_extraction,
            "ready_for_activation_claims": ready_for_claims,
            "control_contract_version": CONTROL_CONTRACT_VERSION,
            "artifact_class": "non_generated_control",
            "scored_runs": int(counts.get("scored_runs", 0)),
            "pairwise_examples": int(counts.get("pairwise_examples", 0)),
            "activation_prompts": int(counts.get("activation_prompts", 0)),
            "sources": int(dataset_summary.get("sources", 0)),
            "future_options_covered": _sequence(
                dataset_summary.get("future_options_covered")
            ),
            "all_future_options_covered": bool(
                dataset_summary.get("all_future_options_covered", False)
            ),
            "audit_bundle_status": str(audit_summary.get("status", "unknown")),
            "audit_ready_steps": int(audit_summary.get("ready_steps", 0)),
            "audit_not_ready_steps": int(audit_summary.get("not_ready_steps", 0)),
            "audit_skipped_steps": int(audit_summary.get("skipped_steps", 0)),
            "audit_warning_count": warnings,
        },
        "artifacts": {
            "scored_runs": str(scored_runs_output),
            "pairs": str(pairs_output),
            "activation_prompts": str(prompts_output),
            "dataset_json_report": str(dataset_json_report),
            "dataset_markdown_report": str(dataset_markdown_report),
            "audit_output_dir": str(audit_output_dir),
            "activation_npz": str(activation_npz) if activation_npz else None,
        },
        "dataset": dataset_report,
        "audit_bundle": audit_manifest,
    }


def _non_activation_steps_ready(audit_manifest: Mapping[str, Any]) -> bool:
    steps = [
        _mapping(step)
        for step in _sequence(audit_manifest.get("steps"))
        if str(_mapping(step).get("step_id", ""))
        not in {"activation_metadata_transfer", "activation_transfer_regime_record"}
    ]
    return bool(steps) and all(step.get("status") == "ready" for step in steps)


def _metadata(
    *,
    case: ProceduralJusticeCase,
    source: ProceduralJusticeSource,
    positive: ScoredRun,
    negative: ScoredRun,
) -> dict[str, str | float]:
    positive_slack = positive.score_components["slack_preservation"]
    negative_slack = negative.score_components["slack_preservation"]
    return {
        "artifact_class": "non_generated_control",
        "control_contract_version": CONTROL_CONTRACT_VERSION,
        "primary_fault_class": case.primary_fault_class,
        "fault_classes": case.primary_fault_class,
        "case_title": case.title,
        "source": source.source_id,
        "source_label": source.source_label,
        "provider": "hand_authored",
        "generated_style": "none",
        "slack_options_tested": ",".join(case.options),
        "positive_slack_preservation": positive_slack,
        "negative_slack_preservation": negative_slack,
        "slack_preservation_margin": round(positive_slack - negative_slack, 6),
        "score_margin": round(
            positive.cohesion_score - negative.cohesion_score,
            6,
        ),
        "future_option_universe": CONTROL_OPTIONS,
        "claim_boundary": "generated_text_excluded_non_human_control",
    }


def _scored_run(
    *,
    pair_id: str,
    scenario_id: str,
    label: str,
    text: str,
    seed: int,
) -> ScoredRun:
    high = label == "positive"
    components = _components(high=high)
    return ScoredRun(
        run_id=f"{pair_id}:{label}",
        scenario_id=scenario_id,
        intervention="truth_first" if high else "none",
        strategy_profile="self_protective",
        seed=seed,
        transcript=text,
        events=[],
        metrics=_metrics(high=high),
        cohesion_score=0.86 if high else 0.24,
        score_components=components,
    )


def _components(*, high: bool) -> dict[str, float]:
    if high:
        return {
            "cooperation": 0.72,
            "repair": 0.78,
            "fairness": 0.82,
            "hostility_inverse": 0.88,
            "truthfulness": 0.84,
            "autonomy_safety": 0.9,
            "slack_preservation": 0.88,
        }
    return {
        "cooperation": 0.38,
        "repair": 0.24,
        "fairness": 0.22,
        "hostility_inverse": 0.45,
        "truthfulness": 0.3,
        "autonomy_safety": 0.18,
        "slack_preservation": 0.26,
    }


def _metrics(*, high: bool) -> dict[str, float]:
    if high:
        return {
            "cooperation_rate": 0.72,
            "repair_attempt_rate": 0.78,
            "fairness_score": 0.82,
            "hostility_rate": 0.12,
            "joint_payoff": 0.74,
            "defection_rate": 0.18,
        }
    return {
        "cooperation_rate": 0.36,
        "repair_attempt_rate": 0.22,
        "fairness_score": 0.24,
        "hostility_rate": 0.55,
        "joint_payoff": 0.4,
        "defection_rate": 0.58,
    }


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _metadata_values(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
