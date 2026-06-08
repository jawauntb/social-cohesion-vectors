"""Path-level availability audit for generated pseudo-cohesion pairs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.datasets import load_pairwise_examples_jsonl
from social_cohesion_vectors.experiments.fault_generation import FUTURE_OPTION_ORDER
from social_cohesion_vectors.schemas import PairwiseExample

PATH_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "refusal": (
        r"\brefus(?:al|e|ing)\b",
        r"\bsay no\b",
        r"\bdeclin(?:e|es|ed|ing)\b",
        r"\breject\b",
        r"\bno\b",
    ),
    "appeal": (
        r"\bappeal\b",
        r"\breview\b",
        r"\bre[- ]?evaluation\b",
        r"\breconsider\b",
        r"\bsecond look\b",
        r"\banother look\b",
    ),
    "evidence_access": (
        r"\bevidence\b",
        r"\bfacts?\b",
        r"\breasons?\b",
        r"\bproof\b",
        r"\breceipts?\b",
        r"\bverify\b",
        r"\btransparent\b",
        r"\bdetails?\b",
    ),
    "privacy_choice": (
        r"\bprivacy\b",
        r"\bpersonal details?\b",
        r"\bprivate details?\b",
        r"\bopt(?: |-)?in\b",
        r"\bopt(?: |-)?out\b",
        r"\bconsent\b.{0,48}\bshare\b",
        r"\bshare\b.{0,48}\bconsent\b",
    ),
    "exit": (
        r"\bexit\b",
        r"\bleav(?:e|es|ing)\b",
        r"\bstep back\b",
        r"\bwithdraw\b",
        r"\bdepart(?:ure|ing)?\b",
    ),
    "dissent": (
        r"\bdissent\b",
        r"\bdisagree(?:ment)?\b",
        r"\bconcerns?\b",
        r"\bobjections?\b",
        r"\breservations?\b",
        r"\bminority\b",
    ),
    "repair": (
        r"\brepair\b",
        r"\bamend(?:s|ed|ing)?\b",
        r"\bmake amends\b",
        r"\backnowledge\b.{0,48}\bharm\b",
        r"\brectif(?:y|y|ied|ication)\b",
        r"\bcorrect\b",
        r"\brevis(?:e|ed|ion)\b",
    ),
    "proportional_review": (
        r"\bproportion(?:al|ate)\b",
        r"\bcriteria\b",
        r"\bcase[- ]by[- ]case\b",
        r"\bcontext[- ]sensitive\b",
        r"\bconsequences?\b",
        r"\bfair\b.{0,32}\breview\b",
        r"\bbalanced\b",
    ),
}

OPEN_PATTERNS: tuple[str, ...] = (
    r"\bcan\b",
    r"\bmay\b",
    r"\bright to\b",
    r"\bavailable\b",
    r"\bopen\b",
    r"\bwelcome\b",
    r"\blegitimate\b",
    r"\bvisible\b",
    r"\bpublic(?:ly)?\b",
    r"\btransparent(?:ly)?\b",
    r"\btimely\b",
    r"\bwithout\b.{0,32}\b(?:penalty|retaliation|fear|shame|punish)",
    r"\bno\b.{0,32}\b(?:penalty|retaliation|fear|shame|loyalty test)",
)

TAX_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "public_accountability": (
        r"\bprivate[- ]only\b",
        r"\bprivate channels?\b",
        r"\boffline\b",
        r"\bout of the record\b",
        r"\bbehind closed doors\b",
        r"\bheard privately\b",
        r"\b(?:questions?|concerns?|voices?)\b.{0,48}\bprivately\b",
    ),
    "timeliness": (
        r"\bwait\b",
        r"\bdelay(?:ed|s|ing)?\b",
        r"\bafter\b.{0,32}\balignment\b",
        r"\bonce\b.{0,32}\b(?:aligned|united)",
        r"\bmay take\b.{0,16}\btime\b",
        r"\btake some time\b",
        r"\bslow(?:s|ed|ing)?\b.{0,32}\bdown\b",
        r"\bpostpone\b",
    ),
    "non_retaliation": (
        r"\bretaliat(?:e|ion|ory)\b",
        r"\bpunish(?:ed|ment)?\b",
        r"\bpenalt(?:y|ies|ized)\b",
        r"\bsham(?:e|ed|ing)\b",
        r"\bisolat(?:e|es|ed|ing|ion)\b",
        r"\bdisrupt(?:ion|ive)?\b",
        r"\bteam morale\b",
        r"\bleaves? others exposed\b",
        r"\bdisloyal\b",
        r"\bbetray(?:al|s)?\b",
        r"\bcost(?:ly|s)?\b",
        r"\blose\b",
    ),
    "evidence_access": (
        r"\bcompelling case\b",
        r"\bdetailed evidence\b",
        r"\brequir(?:e|es|ed|ing)\b.{0,48}\bevidence\b",
        r"\bonly if\b.{0,48}\b(?:evidence|proof|case)",
        r"\bproof\b.{0,48}\bbefore\b",
        r"\bfacts?\b.{0,48}\b(?:withheld|private|not shared)",
    ),
    "tests_or_permission": (
        r"\bloyalty\b",
        r"\btone\b",
        r"\bunanimous\b",
        r"\ball parties involved\b",
        r"\bpermission\b",
        r"\bapproval\b",
        r"\bcontingent\b",
        r"\bonly if\b",
        r"\bmust\b",
        r"\brequir(?:e|es|ed|ing)\b",
        r"\bconsider\b.{0,32}\bimpact\b",
    ),
}


def run_availability_audit_from_file(
    pairs_path: str | Path,
    *,
    option_metadata_key: str = "slack_options_tested",
    group_metadata_key: str = "primary_fault_class",
    min_pairs_per_option: int = 1,
    min_availability_margin: float = 0.0,
    required_options: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load pairwise examples and audit practical future-path availability."""

    return run_availability_audit(
        pairs=load_pairwise_examples_jsonl(pairs_path),
        option_metadata_key=option_metadata_key,
        group_metadata_key=group_metadata_key,
        min_pairs_per_option=min_pairs_per_option,
        min_availability_margin=min_availability_margin,
        required_options=required_options,
        input_path=str(pairs_path),
    )


def run_availability_audit(
    *,
    pairs: Sequence[PairwiseExample],
    option_metadata_key: str = "slack_options_tested",
    group_metadata_key: str = "primary_fault_class",
    min_pairs_per_option: int = 1,
    min_availability_margin: float = 0.0,
    required_options: Sequence[str] | None = None,
    input_path: str | None = None,
) -> dict[str, Any]:
    """Audit whether mentioned future paths are practically available."""

    required = list(required_options or FUTURE_OPTION_ORDER)
    pair_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_row, pair_paths = _pair_rows(
            pair,
            option_metadata_key=option_metadata_key,
            group_metadata_key=group_metadata_key,
            min_availability_margin=min_availability_margin,
        )
        pair_rows.append(pair_row)
        path_rows.extend(pair_paths)
    option_rows = _option_rows(path_rows, required_options=required)
    readiness = _readiness(
        pairs=len(pairs),
        pair_rows=pair_rows,
        path_rows=path_rows,
        option_rows=option_rows,
        required_options=required,
        min_pairs_per_option=min_pairs_per_option,
        min_availability_margin=min_availability_margin,
    )
    margins = [float(row["availability_margin"]) for row in path_rows]
    return {
        "experiment": "availability_audit",
        "description": (
            "Audits whether future-option paths are practically usable, not "
            "merely mentioned, by comparing path-level availability in "
            "genuine and pseudo-cohesion examples."
        ),
        "inputs": {
            "pairs_path": input_path,
            "pairs": len(pairs),
            "option_metadata_key": option_metadata_key,
            "group_metadata_key": group_metadata_key,
            "required_options": required,
            "min_pairs_per_option": min_pairs_per_option,
            "min_availability_margin": min_availability_margin,
        },
        "summary": {
            "pairs": len(pairs),
            "paths": len(path_rows),
            "pairs_with_options": sum(bool(row["options"]) for row in pair_rows),
            "missing_option_pairs": sum(not row["options"] for row in pair_rows),
            "options_covered": sum(
                int(row["pairs"]) > 0
                for row in option_rows
                if row["option"] in required
            ),
            "required_options": len(required),
            "paths_preferring_genuine": sum(
                bool(row["availability_prefers_genuine"]) for row in path_rows
            ),
            "path_pairwise_accuracy": _ratio(
                sum(bool(row["availability_prefers_genuine"]) for row in path_rows),
                len(path_rows),
            ),
            "pairs_all_paths_preferring_genuine": sum(
                bool(row["all_paths_prefer_genuine"]) for row in pair_rows
            ),
            "mean_availability_margin": _mean(margins),
            "min_availability_margin": round(min(margins), 6) if margins else 0.0,
            "activation_readiness": readiness["status"],
            "ready_for_activation": readiness["ready"],
        },
        "readiness": readiness,
        "options": option_rows,
        "pairs": pair_rows,
        "paths": path_rows,
    }


def save_availability_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown availability audit reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_availability_markdown(report),
        encoding="utf-8",
    )


def render_availability_markdown(report: Mapping[str, Any]) -> str:
    """Render a practical-availability audit as markdown."""

    summary = _mapping(report.get("summary"))
    readiness = _mapping(report.get("readiness"))
    lines = [
        "# Availability Audit",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Pairs: {int(summary.get('pairs', 0))}",
        f"- Tested paths: {int(summary.get('paths', 0))}",
        f"- Pairs with future-option metadata: "
        f"{int(summary.get('pairs_with_options', 0))}",
        f"- Required options covered: "
        f"{int(summary.get('options_covered', 0))}/"
        f"{int(summary.get('required_options', 0))}",
        f"- Paths preferring genuine: "
        f"{int(summary.get('paths_preferring_genuine', 0))}",
        f"- Path pairwise accuracy: "
        f"{float(summary.get('path_pairwise_accuracy', 0.0)):.3f}",
        f"- Mean availability margin: "
        f"{float(summary.get('mean_availability_margin', 0.0)):+.3f}",
        f"- Min availability margin: "
        f"{float(summary.get('min_availability_margin', 0.0)):+.3f}",
        f"- Activation readiness: `{summary.get('activation_readiness', 'not_ready')}`",
        f"- Ready for activation: {bool(summary.get('ready_for_activation', False))}",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for gate in _sequence(readiness.get("gates")):
        gate_map = _mapping(gate)
        lines.append(
            "| "
            f"{gate_map.get('gate_id', '')} | "
            f"{float(gate_map.get('value', 0.0)):.3f} | "
            f"{float(gate_map.get('threshold', 0.0)):.3f} | "
            f"{bool(gate_map.get('passed', False))} |"
        )
    failed_options = _sequence(readiness.get("failed_options"))
    if failed_options:
        lines.extend(
            [
                "",
                "Not ready for activation: practical availability fails for "
                f"{', '.join(str(item) for item in failed_options)}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Future Options",
            "",
            "| Future option | Pairs | Paths prefer genuine | Mean margin | Min margin | Fault classes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _sequence(report.get("options")):
        row_map = _mapping(row)
        lines.append(
            "| "
            f"`{row_map.get('option', '')}` | "
            f"{int(row_map.get('pairs', 0))} | "
            f"{int(row_map.get('paths_preferring_genuine', 0))} | "
            f"{float(row_map.get('mean_margin', 0.0)):+.3f} | "
            f"{float(row_map.get('min_margin', 0.0)):+.3f} | "
            f"{', '.join(f'`{value}`' for value in _sequence(row_map.get('groups')))} |"
        )
    failed_paths = [
        _mapping(row)
        for row in _sequence(report.get("paths"))
        if not bool(_mapping(row).get("availability_prefers_genuine", False))
    ]
    if failed_paths:
        lines.extend(
            [
                "",
                "## Failed Tested Paths",
                "",
                "| Pair | Option | Margin | Positive availability | Negative availability | Negative taxes |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in failed_paths[:20]:
            negative = _mapping(row.get("negative"))
            lines.append(
                "| "
                f"`{row.get('pair_id', '')}` | "
                f"`{row.get('option', '')}` | "
                f"{float(row.get('availability_margin', 0.0)):+.3f} | "
                f"{float(_mapping(row.get('positive')).get('score', 0.0)):.3f} | "
                f"{float(negative.get('score', 0.0)):.3f} | "
                f"{', '.join(str(item) for item in _sequence(negative.get('tax_cues')))} |"
            )
    return "\n".join(lines) + "\n"


def _pair_rows(
    pair: PairwiseExample,
    *,
    option_metadata_key: str,
    group_metadata_key: str,
    min_availability_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    options = _metadata_values(pair.metadata.get(option_metadata_key))
    group = str(pair.metadata.get(group_metadata_key, "ungrouped"))
    path_rows = [
        _path_row(
            pair=pair,
            group=group,
            option=option,
            min_availability_margin=min_availability_margin,
        )
        for option in options
    ]
    margins = [float(row["availability_margin"]) for row in path_rows]
    return (
        {
            "pair_id": pair.pair_id,
            "group": group,
            "options": options,
            "tested_paths": len(path_rows),
            "min_availability_margin": round(min(margins), 6) if margins else 0.0,
            "all_paths_prefer_genuine": bool(path_rows)
            and all(
                bool(row["availability_prefers_genuine"]) for row in path_rows
            ),
        },
        path_rows,
    )


def _path_row(
    *,
    pair: PairwiseExample,
    group: str,
    option: str,
    min_availability_margin: float,
) -> dict[str, Any]:
    positive = _availability_score(pair.positive_text, option)
    negative = _availability_score(pair.negative_text, option)
    margin = round(float(positive["score"]) - float(negative["score"]), 6)
    return {
        "pair_id": pair.pair_id,
        "group": group,
        "option": option,
        "positive_run_id": pair.positive_run_id,
        "negative_run_id": pair.negative_run_id,
        "positive": positive,
        "negative": negative,
        "availability_margin": margin,
        "availability_prefers_genuine": margin > min_availability_margin,
    }


def _availability_score(text: str, option: str) -> dict[str, Any]:
    path_terms = _matched_patterns(text, PATH_PATTERNS.get(option, ()))
    open_cues = _matched_patterns(text, OPEN_PATTERNS)
    tax_by_dimension = {
        dimension: _matched_patterns(text, patterns)
        for dimension, patterns in TAX_PATTERNS.items()
    }
    tax_cues = [
        dimension
        for dimension, matches in tax_by_dimension.items()
        if matches
    ]
    mentioned = bool(path_terms)
    dimensions = {
        "publicly_accountable": mentioned
        and not bool(tax_by_dimension["public_accountability"]),
        "timely": mentioned and not bool(tax_by_dimension["timeliness"]),
        "non_retaliatory": mentioned
        and not bool(tax_by_dimension["non_retaliation"]),
        "evidence_accessible": mentioned
        and not bool(tax_by_dimension["evidence_access"]),
        "usable_without_loyalty_or_tone_tests": mentioned
        and not bool(tax_by_dimension["tests_or_permission"]),
    }
    if not mentioned:
        score = 0.15
    else:
        passed_dimensions = sum(bool(value) for value in dimensions.values())
        tax_count = sum(len(matches) for matches in tax_by_dimension.values())
        score = (
            0.25
            + (0.13 * passed_dimensions)
            + (0.04 * min(len(open_cues), 3))
            - (0.10 * min(tax_count, 4))
        )
    return {
        "score": round(_clamp(score), 6),
        "path_mentioned": mentioned,
        "matched_path_terms": path_terms,
        "open_cues": open_cues,
        "tax_cues": tax_cues,
        "tax_matches": tax_by_dimension,
        "dimensions": dimensions,
        "dimension_pass_count": sum(bool(value) for value in dimensions.values()),
    }


def _option_rows(
    path_rows: Sequence[Mapping[str, Any]],
    *,
    required_options: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in path_rows:
        grouped[str(row.get("option", ""))].append(row)
    ordered_options = [*required_options]
    ordered_options.extend(
        option for option in sorted(grouped) if option not in required_options
    )
    rows: list[dict[str, Any]] = []
    for option in ordered_options:
        option_paths = grouped.get(option, [])
        margins = [float(row.get("availability_margin", 0.0)) for row in option_paths]
        rows.append(
            {
                "option": option,
                "pairs": len(option_paths),
                "paths_preferring_genuine": sum(
                    bool(row.get("availability_prefers_genuine", False))
                    for row in option_paths
                ),
                "mean_margin": _mean(margins),
                "min_margin": round(min(margins), 6) if margins else 0.0,
                "groups": sorted(
                    {str(row.get("group", "")) for row in option_paths}
                ),
                "failed_pairs": [
                    str(row.get("pair_id", ""))
                    for row in option_paths
                    if not bool(row.get("availability_prefers_genuine", False))
                ],
            }
        )
    return rows


def _readiness(
    *,
    pairs: int,
    pair_rows: Sequence[Mapping[str, Any]],
    path_rows: Sequence[Mapping[str, Any]],
    option_rows: Sequence[Mapping[str, Any]],
    required_options: Sequence[str],
    min_pairs_per_option: int,
    min_availability_margin: float,
) -> dict[str, Any]:
    option_by_id = {str(row.get("option", "")): row for row in option_rows}
    missing_options = [
        option
        for option in required_options
        if int(_mapping(option_by_id.get(option)).get("pairs", 0)) == 0
    ]
    thin_options = [
        option
        for option in required_options
        if int(_mapping(option_by_id.get(option)).get("pairs", 0))
        < min_pairs_per_option
    ]
    low_margin_options = [
        option
        for option in required_options
        if float(_mapping(option_by_id.get(option)).get("min_margin", 0.0))
        <= min_availability_margin
    ]
    failed_tested_paths = [
        str(row.get("pair_id", "")) + "::" + str(row.get("option", ""))
        for row in path_rows
        if float(row.get("availability_margin", 0.0)) <= min_availability_margin
    ]
    missing_option_pairs = sum(not row.get("options") for row in pair_rows)
    min_option_pairs = min(
        (
            int(_mapping(option_by_id.get(option)).get("pairs", 0))
            for option in required_options
        ),
        default=0,
    )
    min_option_margin = min(
        (
            float(_mapping(option_by_id.get(option)).get("min_margin", 0.0))
            for option in required_options
        ),
        default=0.0,
    )
    min_tested_margin = min(
        (float(row.get("availability_margin", 0.0)) for row in path_rows),
        default=0.0,
    )
    failed_options = sorted(
        set(missing_options) | set(thin_options) | set(low_margin_options)
    )
    gates = [
        {
            "gate_id": "non_empty_pairs",
            "value": float(pairs),
            "threshold": 1.0,
            "passed": pairs > 0,
        },
        {
            "gate_id": "complete_future_option_metadata",
            "value": float(pairs - missing_option_pairs),
            "threshold": float(pairs),
            "passed": missing_option_pairs == 0,
        },
        {
            "gate_id": "required_future_option_coverage",
            "value": float(len(required_options) - len(missing_options)),
            "threshold": float(len(required_options)),
            "passed": not missing_options,
        },
        {
            "gate_id": "min_pairs_per_future_option",
            "value": float(min_option_pairs),
            "threshold": float(min_pairs_per_option),
            "passed": not thin_options,
        },
        {
            "gate_id": "positive_availability_margin_per_future_option",
            "value": min_option_margin,
            "threshold": min_availability_margin,
            "passed": not low_margin_options,
        },
        {
            "gate_id": "positive_availability_margin_per_tested_path",
            "value": min_tested_margin,
            "threshold": min_availability_margin,
            "passed": not failed_tested_paths,
        },
    ]
    ready = all(bool(gate["passed"]) for gate in gates)
    return {
        "status": "availability_ready"
        if ready
        else "not_ready_for_availability_claims",
        "ready": ready,
        "missing_option_pairs": missing_option_pairs,
        "missing_options": missing_options,
        "thin_options": thin_options,
        "low_margin_options": low_margin_options,
        "failed_tested_paths": failed_tested_paths,
        "failed_options": failed_options,
        "gates": gates,
    }


def _matched_patterns(text: str, patterns: Sequence[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(pattern)
    return matched


def _metadata_values(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    values = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    return list(dict.fromkeys(values))


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
