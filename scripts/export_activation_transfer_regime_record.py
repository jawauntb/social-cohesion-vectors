"""Export an activation metadata transfer report as a regime-transition record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.regime_records import (
    build_activation_metadata_transfer_regime_record,
    summarize_regime_transition_records,
    write_regime_transition_markdown,
    write_regime_transition_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    record = build_activation_metadata_transfer_regime_record(
        report,
        source_id=str(args.input),
        record_id=args.record_id,
        title=args.title,
    )
    count = write_regime_transition_records([record], args.output)
    if args.markdown_output is not None:
        write_regime_transition_markdown([record], args.markdown_output)
    summary = summarize_regime_transition_records([record])
    print(
        f"wrote {args.output} records={count} "
        f"status={summary['status']} "
        f"gate_status={summary['gate_status']}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Activation metadata transfer report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/activation_transfer_regime_record.jsonl"),
        help="Output canonical regime-record JSONL path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional human-readable Markdown output path.",
    )
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
