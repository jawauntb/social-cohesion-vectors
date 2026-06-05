"""Normalize regime-transition records into canonical JSONL."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.regime_records import (
    load_regime_transition_records,
    summarize_regime_transition_records,
    write_regime_transition_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_regime_transition_records(args.input)
    count = write_regime_transition_records(records, args.output)
    summary = summarize_regime_transition_records(records)
    print(
        f"wrote {args.output} records={count} "
        f"status={summary['status']} "
        f"gate_status={summary['gate_status']} "
        f"residual_findings={summary['residual_findings']}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Regime-transition record JSON, record-list JSON, or JSONL input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/regime_transition_records.jsonl"),
        help="Output canonical JSONL path.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
