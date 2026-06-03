"""Export CK-3/CK-4 cocktail records as perturbation transition JSONL."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.transition_records import (
    load_ck_records_or_report,
    transition_records_from_ck_records,
    transition_records_from_ck_report,
    write_transition_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records, report = load_ck_records_or_report(args.input)
    if report:
        transitions = transition_records_from_ck_report(
            report,
            baseline_recipe_id=args.baseline_recipe_id,
            source_id=str(args.input),
        )
    else:
        transitions = transition_records_from_ck_records(
            records,
            baseline_recipe_id=args.baseline_recipe_id,
            replication_context={"source_id": str(args.input)},
        )
    count = write_transition_records(transitions, args.output)
    print(f"wrote {args.output} transitions={count}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="CK cocktail report JSON, record-list JSON, or generation JSONL input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/ck3_transition_records.jsonl"),
        help="Output transition-record JSONL path.",
    )
    parser.add_argument("--baseline-recipe-id", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
