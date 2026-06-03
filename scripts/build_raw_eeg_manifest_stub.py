"""Build a metadata-only raw EEG bridge manifest from local stubs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.raw_eeg_manifest import (
    DEFAULT_DATASET,
    export_raw_eeg_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    count = export_raw_eeg_manifest(
        dataset_access_path=args.dataset_access,
        stimuli_path=args.stimuli,
        features_path=args.features,
        responses_path=args.responses,
        splits_path=args.splits,
        output_path=args.output,
        dataset=args.dataset,
    )
    print(
        "exported raw EEG metadata manifest: "
        f"dataset={args.dataset} rows={count} output={args.output}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description=(
            "Build a THINGS-EEG2-first metadata-only manifest from CSV/TSV/JSONL "
            "stubs. The command reads metadata paths only, not raw EEG arrays."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-access", required=True, type=Path)
    parser.add_argument("--stimuli", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.processed / "things_eeg2_raw_eeg_manifest_stub.jsonl",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
