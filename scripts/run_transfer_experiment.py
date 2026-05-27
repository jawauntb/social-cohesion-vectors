"""Run local transfer/cross-validation experiments over pairwise probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.experiments.transfer import (
    find_activation_npz,
    run_transfer_from_files,
    save_transfer_reports,
)


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-runs",
        type=Path,
        default=config.paths.processed / "scored_runs.jsonl",
    )
    parser.add_argument(
        "--generated-scored-runs",
        type=Path,
        default=config.paths.processed / "generated_scored_runs.jsonl",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=config.paths.training / "pairwise_probe_dataset.jsonl",
    )
    parser.add_argument(
        "--generated-pairs",
        type=Path,
        default=None,
        help=(
            "Optional generated benchmark pairs. When provided, also reports "
            "scripted-to-generated and generated-to-scripted transfer."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=config.paths.scenarios / "seed_scenarios.json",
    )
    parser.add_argument(
        "--activation-npz",
        type=Path,
        default=None,
        help="Optional activation .npz. Defaults to auto-detecting data/features.",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="Do not auto-detect or evaluate activation vectors.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=config.paths.reports / "transfer_experiment.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=config.paths.reports / "transfer_experiment.md",
    )
    return parser.parse_args()


def main() -> int:
    config = get_config()
    args = parse_args()
    activation_npz = None
    if not args.skip_activation:
        activation_npz = args.activation_npz or find_activation_npz(config.paths.features)

    report = run_transfer_from_files(
        scored_runs_path=args.scored_runs,
        generated_scored_runs_path=args.generated_scored_runs,
        pairs_path=args.pairs,
        generated_pairs_path=args.generated_pairs,
        scenarios_path=args.scenarios,
        activation_npz_path=activation_npz,
    )
    save_transfer_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    print_summary(report, json_output=args.json_output, markdown_output=args.markdown_output)
    return 0


def print_summary(
    report: dict[str, object],
    *,
    json_output: Path,
    markdown_output: Path,
) -> None:
    inputs = report["inputs"]
    if not isinstance(inputs, dict):
        raise TypeError("report is missing inputs")
    text_transfer = report["text_transfer"]
    if not isinstance(text_transfer, dict):
        raise TypeError("report is missing text_transfer")
    summary = text_transfer.get("summary", [])
    activation = report.get("activation_transfer")

    print(
        "transfer experiment: "
        f"pairs={inputs.get('n_pairs', 0)} "
        f"generated_pairs={inputs.get('n_generated_pairs', 0)} "
        f"scenarios={inputs.get('n_scenarios', 0)} "
        f"kinds={inputs.get('n_scenario_kinds', 0)}"
    )
    for row in summary if isinstance(summary, list) else []:
        if isinstance(row, dict):
            print(
                f"{row['split']} {row['baseline']}: "
                f"test_acc={float(row['mean_test_accuracy']):.3f} "
                f"folds={row['folds']} pairs={row['test_pairs']}"
            )
    if isinstance(activation, dict):
        pair_folds = activation.get("leave_one_pair_out", [])
        scenario_folds = activation.get("leave_one_scenario_out", [])
        print(
            "activation_vector: "
            f"pair_folds={len(pair_folds) if isinstance(pair_folds, list) else 0} "
            f"scenario_folds={len(scenario_folds) if isinstance(scenario_folds, list) else 0}"
        )
    print(f"wrote {json_output}")
    print(f"wrote {markdown_output}")


if __name__ == "__main__":
    raise SystemExit(main())
