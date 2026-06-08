from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from social_cohesion_vectors.experiments.activation_metadata_transfer import (
    render_activation_metadata_transfer_markdown,
    run_activation_metadata_transfer,
)
from social_cohesion_vectors.schemas import PairwiseExample


def test_activation_metadata_transfer_holds_out_pair_groups(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez(
        activation_path,
        activations=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.1],
                [0.0, 0.1],
                [2.0, 0.2],
                [0.0, 0.2],
            ],
            dtype=np.float64,
        ),
        pair_ids=np.asarray(["p1", "p1", "p2", "p2", "p3", "p3"], dtype=str),
        labels=np.asarray(
            ["positive", "negative", "positive", "negative", "positive", "negative"],
            dtype=str,
        ),
    )
    pairs = [
        _pair(
            "p1",
            "a",
            fault_classes="a,shared_fault",
            source="generated_fault_class_offline",
            provider="offline",
            generated_style="cue_balanced",
            cohort="seed",
        ),
        _pair(
            "p2",
            "b",
            fault_classes="b,shared_fault",
            source="generated_fault_class_anthropic",
            provider="anthropic",
            generated_style="api_authored",
            cohort="api",
        ),
        _pair(
            "p3",
            "b",
            fault_classes="b",
            source="generated_fault_class_anthropic",
            provider="anthropic",
            generated_style="api_authored",
            cohort="api",
        ),
    ]

    report = run_activation_metadata_transfer(
        activation_npz=activation_path,
        pairs=pairs,
        metadata_key="primary_fault_class",
        coverage_metadata_keys=["cohort"],
    )
    markdown = render_activation_metadata_transfer_markdown(report)

    assert report["summary"]["folds"] == 2
    assert report["summary"]["mean_test_accuracy"] == 1.0
    assert _coverage_row(report, "source") == {
        "metadata_key": "source",
        "pairs_with_values": 3,
        "missing_pairs": 0,
        "groups": 2,
        "values": [
            "generated_fault_class_anthropic",
            "generated_fault_class_offline",
        ],
    }
    assert _coverage_row(report, "provider")["values"] == ["anthropic", "offline"]
    assert _coverage_row(report, "fault_classes")["values"] == [
        "a",
        "b",
        "shared_fault",
    ]
    assert _coverage_row(report, "cohort")["values"] == ["api", "seed"]
    assert "Activation Metadata Transfer" in markdown
    assert "Metadata Coverage" in markdown
    assert "`generated_fault_class_anthropic`" in markdown


def _coverage_row(report: Mapping[str, Any], metadata_key: str) -> Mapping[str, Any]:
    inputs = cast(Mapping[str, Any], report["inputs"])
    coverage = cast(list[Mapping[str, Any]], inputs["metadata_coverage"])
    return next(row for row in coverage if row["metadata_key"] == metadata_key)


def _pair(
    pair_id: str,
    fault_class: str,
    **metadata: str,
) -> PairwiseExample:
    pair_metadata: dict[str, str | float] = {"primary_fault_class": fault_class}
    pair_metadata.update(metadata)
    return PairwiseExample(
        pair_id=pair_id,
        scenario_id=pair_id,
        positive_run_id=f"{pair_id}:positive",
        negative_run_id=f"{pair_id}:negative",
        positive_text="positive",
        negative_text="negative",
        positive_score=1.0,
        negative_score=0.0,
        metadata=pair_metadata,
    )
