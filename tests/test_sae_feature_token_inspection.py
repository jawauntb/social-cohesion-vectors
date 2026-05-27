from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "inspect_gpt2_sae_feature_tokens.py"
    )
    spec = importlib.util.spec_from_file_location("sae_feature_token_inspection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_pair_delta_rows_compares_matched_prompt_means() -> None:
    rows = [
        {
            "pair_id": "pseudo-cohesion::consent",
            "label": "positive",
            "sample_id": "consent:positive",
            "mean_activation": 0.8,
            "max_activation": 1.4,
        },
        {
            "pair_id": "pseudo-cohesion::consent",
            "label": "negative",
            "sample_id": "consent:negative",
            "mean_activation": 0.3,
            "max_activation": 1.0,
        },
    ]

    deltas = SCRIPT.pair_delta_rows(rows)

    assert deltas == [
        {
            "pair_id": "pseudo-cohesion::consent",
            "positive_sample_id": "consent:positive",
            "negative_sample_id": "consent:negative",
            "mean_delta_positive_minus_negative": 0.5,
            "mean_delta_negative_minus_positive": -0.5,
            "max_delta_positive_minus_negative": 0.4,
            "positive_mean": 0.8,
            "negative_mean": 0.3,
            "positive_max": 1.4,
            "negative_max": 1.0,
        }
    ]


def test_summarize_feature_returns_token_example_and_pair_views() -> None:
    token_rows = [
        {"label": "positive", "activation": 0.5, "pair_id": "p", "token": "repair"},
        {"label": "positive", "activation": 0.0, "pair_id": "p", "token": "choice"},
        {"label": "negative", "activation": 1.0, "pair_id": "p", "token": "unity"},
        {"label": "negative", "activation": 0.5, "pair_id": "p", "token": "must"},
    ]
    example_rows = [
        {
            "label": "positive",
            "pair_id": "p",
            "sample_id": "p:positive",
            "mean_activation": 0.25,
            "max_activation": 0.5,
        },
        {
            "label": "negative",
            "pair_id": "p",
            "sample_id": "p:negative",
            "mean_activation": 0.75,
            "max_activation": 1.0,
        },
    ]

    summary = SCRIPT.summarize_feature(
        feature=24555,
        token_rows=token_rows,
        example_rows=example_rows,
        top_k_tokens=1,
        top_k_examples=1,
        top_k_pairs=1,
    )

    assert summary["summary"]["direction_hint"] == "pseudo_higher"
    assert summary["summary"]["token_mean_pos_minus_neg"] == -0.5
    assert summary["top_tokens"][0]["token"] == "unity"
    assert summary["top_negative_examples"][0]["sample_id"] == "p:negative"
    assert summary["largest_pseudo_minus_genuine_pairs"][0][
        "mean_delta_negative_minus_positive"
    ] == 0.5


def test_top_rows_and_token_context_are_stable() -> None:
    assert SCRIPT.direction_hint(0.2) == "genuine_higher"
    assert SCRIPT.direction_hint(-0.2) == "pseudo_higher"
    assert SCRIPT.direction_hint(0.0) == "tied"
    assert SCRIPT.top_rows(
        [{"score": 1}, {"score": 3}, {"score": 2}], key="score", limit=2
    ) == [
        {"score": 3},
        {"score": 2},
    ]
    assert SCRIPT.token_context(["a", "shared", "plan", "now"], 2, window=1) == (
        "shared [plan] now"
    )


def test_feature_transfer_uses_train_fold_feature_direction() -> None:
    rows = [
        _feature_row("p1", "positive", {1: 2.0, 2: 0.0}),
        _feature_row("p1", "negative", {1: 0.0, 2: 2.0}),
        _feature_row("p2", "positive", {1: 3.0, 2: 0.0}),
        _feature_row("p2", "negative", {1: 0.0, 2: 3.0}),
        _feature_row("p3", "positive", {1: 2.5, 2: 0.0}),
        _feature_row("p3", "negative", {1: 0.0, 2: 2.5}),
    ]

    result = SCRIPT.evaluate_signed_feature_loo(
        rows=rows,
        features=[1, 2],
        activation_metric="mean_activation",
    )

    assert result["pairs"] == 3
    assert result["accuracy"] == 1.0
    assert result["failures"] == []
    assert result["mean_margin"] > 0.0


def test_feature_transfer_report_combines_feature_rows_by_sample() -> None:
    report = SCRIPT.evaluate_feature_transfer(
        example_rows_by_feature={
            1: [
                {
                    "sample_id": "p1:positive",
                    "pair_id": "p1",
                    "label": "positive",
                    "mean_activation": 1.0,
                    "max_activation": 2.0,
                },
                {
                    "sample_id": "p1:negative",
                    "pair_id": "p1",
                    "label": "negative",
                    "mean_activation": 0.0,
                    "max_activation": 0.5,
                },
            ],
            2: [
                {
                    "sample_id": "p1:positive",
                    "pair_id": "p1",
                    "label": "positive",
                    "mean_activation": 0.0,
                    "max_activation": 0.5,
                },
                {
                    "sample_id": "p1:negative",
                    "pair_id": "p1",
                    "label": "negative",
                    "mean_activation": 1.0,
                    "max_activation": 2.0,
                },
            ],
        },
        features=[1, 2],
    )

    assert report["n_examples"] == 2
    assert report["n_pairs"] == 1
    assert [metric["activation_metric"] for metric in report["metrics"]] == [
        "mean_activation",
        "max_activation",
    ]


def _feature_row(
    pair_id: str,
    label: str,
    values: dict[int, float],
) -> dict[str, object]:
    return {
        "sample_id": f"{pair_id}:{label}",
        "pair_id": pair_id,
        "label": label,
        "features": {
            feature: {"mean_activation": value, "max_activation": value}
            for feature, value in values.items()
        },
    }
