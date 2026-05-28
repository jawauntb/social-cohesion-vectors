from __future__ import annotations

from social_cohesion_vectors.experiments.fault_taxonomy import (
    annotation_for_contrast,
    base_contrast_id,
    summarize_pseudo_report_by_fault_class,
    summarize_sae_report_by_fault_class,
    taxonomy_summary,
)
from social_cohesion_vectors.experiments.pseudo_cohesion import (
    default_examples,
    run_experiment,
)


def test_default_pseudo_cohesion_contrasts_have_fault_annotations() -> None:
    contrast_ids = {example.contrast_id for example in default_examples()}
    summary = taxonomy_summary(contrast_ids)

    assert len(contrast_ids) == 30
    assert summary["annotated_contrasts"] == 30
    assert summary["missing_contrasts"] == []
    assert summary["fault_class_counts"]["consent_bypass"] >= 4
    assert summary["guardrail_failure_counts"]["truth"] >= 6


def test_base_contrast_id_normalizes_expanded_pair_and_sample_ids() -> None:
    assert (
        base_contrast_id(
            "pseudo-cohesion::privacy_after_incident__clean_truth_terms:positive"
        )
        == "privacy_after_incident"
    )
    annotation = annotation_for_contrast(
        "pseudo-cohesion::support_exit_rights__meeting_note"
    )
    assert annotation is not None
    assert annotation.contrast_id == "support_exit_rights"


def test_pseudo_report_can_be_grouped_by_fault_class() -> None:
    report = run_experiment(lexical_score_fn=None)
    grouped = summarize_pseudo_report_by_fault_class(report)
    rows = {row["fault_class"]: row for row in grouped["rows"]}

    assert rows["consent_bypass"]["contrasts"] >= 4
    assert rows["privacy_bypass"]["contrasts"] == 3
    assert rows["truth_suppression"]["mean_scorer_margin_genuine_minus_pseudo"] <= 1


def test_sae_report_groups_pair_deltas_and_failures_by_fault_class() -> None:
    sae_report = {
        "prompts_path": "data/training/pseudo_cohesion_clean_activation_prompts.jsonl",
        "model_id": "gpt2-small",
        "release": "gpt2-small-resid-post-v5-32k",
        "sae_id": "blocks.11.hook_resid_post",
        "feature_reports": [
            {
                "feature": 3056,
                "pair_deltas": [
                    {
                        "pair_id": (
                            "pseudo-cohesion::privacy_after_incident"
                            "__clean_truth_terms"
                        ),
                        "mean_delta_positive_minus_negative": 1.25,
                    },
                    {
                        "pair_id": "pseudo-cohesion::trust_rebuild",
                        "mean_delta_positive_minus_negative": -0.5,
                    },
                ],
            }
        ],
        "transfer_evaluation": {
            "metrics": [
                {
                    "activation_metric": "mean_activation",
                    "ensemble": {
                        "failures": [
                            {
                                "pair_id": (
                                    "pseudo-cohesion::privacy_after_incident"
                                    "__clean_truth_terms"
                                )
                            }
                        ]
                    },
                    "single_features": [
                        {
                            "feature": 3056,
                            "failures": [
                                {"pair_id": "pseudo-cohesion::trust_rebuild"}
                            ],
                        }
                    ],
                }
            ]
        },
    }

    grouped = summarize_sae_report_by_fault_class(sae_report)
    deltas = {
        (row["feature"], row["fault_class"]): row
        for row in grouped["feature_fault_deltas"]
    }
    failures = {
        (row["features"], row["fault_class"]): row
        for row in grouped["transfer_failures"]
    }

    assert deltas[(3056, "privacy_bypass")]["mean_delta_genuine_minus_pseudo"] == 1.25
    assert deltas[(3056, "truth_suppression")]["mean_delta_genuine_minus_pseudo"] == -0.5
    assert failures[("ensemble", "privacy_bypass")]["failures"] == 1
    assert failures[("3056", "verification_blocking")]["failures"] == 1
