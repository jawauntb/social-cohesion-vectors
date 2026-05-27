from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_activation_vector_experiment import pairwise_projection_metrics
from scripts.run_modal_activation_extraction import default_output_path


def test_pairwise_projection_metrics() -> None:
    metrics = pairwise_projection_metrics(
        pair_ids=np.array(["a", "a", "b", "b"]),
        labels=np.array(["positive", "negative", "positive", "negative"]),
        projections=np.array([2.0, 1.0, 0.0, 3.0]),
    )

    assert metrics["n_pairs"] == 2
    assert metrics["pairwise_accuracy"] == 0.5
    assert metrics["mean_projection_margin"] == -1.0


def test_default_output_path_slugs_model_id() -> None:
    path = default_output_path("Qwen/Qwen2.5-0.5B-Instruct", -1)

    assert path.name == "activation_prompts__Qwen__Qwen2.5-0.5B-Instruct__layer-1.npz"
