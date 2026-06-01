from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_affect_activation_residualization.py"
    )
    spec = importlib.util.spec_from_file_location(
        "affect_activation_residualization",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_affect_activation_residualization_preserves_non_affect_signal() -> None:
    pairs = []
    activations = []
    pair_ids = []
    labels = []
    affect_vectors = {
        "anger": np.array([1.0, 0.0, 0.0, 0.0]),
        "sadness": np.array([0.0, 1.0, 0.0, 0.0]),
        "fear": np.array([0.0, 0.0, 1.0, 0.0]),
        "happy": np.array([-1.0, 0.0, 0.0, 0.0]),
    }
    cohesion = np.array([0.0, 0.0, 0.0, 1.0])
    for index, affect_label in enumerate(("anger", "sadness", "fear", "happy")):
        pair_id = f"pair-{index}"
        pairs.append(
            {
                "pair_id": pair_id,
                "metadata": {
                    "affect_label": affect_label,
                    "mechanism": "consent",
                    "negative_pole": "rigid_boundary_reification",
                },
            }
        )
        base = affect_vectors[affect_label]
        activations.append(base + cohesion)
        pair_ids.append(pair_id)
        labels.append("positive")
        activations.append(base - cohesion)
        pair_ids.append(pair_id)
        labels.append("negative")

    report = SCRIPT.run_affect_activation_residualization(
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=np.asarray(pair_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
        pairs=pairs,
        activation_npz="synthetic.npz",
        pairs_path="pairs.jsonl",
    )
    markdown = SCRIPT.render_markdown(report)

    assert report["summary"]["original_pair_loo_accuracy"] == 1.0
    assert report["summary"]["residualized_pair_loo_accuracy"] == 1.0
    assert report["summary"]["mean_affect_subspace_rank"] > 0.0
    assert "Affect Activation Residualization" in markdown
