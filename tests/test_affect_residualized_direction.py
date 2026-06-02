from __future__ import annotations

import numpy as np

from social_cohesion_vectors.activations.contrastive import load_direction
from social_cohesion_vectors.experiments.affect_residualized_direction import (
    render_affect_residualized_direction_markdown,
    train_affect_residualized_direction,
)


def test_affect_residualized_direction_is_orthogonal_to_affect_basis(
    tmp_path,
) -> None:
    pairs = []
    activations = []
    pair_ids = []
    labels = []
    affect_vectors = {
        "anger": np.array([2.0, 0.0, 0.0, 0.0]),
        "sadness": np.array([0.0, 2.0, 0.0, 0.0]),
        "fear": np.array([0.0, 0.0, 2.0, 0.0]),
        "happy": np.array([-2.0, 0.0, 0.0, 0.0]),
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

    output = tmp_path / "direction.npz"
    direction, report = train_affect_residualized_direction(
        activations=np.asarray(activations, dtype=np.float64),
        pair_ids=pair_ids,
        labels=labels,
        pairs=pairs,
        activation_npz="synthetic.npz",
        pairs_path="pairs.jsonl",
        output_path=output,
    )
    markdown = render_affect_residualized_direction_markdown(report)
    loaded = load_direction(output)

    assert report["summary"]["affect_subspace_rank"] > 0
    assert report["summary"]["max_abs_affect_basis_dot"] < 1e-10
    assert report["summary"]["residualized_pairwise_accuracy"] == 1.0
    assert report["summary"]["residualized_min_margin"] > 0.0
    assert np.allclose(loaded.direction, direction.direction)
    assert "Affect-Residualized Direction" in markdown
