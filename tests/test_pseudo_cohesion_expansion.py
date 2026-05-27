from __future__ import annotations

import pytest

from social_cohesion_vectors.experiments.pseudo_cohesion import (
    PseudoCohesionExample,
    pairwise_examples_from_pseudo_cohesion,
)
from social_cohesion_vectors.experiments.pseudo_cohesion_expansion import (
    DEFAULT_VARIANTS,
    expanded_examples,
    select_variants,
    variant_names,
)


def test_expanded_examples_add_neutral_variant_pairs() -> None:
    seed = [
        PseudoCohesionExample(
            example_id="pseudo_pressure",
            label="pseudo_cohesion",
            category="coercive",
            contrast_id="consent",
            text="We should stay united, so you need to comply.",
            expected_signal="Uses unity language to pressure compliance.",
        ),
        PseudoCohesionExample(
            example_id="genuine_choice",
            label="genuine_cohesion",
            category="autonomy",
            contrast_id="consent",
            text="You can choose freely, and no one should punish refusal.",
            expected_signal="Preserves refusal rights.",
        ),
    ]

    expanded = expanded_examples(seed, variants=DEFAULT_VARIANTS[:2])
    pairs = pairwise_examples_from_pseudo_cohesion(expanded)

    assert len(expanded) == 6
    assert len(pairs) == 3
    assert {example.label for example in expanded} == {
        "pseudo_cohesion",
        "genuine_cohesion",
    }
    assert "consent__meeting_note" in {example.contrast_id for example in expanded}


def test_select_variants_validates_names() -> None:
    assert variant_names() == ("meeting_note", "facilitator_script", "policy_update")
    assert [variant.name for variant in select_variants(["policy_update"])] == [
        "policy_update"
    ]

    with pytest.raises(ValueError, match="Unknown expansion variant"):
        select_variants(["unknown"])
