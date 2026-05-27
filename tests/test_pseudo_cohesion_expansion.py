from __future__ import annotations

import pytest

from social_cohesion_vectors.experiments.pseudo_cohesion import (
    PseudoCohesionExample,
    pairwise_examples_from_pseudo_cohesion,
)
from social_cohesion_vectors.experiments.pseudo_cohesion_expansion import (
    CLEAN_VARIANTS,
    DEFAULT_VARIANTS,
    expanded_examples,
    select_variants,
    variant_names,
    variant_set_names,
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
    assert variant_set_names() == ("wrapped", "clean", "all")
    assert variant_names() == (
        "meeting_note",
        "facilitator_script",
        "policy_update",
        "clean_agency_terms",
        "clean_truth_terms",
        "clean_group_terms",
    )
    assert [variant.name for variant in select_variants(["policy_update"])] == [
        "policy_update"
    ]
    assert [variant.name for variant in select_variants(None)] == [
        "meeting_note",
        "facilitator_script",
        "policy_update",
    ]
    assert [variant.name for variant in select_variants(None, variant_set="clean")] == [
        "clean_agency_terms",
        "clean_truth_terms",
        "clean_group_terms",
    ]

    with pytest.raises(ValueError, match="Unknown expansion variant"):
        select_variants(["unknown"])


def test_clean_variants_rewrite_text_without_wrappers() -> None:
    seed = [
        PseudoCohesionExample(
            example_id="pseudo_pressure",
            label="pseudo_cohesion",
            category="coercive",
            contrast_id="consent",
            text=(
                "The group needs consent now. You must comply with the mutual-aid "
                "plan and treat refusal as a threat to unity."
            ),
            expected_signal="Uses unity language to pressure compliance.",
        ),
        PseudoCohesionExample(
            example_id="genuine_choice",
            label="genuine_cohesion",
            category="autonomy",
            contrast_id="consent",
            text="The group can preserve choice and let people opt out.",
            expected_signal="Preserves refusal rights.",
        ),
    ]

    expanded = expanded_examples(
        seed,
        variants=CLEAN_VARIANTS[:1],
        include_seed=False,
    )

    assert len(expanded) == 2
    assert expanded[0].contrast_id == "consent__clean_agency_terms"
    assert expanded[0].text == (
        "The group needs agreement now. You have to go along with the mutual "
        "aid plan and treat saying no as a threat to unity."
    )
    assert not expanded[0].text.startswith("Meeting note")
