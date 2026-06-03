from __future__ import annotations

import pytest

from social_cohesion_vectors.modal_app.functions.activation_steering import (
    _scheduled_strength,
)


def test_scheduled_strength_supports_ck4_token_windows() -> None:
    assert (
        _scheduled_strength(
            0.5,
            schedule="first-4",
            timing="generate",
            state={"forward_calls": 5},
        )
        == 0.5
    )
    assert (
        _scheduled_strength(
            0.5,
            schedule="first-4",
            timing="generate",
            state={"forward_calls": 6},
        )
        == 0.0
    )
    assert (
        _scheduled_strength(
            0.35,
            schedule="after-4",
            timing="generate",
            state={"forward_calls": 5},
        )
        == 0.0
    )
    assert (
        _scheduled_strength(
            0.35,
            schedule="after-4",
            timing="generate",
            state={"forward_calls": 6},
        )
        == 0.35
    )


def test_scheduled_strength_supports_decay_and_ramp() -> None:
    assert _scheduled_strength(
        1.0,
        schedule="decay-5",
        timing="generate",
        state={"forward_calls": 2},
    ) == pytest.approx(1.0)
    assert _scheduled_strength(
        1.0,
        schedule="decay-5",
        timing="generate",
        state={"forward_calls": 4},
    ) == pytest.approx(0.5)
    assert _scheduled_strength(
        1.0,
        schedule="ramp-3-5",
        timing="generate",
        state={"forward_calls": 4},
    ) == pytest.approx(1.0 / 3.0)
    assert _scheduled_strength(
        1.0,
        schedule="ramp-3-5",
        timing="generate",
        state={"forward_calls": 6},
    ) == pytest.approx(1.0)


def test_scheduled_strength_ignores_schedule_outside_generate_timing() -> None:
    assert (
        _scheduled_strength(
            0.75,
            schedule="first-4",
            timing="prefill",
            state={"forward_calls": 1},
        )
        == 0.75
    )
