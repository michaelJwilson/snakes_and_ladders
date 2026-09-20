"""Each accept arm is the line it replaced, decision for decision and draw for draw.

Issue #857 gave twelve sites one accept step. The claim a fold makes is that
nothing moved, and for a Metropolis test that claim has two halves: the same
decision, and the same number of uniforms taken from the generator in the same
order. A sampler that accepts identically while drawing one uniform more per
step returns a different chain from the next step onward, and every seeded
chain in this directory would fail --- which is the referee, one level up. This
module is the referee one level down: each arm runs beside the inline form the
site wrote, on two generators seeded alike, and the streams are read afterwards
to show they are level.

The comparison is the closed form rather than a backend or a recorded chain, so
the kind is ``analytic``.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sample.accept import (
    accept,
    accept_at,
    accept_at_temperature,
    accept_drawn,
    accept_ratio,
    accept_with,
    acceptance_probability,
)

#: Draws per arm. Enough that every branch is taken thousands of times: at a
#: log ratio of mean zero and spread two, 50% short-circuit and the rest draw.
DRAWS = 10_000


def _log_ratios() -> np.ndarray:
    """Log ratios covering every branch: negative, zero, positive and ``-inf``."""
    values = np.random.default_rng(857).normal(0.0, 2.0, DRAWS)
    values[::97] = 0.0
    values[::331] = -np.inf
    return values


def _level(first: np.random.Generator, second: np.random.Generator) -> bool:
    """Whether two generators have taken the same number of draws."""
    return bool(first.random() == second.random())


@pytest.mark.analytic
def test_every_arm_is_its_inline_form_decision_for_decision_and_draw_for_draw() -> None:
    folded, inline = np.random.default_rng(1), np.random.default_rng(1)
    for log_ratio in _log_ratios():
        expected = log_ratio >= 0.0 or inline.random() < np.exp(log_ratio)
        assert accept(log_ratio, folded) == expected
    assert _level(folded, inline)

    folded, inline = np.random.default_rng(2), np.random.default_rng(2)
    for log_ratio in _log_ratios():
        expected = log_ratio >= 0.0 or inline.random() < np.exp(log_ratio)
        assert accept_drawn(log_ratio, folded.random) == expected
    assert _level(folded, inline)

    uniforms = np.random.default_rng(3).random(DRAWS)
    for log_ratio, uniform in zip(_log_ratios(), uniforms, strict=True):
        expected = log_ratio >= 0.0 or uniform < np.exp(log_ratio)
        assert accept_with(log_ratio, float(uniform)) == expected

    # The exponentiated arm: above one, below one, and the `nan` a proposal
    # whose energy is not finite produces.
    ratios = np.exp(_log_ratios())
    ratios[::53] = np.nan
    for ratio, uniform in zip(ratios, uniforms, strict=True):
        assert accept_ratio(float(ratio), float(uniform)) == (uniform < ratio)
        assert acceptance_probability(float(ratio)) == (
            0.0 if np.isnan(ratio) else min(1.0, float(ratio))
        )

    for beta in (0.5, 2.0, np.inf):
        folded, inline = np.random.default_rng(4), np.random.default_rng(4)
        for difference in _log_ratios():
            expected = difference >= 0.0 or (
                bool(np.isfinite(beta)) and inline.random() < np.exp(beta * difference)
            )
            assert accept_at(beta, float(difference), folded) == expected
        assert _level(folded, inline)

    for temperature in (0.0, 0.7, 3.0):
        folded, inline = np.random.default_rng(5), np.random.default_rng(5)
        for gain in _log_ratios():
            if gain >= 0.0:
                expected = True
            elif temperature == 0.0:
                expected = False
            else:
                expected = bool(inline.random() < np.exp(gain / temperature))
            assert accept_at_temperature(temperature, float(gain), folded.random) is (
                expected
            )
        assert _level(folded, inline)


@pytest.mark.smoke
def test_the_zero_temperature_limit_is_taken_without_a_draw() -> None:
    """``beta = inf`` refuses a step down and takes a step up, drawing nothing.

    The limit rather than its neighbourhood: a negative difference at infinite
    beta is refused exactly, not accepted with a probability that underflowed,
    and a generator that is never touched says so.
    """
    rng = np.random.default_rng(6)
    before = rng.bit_generator.state

    assert not accept_at(np.inf, -1e-12, rng)
    assert accept_at(np.inf, 0.0, rng)
    assert accept_at(np.inf, 3.0, rng)
    assert not accept_at_temperature(0.0, -1e-12, rng.random)
    assert accept_at_temperature(0.0, 0.0, rng.random)

    assert rng.bit_generator.state == before
