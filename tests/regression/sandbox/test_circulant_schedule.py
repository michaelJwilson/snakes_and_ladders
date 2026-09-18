"""The circulant case both routes describe (issue #658).

`sandbox/CLAUDE.md`'s **superseded in capability** clause requires a conserved
route to referee a named case against the route that replaced it. Here it is
the circulant one: where the kernel *is* a circulant at a rate per step, the
rate form and the matrix form are the same chain, and this pins them bitwise.

Where the kernel is not a circulant the rate form has nothing to say, which is
why it was superseded rather than kept.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from snakes_and_ladders.sandbox.circulant_schedule import circulant_schedule
from snakes_and_ladders.sim.spatio_sequential import (
    canonical_spatio_sequential,
    circulant_transition,
)


@pytest.mark.smoke
def test_the_schedule_is_what_the_matrix_form_takes_bitwise() -> None:
    """The referee: same chain, same numbers, no tolerance.

    Each step's matrix is `circulant_transition` at that step's rate, which is
    the live function, so equality here is exact and not an agreement.
    """
    params = canonical_spatio_sequential()
    rates = np.linspace(0.55, 0.95, params.n_positions - 1)

    stack = circulant_schedule(params.n_states, rates)
    varying = replace(params, self_transition=stack)

    assert varying.transition.shape == (
        params.n_positions - 1,
        params.n_states,
        params.n_states,
    )
    for step, rate in enumerate(rates):
        assert np.array_equal(
            varying.transition[step], circulant_transition(params.n_states, float(rate))
        )


@pytest.mark.mathematical
def test_a_constant_schedule_is_the_scalar_the_params_take() -> None:
    """Where the rate does not vary, the two forms meet the scalar default."""
    params = canonical_spatio_sequential()
    steps = params.n_positions - 1

    stack = circulant_schedule(params.n_states, np.full(steps, 0.7))
    scalar = replace(params, self_transition=0.7).transition

    for step in range(steps):
        assert np.array_equal(stack[step], scalar)


@pytest.mark.edge_case
def test_a_schedule_that_is_not_one_rate_per_step_is_refused() -> None:
    """The shape the route is for, and nothing else."""
    with pytest.raises(ValueError, match="one rate per transition"):
        circulant_schedule(2, np.full((3, 2), 0.5))
    with pytest.raises(ValueError, match="strictly"):
        circulant_schedule(2, np.array([0.5, 1.5]))
