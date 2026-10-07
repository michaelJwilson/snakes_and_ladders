"""The exact square-lattice results against enumeration and against each other (issue #1276).

Kaufman's torus energy is checked against every configuration of the 3x3
and 4x4 tori, summed by :func:`sal.sim.potts.energies`, which shares no
code with it. Onsager's infinite-lattice energy is checked as Kaufman's
large-``L`` limit, and Baxter's critical energy as Onsager's value at
``q = 2``: three exact results from three papers, each a referee for the
others.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import (
    baxter_critical_energy,
    critical_coupling,
    energies,
    kaufman_energy,
    onsager_energy,
    yang_magnetization,
)

#: Couplings as fractions of the transition's, either side of it and on it.
FRACTIONS = (0.5, 0.8, 1.0, 1.2, 2.0)
#: Enumeration and the closed form are both double-precision sums of at most
#: 65,536 terms; agreement is required to this relative error.
ENUMERATION_TOLERANCE = 1e-12


def _enumerated_energy(coupling: float, side: int) -> float:
    """The Boltzmann mean of ``energies / N`` over every ``q = 2`` labelling of the torus."""
    graph = lattice_graph((side, side), BoundaryCondition.PERIODIC, coupling)
    states = np.array(list(itertools.product((0, 1), repeat=side * side)))
    values = energies(graph, np.zeros(2), states)
    weights = np.exp(-(values - values.min()))
    return float((weights * values).sum() / weights.sum()) / (side * side)


@pytest.mark.oracle
@pytest.mark.parametrize("side", [3, 4])
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_kaufman_energy_is_the_enumerated_energy_of_the_torus(
    side: int, fraction: float
) -> None:
    coupling = fraction * critical_coupling(2)
    exact = _enumerated_energy(coupling, side)

    assert kaufman_energy(coupling, side) == pytest.approx(
        exact, rel=ENUMERATION_TOLERANCE
    )


@pytest.mark.analytic
@pytest.mark.parametrize("fraction", [0.8, 1.2])
def test_kaufman_energy_reaches_onsager_exponentially_away_from_the_transition(
    fraction: float,
) -> None:
    # The correlation length is under 3 sites at both couplings, so at L = 64
    # the torus differs from the plane by about exp(-20); measured 3.2e-12 at
    # 0.8 and 0 at 1.2.
    coupling = fraction * critical_coupling(2)

    assert kaufman_energy(coupling, 64) == pytest.approx(
        onsager_energy(coupling), abs=1e-10
    )


@pytest.mark.analytic
def test_kaufman_energy_at_the_transition_approaches_onsager_as_one_over_l() -> None:
    # The energy's finite-size exponent at q = 2 is d - y_t = 1, so L times
    # the gap is one amplitude at every size; measured 0.27428 to 0.27430.
    coupling = critical_coupling(2)
    scaled = [
        side * (kaufman_energy(coupling, side) - onsager_energy(coupling))
        for side in (64, 128, 256)
    ]

    assert scaled == pytest.approx([scaled[-1]] * 3, rel=1e-3)
    assert scaled[-1] < 0.0


@pytest.mark.analytic
def test_baxter_critical_energy_at_two_states_is_onsager_at_the_transition() -> None:
    assert baxter_critical_energy(2) == pytest.approx(
        onsager_energy(critical_coupling(2)), rel=1e-12
    )


@pytest.mark.analytic
def test_baxter_critical_like_fraction_falls_with_the_state_count() -> None:
    # (1 + 1/sqrt q) / 2 of bonds are like at the transition, decreasing in q.
    fractions = [
        baxter_critical_energy(q) / (-2.0 * critical_coupling(q)) for q in (2, 3, 4)
    ]

    assert fractions == pytest.approx(
        [(1 + 1 / math.sqrt(q)) / 2 for q in (2, 3, 4)], rel=1e-12
    )
    assert fractions[0] > fractions[1] > fractions[2]


@pytest.mark.analytic
def test_yang_magnetization_vanishes_at_the_transition_and_saturates_far_above() -> (
    None
):
    critical = critical_coupling(2)

    assert yang_magnetization(0.8 * critical) == 0.0
    assert yang_magnetization(critical) == 0.0
    assert (
        0.0 < yang_magnetization(1.01 * critical) < yang_magnetization(1.2 * critical)
    )
    assert yang_magnetization(20.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.smoke
def test_exact_results_refuse_what_they_do_not_cover() -> None:
    with pytest.raises(ValueError, match="continuous"):
        baxter_critical_energy(5)
    with pytest.raises(ValueError, match="side"):
        kaufman_energy(1.0, 2)
    with pytest.raises(ValueError, match="positive"):
        onsager_energy(0.0)
