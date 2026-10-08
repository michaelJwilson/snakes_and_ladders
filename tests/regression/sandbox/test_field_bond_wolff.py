"""The field-aware Wolff bond against the enumerated Potts law (issue #1390, move mix).

:mod:`sal.sandbox.field_bond_wolff` bonds a same-label edge with
``p = 1 - exp(-beta (J - dh)_+)`` and accepts on Wolff's boundary-only
ratio. Enumerating its kernel on a 4-cycle at ``q = 3`` (81 states) shows the
J-only control keeps the Boltzmann law to rounding and the field-aware bond
does not: the inside edges' connectivity no longer cancels between a move and
its reverse. The move is declined on this test, not measured.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.sandbox import field_bond_wolff as fbw

#: A 4-cycle, J = 1, a N(0, 1) field over q = 3 labels, at beta = 1.
EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
COUPLING = np.ones(4)
ROWS = np.random.default_rng(0).normal(size=(4, 3))
BETA = 1.0


def _flux_gap(field_aware: bool) -> float:
    # Largest |pi_x T_xy - pi_y T_yx| over state pairs: zero under detailed balance.
    kernel = fbw.transition_matrix(EDGES, COUPLING, ROWS, BETA, field_aware=field_aware)
    law = fbw.boltzmann(EDGES, COUPLING, ROWS, BETA)
    flux = law[:, None] * kernel
    return float(np.max(np.abs(flux - flux.T)))


@pytest.mark.oracle
def test_the_j_only_bond_keeps_the_enumerated_law() -> None:
    # The control: Wolff's bond with a Metropolis recolour is reversible
    # against the Boltzmann law enumerated over all 81 states.
    assert _flux_gap(field_aware=False) < 1e-15


@pytest.mark.oracle
def test_the_field_aware_bond_breaks_detailed_balance() -> None:
    # The same acceptance with the field-aware bond: detailed balance fails by
    # orders of magnitude above rounding, and the kernel's stationary law is
    # not the Boltzmann law.
    assert _flux_gap(field_aware=True) > 1e-4
    kernel = fbw.transition_matrix(EDGES, COUPLING, ROWS, BETA, field_aware=True)
    law = fbw.boltzmann(EDGES, COUPLING, ROWS, BETA)
    assert 0.5 * np.abs(law @ kernel - law).sum() > 1e-4


@pytest.mark.oracle
def test_the_step_draws_from_the_enumerated_kernel() -> None:
    # The sampler the kernel describes: 20,000 steps from one state land on
    # each target at the kernel's rate, within five binomial standard errors.
    start = np.array([0, 0, 0, 1], dtype=np.int64)
    kernel = fbw.transition_matrix(EDGES, COUPLING, ROWS, BETA, field_aware=True)
    row = kernel[int(np.ravel_multi_index(tuple(start), (3,) * 4))]
    rng = np.random.default_rng(1)
    draws = 20_000
    counts = np.zeros(81)
    for _ in range(draws):
        moved = fbw.step(EDGES, COUPLING, ROWS, start, BETA, rng)
        counts[np.ravel_multi_index(tuple(moved), (3,) * 4)] += 1
    se = np.sqrt(row * (1.0 - row) / draws)
    assert np.all(np.abs(counts / draws - row) <= 5.0 * se + 1e-12)
