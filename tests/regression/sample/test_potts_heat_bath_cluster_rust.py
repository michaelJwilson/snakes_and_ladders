"""The compiled heat-bath Swendsen-Wang pass against the enumerated law (issue #1364).

``oxisal.swendsen_wang_heat_bath_sweeps`` draws from a ChaCha8 stream seeded
by one draw of the run's generator, so its chain is of the oracle's law and is
not the oracle's chain: bitwise agreement with the NumPy pass is not expected
and is not asserted. The referees:

* :func:`tests._chains.enumerated_law` on #1322's 2x3 instance (a per-site
  field with label 2 forbidden at site 0), against which the pass, alone and
  composed with a Gibbs sweep, is fitted by chi-square on the Rust route, at
  the significance and pooling `test_potts_recolour.py` declares;
* the exact one-step transition row of `test_potts_heat_bath_cluster.py`'s
  4-site kernel, built from the bond probability and the per-cluster softmax
  and sharing no code with the kernel, cell by cell;
* certain bonds and a field allowing one label, which fix the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    PottsMove,
    sample_potts,
    swendsen_wang_heat_bath_sweep,
)
from sal.sample.potts_mcmc.sweeps import _heat_bath_pass_rust
from sal.sim.graph import BoundaryCondition, lattice_graph

from tests._chains import cell_counts, enumerated_law
from tests.regression.sample.test_potts_heat_bath_cluster import (
    BETA,
    DRAWS,
    ORIGIN,
    _graph,
    _kernel,
    _outside,
    _rows,
    _states,
)
from tests.regression.sample.test_potts_recolour import (
    FIELD,
    RECORDED,
    SIGNIFICANCE,
    THINNING,
    _pooled_p_value,
)
from tests.regression.sample.test_potts_recolour import _graph as _lattice

SEED = 1364
#: ``1 - exp(-50)`` is 1.0 in ``float64``, so every like edge bonds.
CERTAIN = 50.0


@pytest.mark.oracle
@pytest.mark.parametrize(
    "moves",
    [
        (PottsMove.SWENDSEN_WANG_HEAT_BATH,),
        (PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.SINGLE_SITE),
    ],
    ids=["heat-bath", "heat-bath+gibbs"],
)
def test_the_rust_pass_leaves_the_boltzmann_law_invariant(
    moves: tuple[PottsMove, ...],
) -> None:
    """Chi-square of the thinned Rust chain against the enumerated law, with no visit off its support."""
    graph = _lattice()
    index, probability = enumerated_law(graph, FIELD)
    chain = sample_potts(
        graph,
        FIELD,
        moves,
        np.random.default_rng(SEED),
        RECORDED,
        burn_in=RECORDED // 10,
        thin=THINNING,
        cluster_backend=Backend.RUST,
    )
    counts = cell_counts(index, chain.states)
    support = probability > 0
    assert counts[~support].sum() == 0
    assert _pooled_p_value(probability[support], counts[support]) > SIGNIFICANCE


@pytest.mark.oracle
def test_the_rust_pass_draws_the_enumerated_kernel_row() -> None:
    """40,000 single passes from ``(0, 0, 0, 1)``, every one of 81 cells within four standard errors."""
    graph, rows = _graph(), _rows()
    index = {state: k for k, state in enumerate(_states())}
    exact = _kernel(PottsMove.SWENDSEN_WANG_HEAT_BATH, rows)[index[ORIGIN]]
    rng = np.random.default_rng(SEED)
    counts = np.zeros(len(index))
    for _ in range(DRAWS):
        state = np.array(ORIGIN, dtype=np.int64)
        swendsen_wang_heat_bath_sweep(state, graph, rows, rng, BETA)
        counts[index[tuple(state.tolist())]] += 1

    assert _outside(counts / DRAWS, exact).size == 0


@pytest.mark.analytic
def test_certain_bonds_move_the_whole_lattice_to_the_one_allowed_label() -> None:
    """A uniform start is one cluster under certain bonds; a batch of five keeps it whole."""
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, CERTAIN)
    rows = np.zeros((graph.n_nodes, 3))
    rows[:, :2] = -np.inf
    state = np.zeros(graph.n_nodes, dtype=np.int64)

    n_clusters = _heat_bath_pass_rust(
        state, graph, rows, np.random.default_rng(SEED), 1.0, n_sweeps=5
    )

    assert n_clusters.tolist() == [1] * 5
    assert state.tolist() == [2] * graph.n_nodes


@pytest.mark.analytic
def test_a_negative_coupling_is_refused_by_the_kernel() -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, -0.5)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    with pytest.raises(ValueError, match="negative"):
        swendsen_wang_heat_bath_sweep(
            state, graph, np.zeros((4, 3)), np.random.default_rng(0)
        )


@pytest.mark.oracle
def test_a_pass_at_the_wrong_temperature_is_refuted_by_the_row() -> None:
    """The same row, the pass run at half :data:`BETA`: cells leave four standard errors, so the row has power."""
    graph, rows = _graph(), _rows()
    index = {state: k for k, state in enumerate(_states())}
    exact = _kernel(PottsMove.SWENDSEN_WANG_HEAT_BATH, rows)[index[ORIGIN]]
    rng = np.random.default_rng(SEED)
    counts = np.zeros(len(index))
    for _ in range(DRAWS):
        state = np.array(ORIGIN, dtype=np.int64)
        swendsen_wang_heat_bath_sweep(state, graph, rows, rng, BETA / 2)
        counts[index[tuple(state.tolist())]] += 1

    assert _outside(counts / DRAWS, exact).size > 0
