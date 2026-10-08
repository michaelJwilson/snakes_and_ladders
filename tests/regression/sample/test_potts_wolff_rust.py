"""The compiled Wolff step against the enumerated law and the oracle's cluster (issue #1362).

``oxisal.wolff_sweeps`` draws from a ChaCha8 stream seeded by one draw of the
run's generator, so its chain is of the oracle's law and is not the oracle's
chain: bitwise agreement with the Python step is not expected and is not
asserted. The referees:

* :func:`tests._chains.enumerated_law` on #1322's 2x3 instance (a per-site
  field with one forbidden label), against which both recolourings, alone and
  composed with a Gibbs sweep, are fitted by chi-square on the Rust route, at
  the significance and pooling `test_potts_recolour.py` declares;
* a cluster fixed by construction --- a bond probability of one, so the
  cluster is the seed's like component --- on which the two routes' sizes,
  :func:`step_visits` and an anneal's ``spent`` must agree exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    PottsMove,
    Recolour,
    anneal_potts,
    move_set,
    sample_potts,
    wolff_heat_bath_sweep,
    wolff_sweep,
)
from sal.sample.potts_mcmc.chains import step_visits
from sal.sample.schedule import ramp
from sal.sim.graph import BoundaryCondition, lattice_graph

from tests._chains import cell_counts, enumerated_law
from tests.regression.sample.test_potts_recolour import (
    FIELD,
    RECORDED,
    SIGNIFICANCE,
    THINNING,
    _graph,
    _pooled_p_value,
)

SEED = 1362
#: ``1 - exp(-50)`` is 1.0 in ``float64`` on both routes, so every like
#: edge reached bonds and the cluster is fixed by the state.
CERTAIN = 50.0


@pytest.mark.oracle
@pytest.mark.parametrize(
    "moves",
    [
        (PottsMove.WOLFF,),
        (PottsMove.WOLFF, PottsMove.SINGLE_SITE),
        (PottsMove.WOLFF_HEAT_BATH,),
        (PottsMove.WOLFF_HEAT_BATH, PottsMove.SINGLE_SITE),
    ],
    ids=["uniform", "uniform+gibbs", "heat-bath", "heat-bath+gibbs"],
)
def test_the_rust_wolff_step_leaves_the_boltzmann_law_invariant(
    moves: tuple[PottsMove, ...],
) -> None:
    """Chi-square of the thinned Rust chain against the enumerated law, with no visit off its support."""
    graph = _graph()
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


@pytest.mark.analytic
@pytest.mark.parametrize("heat_bath", [False, True], ids=["uniform", "heat-bath"])
def test_a_fixed_cluster_has_the_same_size_and_visits_on_both_routes(
    heat_bath: bool,
) -> None:
    """The seed's like component, the same set on both routes: ``{0, 1, 3, 4}`` from site 0, or every site."""
    graph = lattice_graph((2, 3), BoundaryCondition.OPEN, CERTAIN)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rows = np.zeros((graph.n_nodes, 3))
    # The heat-bath step takes no root, so its state is one component.
    start = [0] * 6 if heat_bath else [0, 0, 1, 0, 0, 1]
    expected = 6 if heat_bath else 4
    sizes = {}
    for backend in (Backend.PYTHON, Backend.RUST):
        state = np.array(start, dtype=np.int64)
        rng = np.random.default_rng(SEED)
        sizes[backend] = (
            wolff_heat_bath_sweep(
                state, rows, offsets, neighbours, couplings, rng, backend=backend
            )
            if heat_bath
            else wolff_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                root=0,
                backend=backend,
            )
        )
    assert sizes[Backend.PYTHON] == sizes[Backend.RUST] == expected
    visits = {each: step_visits(PottsMove.WOLFF, graph, n) for each, n in sizes.items()}
    # Each member read with its neighbours and written: 1 + 2 * 7 // 6 = 3 per site.
    assert visits[Backend.PYTHON] == visits[Backend.RUST] == expected * 3


@pytest.mark.analytic
@pytest.mark.parametrize("move", [PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH])
def test_an_anneal_spends_the_same_on_both_routes_when_the_cluster_is_fixed(
    move: PottsMove,
) -> None:
    """A uniform start and certain bonds make every cluster the whole lattice, so ``spent`` is exact."""
    graph = lattice_graph((2, 3), BoundaryCondition.OPEN, CERTAIN)
    n_steps = 12
    spent = {
        backend: anneal_potts(
            graph,
            np.zeros(3),
            ramp.linear(2.0, 0.5, n_steps),
            np.random.default_rng(SEED),
            move=move,
            recolour=Recolour.UNIFORM,
            cluster_backend=backend,
            start=np.zeros(graph.n_nodes, dtype=np.int64),
        ).spent
        for backend in (Backend.PYTHON, Backend.RUST)
    }
    # Every move of the set charged as its cluster spans the six sites.
    per_step = sum(
        step_visits(each, graph, graph.n_nodes)
        for each in move_set(move, Recolour.UNIFORM)
    )
    assert spent[Backend.PYTHON] == spent[Backend.RUST] == n_steps * per_step
