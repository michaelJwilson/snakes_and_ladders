"""The Potts cluster moves on a field that forbids labels, against their enumerated kernels (issue #1146).

A forbidden label is a ``-inf`` field entry (:func:`sal.sim.potts.forbid`).
On the 4-site, q = 3 graph of `test_potts_heat_bath_cluster.py` with three
entries forbidden, every transition of Swendsen-Wang, Wolff and their
heat-bath forms is enumerated: a recolour onto a label any member forbids is
rejected, one off a forbidden label onto an allowed one is accepted, and a
heat-bath cluster forbidding every label keeps its own. Referees: each kernel
keeps ``exp(-beta E)`` restricted to the allowed labellings, never leaves
them, and carries every start, forbidden ones included, to that law, each to
1e-12; each move's empirical row from a forbidden start matches its kernel's
cell by cell with ``RuntimeWarning`` an error. On the #1139 3x3 fixture every
cluster move, Niedermayer's and both Swendsen-Wang backends included, leaves
a forbidden start without a warning and never re-enters a forbidden label.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    adjacency_lists,
    swendsen_wang_heat_bath_sweep,
    swendsen_wang_sweep,
    wolff_heat_bath_sweep,
    wolff_sweep,
)
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energy, forbid

from tests.regression.sample.test_potts_heat_bath_cluster import (
    BETA,
    COUPLINGS,
    EDGES,
    N_STATES,
    _components,
    _graph,
    _rows,
    _states,
)

#: Site 1 forbids label 0, site 2 label 1, site 3 label 2: 24 of the 81
#: labellings are allowed, and a cluster of sites 1-3 forbids every label.
ALLOWED = np.ones((4, N_STATES), dtype=bool)
ALLOWED[1, 0] = ALLOWED[2, 1] = ALLOWED[3, 2] = False
#: Forbidden at site 2. From it the whole graph proposes only forbidden
#: labels, the triangle proposing 0 is forbidden to forbidden, and the heat
#: bath of all four sites has no label to draw and keeps 1, where `argmax`
#: of an all ``-inf`` row would give 0.
ORIGIN = (1, 1, 1, 1)
DRAWS = 40_000
SIGMAS = 4.0
#: Steps after which every row of the kernel's power is the restricted law.
MIXED = 4096

Law = Callable[[np.ndarray, list[int], int], np.ndarray]


def _field() -> np.ndarray:
    return forbid(_rows(), ALLOWED)


def _metropolis(rows: np.ndarray, cluster: list[int], current: int) -> np.ndarray:
    """A uniform proposal over the q labels, accepted by the forbidden-label rule."""
    sums = BETA * rows[cluster].sum(axis=0)
    law = np.zeros(N_STATES)
    for label in range(N_STATES):
        if label == current or sums[label] == -np.inf:
            continue
        accept = (
            1.0
            if sums[current] == -np.inf
            else min(1.0, float(np.exp(sums[label] - sums[current])))
        )
        law[label] = accept / N_STATES
    law[current] = 1.0 - law.sum()
    return law


def _heat_bath(rows: np.ndarray, cluster: list[int], current: int) -> np.ndarray:
    """``softmax(beta sum_C h)``, or the cluster's own label where every weight is zero."""
    weights = np.exp(BETA * rows[cluster].sum(axis=0))
    if weights.sum() == 0.0:
        return np.asarray(np.eye(N_STATES)[current])
    return np.asarray(weights / weights.sum())


#: Each move as (its cluster law, whether every cluster is relabelled).
MOVES: dict[str, tuple[Law, bool]] = {
    "swendsen_wang": (_metropolis, True),
    "wolff": (_metropolis, False),
    "swendsen_wang_heat_bath": (_heat_bath, True),
    "wolff_heat_bath": (_heat_bath, False),
}


def _kernel(law: Law, every: bool, rows: np.ndarray) -> np.ndarray:
    """The exact ``(81, 81)`` transition matrix, as `test_potts_heat_bath_cluster._kernel` builds it."""
    states = _states()
    index = {state: k for k, state in enumerate(states)}
    kernel = np.zeros((len(states), len(states)))
    bond = [1.0 - np.exp(-BETA * coupling) for coupling in COUPLINGS]
    for state in states:
        like = [k for k, (i, j) in enumerate(EDGES) if state[i] == state[j]]
        for bonded in itertools.product((False, True), repeat=len(like)):
            weight = float(
                np.prod(
                    [
                        bond[k] if b else 1.0 - bond[k]
                        for k, b in zip(like, bonded, strict=True)
                    ]
                )
            )
            clusters = _components(
                [EDGES[k] for k, b in zip(like, bonded, strict=True) if b]
            )
            relabelled = [clusters] if every else [[c] for c in clusters for _ in c]
            share = 1.0 if every else 0.25
            for moved in relabelled:
                laws = [law(rows, cluster, state[cluster[0]]) for cluster in moved]
                for labels in itertools.product(range(N_STATES), repeat=len(moved)):
                    target = list(state)
                    for cluster, label in zip(moved, labels, strict=True):
                        for site in cluster:
                            target[site] = label
                    probability = float(
                        np.prod(
                            [p[label] for p, label in zip(laws, labels, strict=True)]
                        )
                    )
                    kernel[index[state], index[tuple(target)]] += (
                        share * weight * probability
                    )
    return kernel


def _allowed_states() -> np.ndarray:
    return np.array([ALLOWED[np.arange(4), state].all() for state in _states()])


def _restricted_boltzmann() -> np.ndarray:
    """``exp(-beta E)`` on the allowed labellings, ``E`` on the finite field, zero elsewhere."""
    graph, finite = _graph(), _rows()
    weights = np.array(
        [np.exp(-BETA * energy(graph, finite, np.array(state))) for state in _states()]
    )
    weights[~_allowed_states()] = 0.0
    return np.asarray(weights / weights.sum())


@pytest.mark.oracle
@pytest.mark.parametrize("move", sorted(MOVES))
def test_the_enumerated_kernel_keeps_the_restricted_law(move: str) -> None:
    # Rows sum to one; no allowed labelling moves to a forbidden one (exactly
    # zero); pi K = pi with pi the restricted Boltzmann law; and every row of
    # K^4096, forbidden starts included, is pi --- each to 1e-12.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        kernel = _kernel(*MOVES[move], _field())
    allowed = _allowed_states()
    pi = _restricted_boltzmann()

    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert not kernel[np.ix_(allowed, ~allowed)].any()
    np.testing.assert_allclose(pi @ kernel, pi, rtol=0.0, atol=1e-12)
    limit = np.linalg.matrix_power(kernel, MIXED)
    np.testing.assert_allclose(limit, np.tile(pi, (81, 1)), rtol=0.0, atol=1e-12)


def _step(
    move: str, backend: Backend
) -> Callable[[np.ndarray, np.random.Generator], None]:
    """One step of ``move`` on the 4-site graph at :data:`BETA`, on ``backend``."""
    graph, rows = _graph(), _field()
    offsets, neighbours, couplings = graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    adjacency = (offsets, neighbours, couplings)

    def step(state: np.ndarray, rng: np.random.Generator) -> None:
        if move == "swendsen_wang":
            swendsen_wang_sweep(state, graph, rows, rng, beta=BETA, backend=backend)
        elif move == "wolff":
            wolff_sweep(state, rows, *adjacency, rng, beta=BETA, lists=lists)
        elif move == "swendsen_wang_heat_bath":
            swendsen_wang_heat_bath_sweep(state, graph, rows, rng, BETA, backend)
        else:
            wolff_heat_bath_sweep(state, rows, *adjacency, rng, beta=BETA, lists=lists)

    return step


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("move", "backend"),
    [(move, Backend.PYTHON) for move in sorted(MOVES)]
    + [("swendsen_wang", Backend.RUST)],
    ids=str,
)
def test_each_move_draws_its_enumerated_row_from_a_forbidden_start(
    move: str, backend: Backend
) -> None:
    # 40,000 steps from (1, 1, 1, 1), RuntimeWarning an error: every one of
    # the 81 cells within four binomial standard errors of the kernel's row,
    # and a cell the kernel never reaches never reached.
    index = {state: k for k, state in enumerate(_states())}
    exact = _kernel(*MOVES[move], _field())[index[ORIGIN]]
    step = _step(move, backend)
    rng = np.random.default_rng(1146)
    counts = np.zeros(len(index))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(DRAWS):
            state = np.array(ORIGIN, dtype=np.int64)
            step(state, rng)
            counts[index[tuple(state.tolist())]] += 1
    empirical = counts / DRAWS

    error = np.sqrt(exact * (1.0 - exact) / DRAWS)
    assert np.flatnonzero(np.abs(empirical - exact) > SIGMAS * error).size == 0


#: The #1139 fixture: label 0 nowhere, label 2 on every other site.
LATTICE = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
LATTICE_ALLOWED = np.ones((LATTICE.n_nodes, 3), dtype=bool)
LATTICE_ALLOWED[:, 0] = False
LATTICE_ALLOWED[::2, 2] = False
LATTICE_FIELD = forbid(
    np.random.default_rng(1081).normal(0.0, 1.0, (LATTICE.n_nodes, 3)),
    LATTICE_ALLOWED,
)
SWEEPS = 300
LATTICE_MOVES = (
    "swendsen_wang_python",
    "swendsen_wang_rust",
    "wolff",
    "niedermayer",
    "swendsen_wang_heat_bath",
    "wolff_heat_bath",
)
