"""The heat-bath cluster moves against their transition kernels, enumerated exactly (issue #1142).

On a 4-site graph at q = 3 every transition is enumerable: each subset of
like edges bonded, and each label of each cluster. The kernel is built here
from the bond probability ``1 - exp(-beta J)`` and the per-cluster law
``softmax(beta sum_C h[i, :])``, sharing no code with the moves. It is checked
to keep ``exp(-beta E)`` with ``E`` from :func:`sal.sim.potts.energy`, and
each move's empirical transitions from one state are checked against the
kernel's row cell by cell. An ablation that drops the field from the label
draw is refuted by the same row, which is the evidence the row has power.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    PottsMove,
    adjacency_lists,
    heat_bath_labels,
    sample_potts,
    sweeps,
    swendsen_wang_heat_bath_sweep,
    wolff_heat_bath_sweep,
)
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

#: A triangle and a pendant: clusters of every size from 1 to 4. The
#: couplings are unequal, so a kernel that ignored ``J`` per edge would fail.
EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))
COUPLINGS = (0.8, 0.5, 1.3, 1.0)
N_STATES, BETA = 3, 0.7
DRAWS = 40_000
#: Standard errors a cell's empirical frequency may sit from the kernel's.
SIGMAS = 4.0
#: A state with a three-site cluster to split and a pendant to move.
ORIGIN = (0, 0, 0, 1)
MOVES = (PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.WOLFF_HEAT_BATH)


def _graph() -> PottsGraph:
    return PottsGraph(4, EDGES, COUPLINGS)


def _rows() -> np.ndarray:
    # A field of the couplings' scale, one row per site, from a declared seed.
    return np.random.default_rng(1142).normal(0.0, 1.5, (4, N_STATES))


def _states() -> list[tuple[int, ...]]:
    return list(itertools.product(range(N_STATES), repeat=4))


def _components(bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the 4 sites under ``bonds``, by relabelling to a fixed point."""
    component = list(range(4))
    changed = True
    while changed:
        changed = False
        for i, j in bonds:
            low = min(component[i], component[j])
            if component[i] != low or component[j] != low:
                component[i] = component[j] = low
                changed = True
    groups: dict[int, list[int]] = {}
    for site, root in enumerate(component):
        groups.setdefault(root, []).append(site)
    return list(groups.values())


def _label_law(rows: np.ndarray, cluster: list[int]) -> np.ndarray:
    weights = np.exp(BETA * rows[cluster].sum(axis=0))
    return np.asarray(weights / weights.sum())


def _kernel(move: PottsMove, rows: np.ndarray) -> np.ndarray:
    """The exact ``(81, 81)`` transition matrix of ``move``."""
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
            # Swendsen-Wang relabels every cluster; Wolff relabels the seed's,
            # the seed uniform over the sites, and a Wolff cluster grown
            # through like edges is the seed's component under independent
            # bonds on all of them.
            if move is PottsMove.SWENDSEN_WANG_HEAT_BATH:
                relabelled, share = [clusters], 1.0
            else:
                relabelled, share = [[c] for c in clusters for _ in c], 0.25
            for moved in relabelled:
                laws = [_label_law(rows, cluster) for cluster in moved]
                for labels in itertools.product(range(N_STATES), repeat=len(moved)):
                    target = list(state)
                    for cluster, label in zip(moved, labels, strict=True):
                        for site in cluster:
                            target[site] = label
                    probability = float(
                        np.prod(
                            [
                                law[label]
                                for law, label in zip(laws, labels, strict=True)
                            ]
                        )
                    )
                    kernel[index[state], index[tuple(target)]] += (
                        share * weight * probability
                    )
    return kernel


def _boltzmann(rows: np.ndarray) -> np.ndarray:
    graph = _graph()
    weights = np.array(
        [np.exp(-BETA * energy(graph, rows, np.array(state))) for state in _states()]
    )
    return np.asarray(weights / weights.sum())


def _empirical_row(move: PottsMove, rows: np.ndarray, seed: int) -> np.ndarray:
    """Frequencies of :data:`DRAWS` single steps of ``move`` from :data:`ORIGIN`."""
    graph = _graph()
    offsets, neighbours, couplings = graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    index = {state: k for k, state in enumerate(_states())}
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(index))
    for _ in range(DRAWS):
        state = np.array(ORIGIN, dtype=np.int64)
        if move is PottsMove.SWENDSEN_WANG_HEAT_BATH:
            swendsen_wang_heat_bath_sweep(
                state, graph, rows, rng, BETA, backend=Backend.PYTHON
            )
        else:
            wolff_heat_bath_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                beta=BETA,
                lists=lists,
            )
        counts[index[tuple(state.tolist())]] += 1
    return counts / DRAWS


def _outside(empirical: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Cells further than :data:`SIGMAS` binomial standard errors from the kernel's row."""
    error = np.sqrt(exact * (1.0 - exact) / DRAWS)
    return np.flatnonzero(np.abs(empirical - exact) > SIGMAS * error)


@pytest.mark.oracle
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_the_enumerated_kernel_keeps_the_boltzmann_law(move: PottsMove) -> None:
    # Rows sum to one, pi K = pi and pi_s K_st = pi_t K_ts, each to 1e-12:
    # the move is exact at this beta, with no accept step.
    rows = _rows()
    kernel = _kernel(move, rows)
    pi = _boltzmann(rows)
    flow = pi[:, None] * kernel

    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(pi @ kernel, pi, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(flow, flow.T, rtol=0.0, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_each_move_draws_its_enumerated_kernel_row(move: PottsMove) -> None:
    # 40,000 steps from (0, 0, 0, 1), every one of the 81 cells within four
    # binomial standard errors of the kernel's row; a cell the kernel never
    # reaches is never reached.
    rows = _rows()
    index = {state: k for k, state in enumerate(_states())}
    exact = _kernel(move, rows)[index[ORIGIN]]

    empirical = _empirical_row(move, rows, seed=11)

    assert _outside(empirical, exact).size == 0


@pytest.mark.oracle
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_a_label_drawn_without_the_field_is_refuted(
    move: PottsMove, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A uniform label per cluster, the field dropped: the row the test above
    # admits is left by cells at more than four standard errors.
    def uniform(log_weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return heat_bath_labels(np.zeros_like(log_weights), rng)

    monkeypatch.setattr(sweeps, "heat_bath_labels", uniform)
    rows = _rows()
    index = {state: k for k, state in enumerate(_states())}
    exact = _kernel(move, rows)[index[ORIGIN]]

    empirical = _empirical_row(move, rows, seed=11)

    assert _outside(empirical, exact).size > 0


@pytest.mark.oracle
def test_the_heat_bath_draws_the_softmax() -> None:
    # Gumbel-max against the closed form softmax(w): 200,000 draws, each
    # label's frequency within four binomial standard errors.
    log_weights = np.array([[0.0, 0.8, -1.6, 0.4]])
    draws = heat_bath_labels(
        np.repeat(log_weights, 200_000, axis=0), np.random.default_rng(0)
    )
    law = np.exp(log_weights[0]) / np.exp(log_weights[0]).sum()
    frequency = np.bincount(draws, minlength=4) / draws.size

    error = np.sqrt(law * (1.0 - law) / draws.size)
    assert np.all(np.abs(frequency - law) < SIGMAS * error)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_compiled_union_find_is_the_same_heat_bath_chain() -> None:
    # The pass differs between backends in the union-find alone, whose roots
    # `test_bond_roots.py` pins bitwise, so the chains are equal.
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 0.8)
    field = np.random.default_rng(7).normal(0.0, 0.6, (graph.n_nodes, N_STATES))
    runs = [
        sample_potts(
            graph,
            field,
            PottsMove.SWENDSEN_WANG_HEAT_BATH,
            np.random.default_rng(1142),
            200,
            cluster_backend=backend,
        ).states
        for backend in (Backend.PYTHON, Backend.RUST)
    ]

    assert np.array_equal(runs[0], runs[1])
