"""Parallel tempering on cluster-move replicas against the enumerated product kernel (issue #1156).

On the 4-site, q = 3 graph of `test_potts_heat_bath_cluster.py` every
transition of a two-rung ladder is enumerable: each rung's move kernel at its
own ``beta``, then the exchange accepted on ``(beta_1 - beta_2)(E_1 - E_2)``.
The kernels are built here from the bond probability ``1 - exp(-beta J)``
and each cluster's label law --- uniform proposal with the field accept step
for Swendsen-Wang and Wolff, ``softmax(beta sum_C h)`` for their heat-bath
variants --- sharing no code with the moves. The product kernel is checked to
keep ``exp(-beta_1 E) (x) exp(-beta_2 E)``, and :func:`parallel_tempering`'s
one-step draws from a fixed pair against the kernel's row.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import PottsMove, chains, parallel_tempering
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

from tests.regression.sample.test_potts_heat_bath_cluster import (
    COUPLINGS,
    EDGES,
    N_STATES,
    _components,
    _graph,
    _rows,
    _states,
)

#: The two rungs, coldest first; their ``beta`` are 1.0 and 0.4.
TEMPERATURES = (1.0, 2.5)
BETAS = tuple(1.0 / temperature for temperature in TEMPERATURES)
#: The fixed pair a step starts from: a three-site cluster and a pendant at
#: the cold rung, two two-site blocks at the hot one.
ORIGIN = ((0, 0, 0, 1), (1, 1, 2, 2))
#: Standard errors a cell's frequency may sit from the kernel's row.
SIGMAS = 4.0
#: Cells expected fewer times than this are pooled into one cell, so a cell
#: tested alone carries a normal tail rather than a Poisson one.
POOLED_BELOW = 25.0
MOVES = (
    PottsMove.SWENDSEN_WANG,
    PottsMove.WOLFF,
    PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF_HEAT_BATH,
)
HEAT_BATH = frozenset({PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.WOLFF_HEAT_BATH})
SWENDSEN_WANG = frozenset({PottsMove.SWENDSEN_WANG, PottsMove.SWENDSEN_WANG_HEAT_BATH})


def _label_kernel(
    move: PottsMove, rows: np.ndarray, cluster: list[int], beta: float
) -> np.ndarray:
    """``(q, q)``: the cluster's label law from each current label."""
    log_weights = beta * rows[cluster].sum(axis=0)
    if move in HEAT_BATH:
        law = np.exp(log_weights - log_weights.max())
        return np.tile(law / law.sum(), (N_STATES, 1))
    # A uniform proposal over all q labels, accepted on the field difference.
    kernel = np.zeros((N_STATES, N_STATES))
    for current in range(N_STATES):
        for proposed in range(N_STATES):
            if proposed != current:
                ratio = np.exp(min(0.0, log_weights[proposed] - log_weights[current]))
                kernel[current, proposed] = ratio / N_STATES
        kernel[current, current] = 1.0 - kernel[current].sum()
    return kernel


def _kernel(move: PottsMove, rows: np.ndarray, beta: float) -> np.ndarray:
    """The exact ``(81, 81)`` transition matrix of ``move`` at ``beta``."""
    states = _states()
    index = {state: k for k, state in enumerate(states)}
    kernel = np.zeros((len(states), len(states)))
    bond = [1.0 - np.exp(-beta * coupling) for coupling in COUPLINGS]
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
            # Swendsen-Wang relabels every cluster; Wolff the seed's, the seed
            # uniform over the four sites.
            if move in SWENDSEN_WANG:
                relabelled, share = [clusters], 1.0
            else:
                relabelled, share = [[c] for c in clusters for _ in c], 0.25
            for moved in relabelled:
                laws = [
                    _label_kernel(move, rows, cluster, beta)[state[cluster[0]]]
                    for cluster in moved
                ]
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


def _energies(rows: np.ndarray) -> np.ndarray:
    graph = _graph()
    return np.array([energy(graph, rows, np.array(state)) for state in _states()])


def _boltzmann(rows: np.ndarray, beta: float) -> np.ndarray:
    weights = np.exp(-beta * _energies(rows))
    return np.asarray(weights / weights.sum())


def _exchange(rows: np.ndarray) -> np.ndarray:
    """``alpha[a, b]``: the exchange's acceptance with ``a`` at the cold rung and ``b`` at the hot."""
    levels = _energies(rows)
    log_ratio = (BETAS[0] - BETAS[1]) * (levels[:, None] - levels[None, :])
    return np.asarray(np.exp(np.minimum(0.0, log_ratio)))


def _step(
    joint: np.ndarray, kernels: tuple[np.ndarray, np.ndarray], alpha: np.ndarray
) -> np.ndarray:
    """One tempering step applied to a joint law ``(81, 81)``: each rung's move, then the exchange."""
    moved = kernels[0].T @ joint @ kernels[1]
    return np.asarray(moved * (1.0 - alpha) + moved.T * alpha.T)


def _empirical_row(move: PottsMove, rows: np.ndarray, draws: int) -> np.ndarray:
    """Frequencies of ``draws`` one-step runs from :data:`ORIGIN`, each on its own spawned generator."""
    graph = _graph()
    index = {state: k for k, state in enumerate(_states())}
    counts = np.zeros((len(index), len(index)))
    start = np.array(ORIGIN, dtype=np.int64)
    for rng in np.random.default_rng(1156).spawn(draws):
        run = parallel_tempering(
            graph,
            rows,
            TEMPERATURES,
            rng,
            1,
            move=move,
            cluster_backend=Backend.PYTHON,
            start=start,
        )
        cold, hot = (tuple(row.tolist()) for row in run.states[0])
        counts[index[cold], index[hot]] += 1
    return counts / draws


def _outside(empirical: np.ndarray, exact: np.ndarray, draws: int) -> list[str]:
    """Cells, or the pool of rare cells, further than :data:`SIGMAS` binomial errors from the row.

    A cell the kernel never reaches fails on one visit.
    """
    empirical, exact = empirical.ravel(), exact.ravel()
    alone = exact * draws >= POOLED_BELOW
    failed = [
        f"cell {cell}: {empirical[cell]:.5f} against {exact[cell]:.5f}"
        for cell in np.flatnonzero(alone)
        if abs(empirical[cell] - exact[cell])
        > SIGMAS * np.sqrt(exact[cell] * (1.0 - exact[cell]) / draws)
    ]
    pooled, pooled_exact = empirical[~alone].sum(), exact[~alone].sum()
    if abs(pooled - pooled_exact) > SIGMAS * np.sqrt(
        pooled_exact * (1.0 - pooled_exact) / draws
    ):
        failed.append(f"pool: {pooled:.5f} against {pooled_exact:.5f}")
    unreachable = np.flatnonzero((exact == 0.0) & (empirical > 0.0))
    failed.extend(f"cell {cell}: unreachable, visited" for cell in unreachable)
    return failed


def _row(move: PottsMove, rows: np.ndarray) -> np.ndarray:
    """The product kernel's row from :data:`ORIGIN`, as an ``(81, 81)`` joint law."""
    index = {state: k for k, state in enumerate(_states())}
    origin = np.zeros((len(index), len(index)))
    origin[index[ORIGIN[0]], index[ORIGIN[1]]] = 1.0
    kernels = (_kernel(move, rows, BETAS[0]), _kernel(move, rows, BETAS[1]))
    return _step(origin, kernels, _exchange(rows))


@pytest.mark.oracle
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_the_product_kernel_keeps_the_product_boltzmann_law(move: PottsMove) -> None:
    # Each rung's kernel keeps its own exp(-beta E) and the move-then-exchange
    # step keeps their product, every entry to 1e-12: exchange is exact on
    # any move set whose rungs are.
    rows = _rows()
    kernels = (_kernel(move, rows, BETAS[0]), _kernel(move, rows, BETAS[1]))
    laws = (_boltzmann(rows, BETAS[0]), _boltzmann(rows, BETAS[1]))
    joint = np.outer(*laws)

    for kernel, law in zip(kernels, laws, strict=True):
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(law @ kernel, law, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        _step(joint, kernels, _exchange(rows)), joint, rtol=0.0, atol=1e-12
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    "draws",
    [
        10_000,
        pytest.param(40_000, marks=pytest.mark.release),
    ],
)
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_one_tempering_step_draws_the_product_kernel_row(
    move: PottsMove, draws: int
) -> None:
    # `draws` one-step runs from the pair ORIGIN, each joint cell expected 25
    # times or more within four binomial standard errors of the product
    # kernel's row, the rarer cells pooled into one held to the same bound,
    # and a cell the kernel never reaches never reached. The per-PR draw count
    # keeps each case under the duration cap; the release one is the issue's.
    rows = _rows()

    failed = _outside(_empirical_row(move, rows, draws), _row(move, rows), draws)

    assert not failed, failed


@pytest.mark.oracle
def test_an_exchange_without_its_energy_term_is_refuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every exchange accepted: the row the test above admits is left by
    # cells at more than four standard errors, the evidence the row has
    # power over the exchange as well as over the moves.
    monkeypatch.setattr(chains, "swap_log_ratio", lambda *_: 0.0)
    rows = _rows()
    move = PottsMove.SWENDSEN_WANG_HEAT_BATH

    failed = _outside(_empirical_row(move, rows, 10_000), _row(move, rows), 10_000)

    assert failed


@pytest.mark.smoke
@pytest.mark.snapshot
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_the_default_move_is_the_single_site_chain_before_the_parameter(
    backend: Backend,
) -> None:
    # The SHA-256 of states, walkers and swap acceptances, and the best
    # energy, taken on the base branch before `move` existed (#1156): the
    # default chain is bitwise what it was, on either backend.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1156).normal(0.0, 0.5, (16, 3))

    run = parallel_tempering(
        graph,
        field,
        (0.5, 1.0, 2.0),
        np.random.default_rng(1156),
        60,
        burn_in=5,
        thin=2,
        backend=backend,
    )

    digest = hashlib.sha256(
        run.states.tobytes() + run.walkers.tobytes() + run.swap_acceptance.tobytes()
    ).hexdigest()
    assert digest == "e43f7d97b95616a1748d8a953649a8179540a150c333bdcce965a28a022d7084"
    assert run.energy == -31.870774272929193


@pytest.mark.analytic
def test_spent_charges_each_move_as_annealing_does() -> None:
    # Swendsen-Wang: one sweep's n_nodes + 2 n_edges per replica step. Wolff:
    # each cluster's sites times 1 + 2 n_edges // n_nodes, so a multiple of
    # that factor and at least one site per replica step.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    steps, replicas = 30, 3
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    per_site = 1 + 2 * len(graph.edges) // graph.n_nodes

    def spent(move: PottsMove) -> int:
        return parallel_tempering(
            graph,
            np.zeros(3),
            (0.5, 1.0, 2.0),
            np.random.default_rng(0),
            steps - 5,
            burn_in=5,
            move=move,
        ).spent

    assert spent(PottsMove.SINGLE_SITE) == steps * replicas * per_sweep
    assert spent(PottsMove.SWENDSEN_WANG) == steps * replicas * per_sweep
    wolff = spent(PottsMove.WOLFF)
    assert wolff % per_site == 0
    assert steps * replicas * per_site <= wolff <= steps * replicas * per_sweep


@pytest.mark.smoke
def test_a_cluster_move_refuses_a_negative_coupling_and_start_is_one_per_rung() -> None:
    antiferromagnet = PottsGraph(3, ((0, 1), (1, 2)), (1.0, -1.0))

    with pytest.raises(ValueError, match="every coupling >= 0"):
        parallel_tempering(
            antiferromagnet,
            np.zeros(2),
            TEMPERATURES,
            np.random.default_rng(0),
            5,
            move=PottsMove.WOLFF,
        )
    with pytest.raises(ValueError, match="one labelling per rung"):
        parallel_tempering(
            _graph(),
            _rows(),
            TEMPERATURES,
            np.random.default_rng(0),
            5,
            start=np.zeros((3, 4), dtype=np.int64),
        )
