"""Parallel tempering with per-rung and mixed moves against the enumerated product kernel (issue #1158).

On the 4-site, q = 3 graph of `test_potts_heat_bath_cluster.py` each rung's
kernel is the product, in order, of its moves' kernels at its own ``beta``:
heat-bath Swendsen-Wang from `test_potts_tempering_cluster.py`, and here the
single-site heat-bath sweep (sites 0 to 3 in turn, each from its exact
conditional) and the ghost-spin pass (like edges bonded at ``1 - exp(-beta
J)``, each site to its own label's ghost at ``1 - exp(-beta K)`` with ``K = h
- min h``, a cluster with a ghost bond kept and every other drawn uniformly).
None shares code with the moves. The two-rung step is each rung's kernel,
then the exchange; it is checked to keep ``exp(-beta_1 E) (x) exp(-beta_2
E)`` to 1e-12, and :func:`parallel_tempering`'s one-step draws from a fixed
pair against its row. :func:`rung_moves` is checked on hand cases.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    PottsMove,
    critical_ratio,
    field_ratio,
    parallel_tempering,
    rung_moves,
)
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, forbid

from tests.regression.sample.test_potts_heat_bath_cluster import (
    COUPLINGS,
    EDGES,
    N_STATES,
    _components,
    _graph,
    _rows,
    _states,
)
from tests.regression.sample.test_potts_tempering_cluster import (
    BETAS,
    ORIGIN,
    TEMPERATURES,
    _boltzmann,
    _exchange,
    _kernel,
    _outside,
    _step,
)

H = PottsMove.SWENDSEN_WANG_HEAT_BATH
S = PottsMove.SINGLE_SITE
G = PottsMove.GHOST_SPIN
#: Two ladders, coldest rung first as :data:`TEMPERATURES` is: the issue's
#: (single-site after heat-bath Swendsen-Wang cold, the cluster move alone
#: hot), and one with the ghost-spin pass on each rung, before a sweep cold
#: and after a cluster move hot.
LADDERS = {
    "cluster-then-sweep": ((H, S), (H,)),
    "ghost-mixed": ((G, S), (H, G)),
}


def _sweep_kernel(rows: np.ndarray, beta: float) -> np.ndarray:
    """The exact ``(81, 81)`` kernel of one heat-bath sweep over sites 0 to 3 in order."""
    states = _states()
    index = {state: k for k, state in enumerate(states)}
    kernel = np.eye(len(states))
    for site in range(4):
        update = np.zeros_like(kernel)
        for state in states:
            log_weights = beta * rows[site].copy()
            for (i, j), coupling in zip(EDGES, COUPLINGS, strict=True):
                if site in (i, j):
                    other = state[j] if site == i else state[i]
                    log_weights[other] += beta * coupling
            law = np.exp(log_weights - log_weights.max())
            law /= law.sum()
            for label in range(N_STATES):
                target = list(state)
                target[site] = label
                update[index[state], index[tuple(target)]] += law[label]
        kernel = kernel @ update
    return kernel


def _ghost_kernel(rows: np.ndarray, beta: float) -> np.ndarray:
    """The exact ``(81, 81)`` kernel of one ghost-spin pass on a finite field."""
    states = _states()
    index = {state: k for k, state in enumerate(states)}
    kernel = np.zeros((len(states), len(states)))
    ghost = rows - rows.min(axis=1, keepdims=True)
    bond = [1.0 - np.exp(-beta * coupling) for coupling in COUPLINGS]
    for state in states:
        like = [k for k, (i, j) in enumerate(EDGES) if state[i] == state[j]]
        to_ghost = [1.0 - np.exp(-beta * ghost[i, state[i]]) for i in range(4)]
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
            for held in itertools.product((False, True), repeat=4):
                share = weight * float(
                    np.prod(
                        [
                            p if h else 1.0 - p
                            for p, h in zip(to_ghost, held, strict=True)
                        ]
                    )
                )
                free = [c for c in clusters if not any(held[i] for i in c)]
                for labels in itertools.product(range(N_STATES), repeat=len(free)):
                    target = list(state)
                    for cluster, label in zip(free, labels, strict=True):
                        for site in cluster:
                            target[site] = label
                    kernel[index[state], index[tuple(target)]] += share / (
                        N_STATES ** len(free)
                    )
    return kernel


def _move_kernel(move: PottsMove, rows: np.ndarray, beta: float) -> np.ndarray:
    if move is S:
        return _sweep_kernel(rows, beta)
    if move is G:
        return _ghost_kernel(rows, beta)
    return _kernel(move, rows, beta)


def _rung_kernel(
    moves: tuple[PottsMove, ...], rows: np.ndarray, beta: float
) -> np.ndarray:
    """The rung's moves' kernels multiplied in the order they run."""
    kernel = np.eye(N_STATES**4)
    for move in moves:
        kernel = kernel @ _move_kernel(move, rows, beta)
    return kernel


def _kernels(
    ladder: tuple[tuple[PottsMove, ...], ...], rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _rung_kernel(ladder[0], rows, BETAS[0]),
        _rung_kernel(ladder[1], rows, BETAS[1]),
    )


def _row(ladder: tuple[tuple[PottsMove, ...], ...], rows: np.ndarray) -> np.ndarray:
    index = {state: k for k, state in enumerate(_states())}
    origin = np.zeros((len(index), len(index)))
    origin[index[ORIGIN[0]], index[ORIGIN[1]]] = 1.0
    return _step(origin, _kernels(ladder, rows), _exchange(rows))


def _empirical_row(
    ladder: tuple[tuple[PottsMove, ...], ...], rows: np.ndarray, draws: int
) -> np.ndarray:
    """Frequencies of ``draws`` one-step runs from :data:`ORIGIN`, each on its own spawned generator."""
    graph = _graph()
    index = {state: k for k, state in enumerate(_states())}
    counts = np.zeros((len(index), len(index)))
    start = np.array(ORIGIN, dtype=np.int64)
    for rng in np.random.default_rng(1158).spawn(draws):
        run = parallel_tempering(
            graph,
            rows,
            TEMPERATURES,
            rng,
            1,
            move=ladder,
            cluster_backend=Backend.PYTHON,
            start=start,
        )
        cold, hot = (tuple(row.tolist()) for row in run.states[0])
        counts[index[cold], index[hot]] += 1
    return counts / draws


@pytest.mark.oracle
@pytest.mark.parametrize("name", LADDERS)
def test_a_mixed_ladder_keeps_the_product_boltzmann_law(name: str) -> None:
    # Each move's kernel keeps its rung's exp(-beta E), so their product in
    # order does, and the move-then-exchange step keeps the product law, every
    # entry to 1e-12: per-rung moves leave the exchange exact.
    ladder, rows = LADDERS[name], _rows()
    kernels = _kernels(ladder, rows)
    laws = (_boltzmann(rows, BETAS[0]), _boltzmann(rows, BETAS[1]))
    joint = np.outer(*laws)

    for rung, beta, law in zip(ladder, BETAS, laws, strict=True):
        for move in rung:
            single = _move_kernel(move, rows, beta)
            np.testing.assert_allclose(single.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(law @ single, law, rtol=0.0, atol=1e-12)
    for kernel, law in zip(kernels, laws, strict=True):
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
@pytest.mark.parametrize("name", LADDERS)
def test_one_mixed_step_draws_the_product_kernel_row(name: str, draws: int) -> None:
    # `draws` one-step runs from the pair ORIGIN, each joint cell expected 25
    # times or more within four binomial standard errors of the product
    # kernel's row, the rarer cells pooled and held to the same bound, and a
    # cell the kernel never reaches never reached.
    ladder, rows = LADDERS[name], _rows()

    failed = _outside(_empirical_row(ladder, rows, draws), _row(ladder, rows), draws)

    assert not failed, failed


@pytest.mark.oracle
def test_a_rung_that_drops_its_second_move_is_refuted() -> None:
    # The cold rung's cluster move without the sweep after it, against the
    # row of both: the largest cell sits 11.9 standard errors off at 10,000
    # draws, the evidence the row has power over a rung running every move it
    # names. The order of the two is not refereed here: reversed, the largest
    # cell moves 1.5 standard errors.
    rows = _rows()

    failed = _outside(
        _empirical_row(((H,), (H,)), rows, 10_000),
        _row(LADDERS["cluster-then-sweep"], rows),
        10_000,
    )

    assert failed


@pytest.mark.smoke
@pytest.mark.snapshot
@pytest.mark.parametrize("move", [S, H, PottsMove.WOLFF], ids=str)
def test_one_move_per_rung_spelled_out_is_the_single_move_bitwise(
    move: PottsMove,
) -> None:
    # A PottsMove and the same move named once per rung, alone or in a
    # one-move tuple, are one chain: the same states, walkers, acceptances,
    # best energy and spend. The default's SHA-256 pin is
    # `test_potts_tempering_cluster.py`'s.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1158).normal(0.0, 0.5, (16, 3))

    def run(spelled: object) -> tuple[bytes, float, int]:
        result = parallel_tempering(
            graph,
            field,
            (0.5, 1.0, 2.0),
            np.random.default_rng(1158),
            20,
            burn_in=3,
            move=spelled,  # type: ignore[arg-type]
        )
        return (
            result.states.tobytes()
            + result.walkers.tobytes()
            + result.swap_acceptance.tobytes(),
            result.energy,
            result.spent,
        )

    assert run(move) == run((move,) * 3) == run([(move,), (move,), (move,)])


@pytest.mark.analytic
def test_spent_sums_each_rungs_moves() -> None:
    # A rung of (Swendsen-Wang, single-site) charges two sweeps a step, a
    # ghost-spin rung one sweep and a ghost bond per site, a single-site rung
    # one sweep.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    steps = 12
    per_sweep = graph.n_nodes + 2 * len(graph.edges)

    run = parallel_tempering(
        graph,
        np.zeros(3),
        (0.5, 1.0, 2.0),
        np.random.default_rng(0),
        steps - 2,
        burn_in=2,
        move=((H, S), G, (S,)),
    )

    assert run.spent == steps * (2 * per_sweep + per_sweep + graph.n_nodes + per_sweep)


@pytest.mark.smoke
def test_a_ladder_of_moves_is_one_entry_per_rung_of_moves() -> None:
    graph, rows = _graph(), _rows()

    def run(move: object) -> None:
        parallel_tempering(
            graph,
            rows,
            TEMPERATURES,
            np.random.default_rng(0),
            2,
            move=move,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="one entry per rung, 2, got 3"):
        run((S, S, S))
    with pytest.raises(ValueError, match="non-empty"):
        run(((), S))
    with pytest.raises(TypeError, match="are PottsMove"):
        run(("single-site", S))
    antiferromagnet = PottsGraph(3, ((0, 1), (1, 2)), (1.0, -1.0))
    with pytest.raises(ValueError, match="every coupling >= 0"):
        parallel_tempering(
            antiferromagnet,
            np.zeros(2),
            TEMPERATURES,
            np.random.default_rng(0),
            2,
            move=((S,), (S, H)),
        )


@pytest.mark.analytic
def test_the_critical_ratio_is_the_square_lattice_transition() -> None:
    # On the square lattice (degree 4) K_c = ln(1 + sqrt(q)) exactly, so
    # beta J / K_c is 1 at T = J / K_c; a triangular lattice's degree 6 scales
    # it by 4 / 6.
    square = lattice_graph((6, 6), BoundaryCondition.PERIODIC, 0.9)
    k_c = critical_coupling(3)

    assert critical_ratio(square, 3, 0.9 / k_c) == pytest.approx(1.0, abs=1e-12)
    assert critical_ratio(square, 3, 0.45 / k_c) == pytest.approx(2.0, abs=1e-12)
    assert critical_ratio(PottsGraph(2, (), ()), 3, 1.0) == 0.0


@pytest.mark.analytic
def test_rung_moves_pairs_a_sweep_with_the_cluster_move_from_the_transition() -> None:
    # Square lattice, J = 1, q = 3: K_c = 1.005. Hot of it (T = 2, ratio
    # 0.50) heat-bath Swendsen-Wang alone; at it (ratio 1) and colder (T =
    # 0.25, ratio 3.98) the cluster move then a single-site sweep.
    graph = lattice_graph((6, 6), BoundaryCondition.PERIODIC, 1.0)
    at = 1.0 / critical_coupling(3)

    assert rung_moves(graph, np.zeros(3), (2.0, at, 0.25)) == ((H,), (H, S), (H, S))


@pytest.mark.analytic
def test_a_strong_field_leaves_the_rule_as_it_is() -> None:
    # The field enters through each cluster's heat-bath label draw, so a field
    # 30 times the coupling's scale picks the moves a zero field picks; a
    # forbidden label is read over the allowed ones, not as an infinite spread.
    graph = lattice_graph((6, 6), BoundaryCondition.PERIODIC, 1.0)
    rng = np.random.default_rng(1158)
    strong = 30.0 * rng.normal(0.0, 1.0, (36, 3))
    ladder = (4.0, 1.0, 0.3)

    assert rung_moves(graph, strong, ladder) == rung_moves(graph, np.zeros(3), ladder)
    forbidden = forbid(np.zeros((36, 3)), np.tile([True, True, False], (36, 1)))
    assert rung_moves(graph, forbidden, ladder) == rung_moves(
        graph, np.zeros(3), ladder
    )
    assert field_ratio(graph, forbidden) == 0.0


@pytest.mark.analytic
def test_the_field_ratio_is_the_median_spread_over_the_coupling() -> None:
    # Spreads 0, 1, 2 on three sites of a path (mean coupling 2, mean degree
    # 4 / 3): median 1 over 2 * 4 / 3, 0.375.
    graph = PottsGraph(3, ((0, 1), (1, 2)), (1.0, 3.0))
    rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    assert field_ratio(graph, rows) == pytest.approx(0.375, abs=1e-12)


@pytest.mark.analytic
def test_an_antiferromagnet_is_single_site_on_every_rung() -> None:
    graph = PottsGraph(3, ((0, 1), (1, 2)), (1.0, -1.0))

    assert rung_moves(graph, np.zeros(2), (2.0, 0.5)) == ((S,), (S,))
