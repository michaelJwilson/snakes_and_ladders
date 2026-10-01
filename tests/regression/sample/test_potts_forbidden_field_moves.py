"""Ghost-spin and label-directed Swendsen-Wang on a field that forbids labels, against their enumerated kernels (issue #1154).

The 4-site, q = 3 graph and the three forbidden entries of
`test_potts_forbidden_labels.py`. Each kernel is built here from the bond
probabilities and the cluster rules, sharing no code with the moves: the
ghost-spin kernel bonds site ``i`` to its own label's ghost with probability
``1 - exp(-beta K)``, ``K`` the field less its minimum over allowed labels,
and draws a free cluster uniformly from the labels every member allows; the
label-directed kernel proposes each cluster onto the target, or uniformly off
it, rejects a proposal some member forbids and accepts one off a forbidden
label. Referees: each kernel keeps ``exp(-beta E)`` restricted to the allowed
labellings, never leaves them, and carries every start to that law, each to
1e-12, at ``beta = 0.7`` and at ``beta = 0``, where the law is uniform over
the allowed labellings; each move's empirical row matches its kernel's cell
by cell with ``RuntimeWarning`` an error. On the #1139 3x3 fixture both moves
leave a forbidden start without a warning and never re-enter a forbidden
label, and the ghost-spin chain from an allowed start no longer freezes.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import ghost_spin_sweep, label_directed_sweep
from sal.sim.potts import energy

from tests.regression.sample.test_potts_forbidden_labels import (
    ALLOWED,
    DRAWS,
    LATTICE,
    LATTICE_ALLOWED,
    LATTICE_FIELD,
    MIXED,
    ORIGIN,
    SIGMAS,
    SWEEPS,
    _allowed_states,
    _field,
)
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

#: An allowed labelling with two clusters to bond: sites 0-1 at 1, 2-3 at 0.
ALLOWED_ORIGIN = (1, 1, 0, 0)
BETAS = (BETA, 0.0)


def _restricted_boltzmann(beta: float) -> np.ndarray:
    """``exp(-beta E)`` on the allowed labellings, ``E`` on the finite field, zero elsewhere."""
    graph, finite = _graph(), _rows()
    weights = np.array(
        [np.exp(-beta * energy(graph, finite, np.array(state))) for state in _states()]
    )
    weights[~_allowed_states()] = 0.0
    return np.asarray(weights / weights.sum())


def _bondings(
    state: tuple[int, ...], beta: float
) -> list[tuple[float, list[list[int]]]]:
    """Each subset of like edges bonded, as its probability and its clusters."""
    bond = [1.0 - np.exp(-beta * coupling) for coupling in COUPLINGS]
    like = [k for k, (i, j) in enumerate(EDGES) if state[i] == state[j]]
    out = []
    for bonded in itertools.product((False, True), repeat=len(like)):
        weight = float(
            np.prod(
                [
                    bond[k] if b else 1.0 - bond[k]
                    for k, b in zip(like, bonded, strict=True)
                ]
            )
        )
        edges = [EDGES[k] for k, b in zip(like, bonded, strict=True) if b]
        out.append((weight, _components(edges)))
    return out


def _relabel(
    kernel: np.ndarray,
    state: tuple[int, ...],
    clusters: list[list[int]],
    laws: list[np.ndarray],
    weight: float,
) -> None:
    """Add ``weight`` times the product of the clusters' label laws to ``state``'s row."""
    index = {s: k for k, s in enumerate(_states())}
    for labels in itertools.product(range(N_STATES), repeat=len(clusters)):
        target = list(state)
        for cluster, label in zip(clusters, labels, strict=True):
            for site in cluster:
                target[site] = label
        probability = float(
            np.prod([p[label] for p, label in zip(laws, labels, strict=True)])
        )
        kernel[index[state], index[tuple(target)]] += weight * probability


def _ghost_kernel(beta: float) -> np.ndarray:
    """The exact ``(81, 81)`` ghost-spin kernel on the forbidden field."""
    rows, states = _rows(), _states()
    floor = np.array([rows[i][ALLOWED[i]].min() for i in range(4)])
    kernel = np.zeros((len(states), len(states)))
    for state in states:
        # A site on an allowed label bonds to its ghost; one on a forbidden
        # label has no ghost bond to form.
        ghost = [
            1.0 - np.exp(-beta * (rows[i, state[i]] - floor[i]))
            if ALLOWED[i, state[i]]
            else 0.0
            for i in range(4)
        ]
        for weight, clusters in _bondings(state, beta):
            for ghosted in itertools.product((False, True), repeat=4):
                share = weight * float(
                    np.prod(
                        [
                            g if b else 1.0 - g
                            for g, b in zip(ghost, ghosted, strict=True)
                        ]
                    )
                )
                if share == 0.0:
                    continue
                laws = []
                for cluster in clusters:
                    common = ALLOWED[cluster].all(axis=0)
                    if any(ghosted[i] for i in cluster) or not common.any():
                        laws.append(np.eye(N_STATES)[state[cluster[0]]])
                    else:
                        laws.append(common / common.sum())
                _relabel(kernel, state, clusters, laws, share)
    return kernel


def _label_directed_kernel(beta: float, target: int) -> np.ndarray:
    """The exact ``(81, 81)`` label-directed kernel at ``target`` on the forbidden field."""
    rows, states = _rows(), _states()
    kernel = np.zeros((len(states), len(states)))
    for state in states:
        for weight, clusters in _bondings(state, beta):
            laws = []
            for cluster in clusters:
                current = state[cluster[0]]
                allows = ALLOWED[cluster].all(axis=0)
                proposals = (
                    {label: 1.0 / N_STATES for label in range(N_STATES)}
                    if current == target
                    else {target: 1.0}
                )
                hastings = np.log(N_STATES) * (1.0 if current == target else -1.0)
                law = np.zeros(N_STATES)
                for label, chance in proposals.items():
                    if not allows[label]:
                        accept = 0.0
                    elif not allows[current]:
                        accept = 1.0
                    else:
                        gain = beta * (rows[cluster, label] - rows[cluster, current])
                        accept = min(1.0, float(np.exp(gain.sum() + hastings)))
                    law[label] += chance * accept
                law[current] += 1.0 - law.sum()
                laws.append(law)
            _relabel(kernel, state, clusters, laws, weight)
    return kernel


def _keeps(kernel: np.ndarray, beta: float) -> None:
    """Rows sum to one, no allowed labelling moves to a forbidden one, and ``pi K = pi``."""
    allowed, pi = _allowed_states(), _restricted_boltzmann(beta)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert not kernel[np.ix_(allowed, ~allowed)].any()
    np.testing.assert_allclose(pi @ kernel, pi, rtol=0.0, atol=1e-12)


def _mixes(step: np.ndarray, beta: float) -> None:
    """Every row of ``step^MIXED``, forbidden starts included, is ``pi``."""
    pi = _restricted_boltzmann(beta)
    limit = np.linalg.matrix_power(step, MIXED)
    np.testing.assert_allclose(limit, np.tile(pi, (81, 1)), rtol=0.0, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("beta", BETAS)
def test_the_enumerated_ghost_kernel_keeps_the_restricted_law(beta: float) -> None:
    # At beta = 0 the law is uniform over the 24 allowed labellings.
    kernel = _ghost_kernel(beta)

    _keeps(kernel, beta)
    _mixes(kernel, beta)


@pytest.mark.oracle
@pytest.mark.parametrize("beta", BETAS)
def test_the_enumerated_label_directed_kernel_keeps_the_restricted_law(
    beta: float,
) -> None:
    # Each target's kernel keeps the law; their cycle, the chain
    # `sample_potts` runs, carries every start to it.
    kernels = [_label_directed_kernel(beta, target) for target in range(N_STATES)]
    for kernel in kernels:
        _keeps(kernel, beta)
    _mixes(kernels[0] @ kernels[1] @ kernels[2], beta)


#: (move, start, beta, backend, target): forbidden and allowed starts, both
#: union-finds, and beta = 0.
ROWS = [
    ("ghost", ORIGIN, BETA, Backend.PYTHON, 0),
    ("ghost", ALLOWED_ORIGIN, BETA, Backend.RUST, 0),
    ("ghost", ORIGIN, 0.0, Backend.PYTHON, 0),
    ("label_directed", ORIGIN, BETA, Backend.PYTHON, 0),
    ("label_directed", ALLOWED_ORIGIN, BETA, Backend.RUST, 1),
    ("label_directed", ORIGIN, 0.0, Backend.PYTHON, 2),
]


@pytest.mark.oracle
@pytest.mark.parametrize(("move", "start", "beta", "backend", "target"), ROWS, ids=str)
def test_each_move_draws_its_enumerated_row(
    move: str,
    start: tuple[int, ...],
    beta: float,
    backend: Backend,
    target: int,
) -> None:
    # 40,000 steps from `start`, RuntimeWarning an error: every one of the
    # 81 cells within four binomial standard errors of the kernel's row, and
    # a cell the kernel never reaches never reached.
    index = {state: k for k, state in enumerate(_states())}
    kernel = (
        _ghost_kernel(beta) if move == "ghost" else _label_directed_kernel(beta, target)
    )
    exact = kernel[index[start]]
    graph, rows = _graph(), _field()
    rng = np.random.default_rng(1154)
    counts = np.zeros(len(index))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(DRAWS):
            state = np.array(start, dtype=np.int64)
            if move == "ghost":
                ghost_spin_sweep(state, graph, rows, rng, beta, backend=backend)
            else:
                label_directed_sweep(
                    state, graph, rows, rng, target, beta, backend=backend
                )
            counts[index[tuple(state.tolist())]] += 1
    empirical = counts / DRAWS

    error = np.sqrt(exact * (1.0 - exact) / DRAWS)
    assert np.flatnonzero(np.abs(empirical - exact) > SIGMAS * error).size == 0


def _lattice_run(move: str, start: np.ndarray) -> list[np.ndarray]:
    """``SWEEPS`` sweeps of ``move`` on the #1139 fixture from ``start``, RuntimeWarning an error."""
    rng = np.random.default_rng(1154)
    state = start.copy()
    visited = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for sweep in range(SWEEPS):
            if move == "ghost":
                ghost_spin_sweep(state, LATTICE, LATTICE_FIELD, rng)
            else:
                label_directed_sweep(state, LATTICE, LATTICE_FIELD, rng, sweep % 3)
            visited.append(state.copy())
    return visited


@pytest.mark.analytic
@pytest.mark.parametrize("move", ["ghost", "label_directed"])
def test_a_forbidden_start_is_left_and_never_re_entered(move: str) -> None:
    # Every site at label 0, which no site allows: the chain reaches an
    # allowed labelling, and from the first one on every labelling is allowed.
    sites = np.arange(LATTICE.n_nodes)
    visited = _lattice_run(move, np.zeros(LATTICE.n_nodes, dtype=np.int64))
    allowed = [bool(LATTICE_ALLOWED[sites, state].all()) for state in visited]

    assert any(allowed)
    first = allowed.index(True)
    assert all(allowed[first:]), f"re-entered after sweep {first}"


@pytest.mark.analytic
def test_the_ghost_chain_moves_from_an_allowed_start() -> None:
    # Every site at label 1, which every site allows. With the minimum over
    # every label the allowed couplings were `+inf`, each site bonded to its
    # own ghost, and the chain stayed at its start for all 300 sweeps; it now
    # visits more than one labelling, every one of them allowed.
    sites = np.arange(LATTICE.n_nodes)
    visited = _lattice_run("ghost", np.ones(LATTICE.n_nodes, dtype=np.int64))
    distinct = {tuple(state.tolist()) for state in visited}

    assert len(distinct) > 1
    assert all(LATTICE_ALLOWED[sites, state].all() for state in visited)
