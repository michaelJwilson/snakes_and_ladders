"""Spin configurations for a `k`-state Potts model on a general graph.

An N-D lattice (:func:`phylo.sim.graph.lattice_graph`) is a constructed case
of :class:`~phylo.sim.graph.PottsGraph`, not a separate code path, per
``docs/tex/main.tex``'s "Potts Models in an External Field" (Mezard &
Montanari, ch. 2; Koller & Friedman for the general framing). A graph
recognized as a 1-D chain with an open boundary is sampled exactly, by the
same backward-message recursion :func:`phylo.opt.potts.simulate_chains`
delegates to here; every other graph is sampled by single-site Gibbs
(heat-bath) Markov chain Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phylo.fixtures import load_declared
from phylo.numerics import logsumexp, sample_rows
from phylo.sim.graph import BoundaryCondition, PottsGraph

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "shape",
        "boundary",
        "n_states",
        "coupling",
        "field",
        "n_samples",
        "burn_in",
        "tolerance",
    }
)


@dataclass(frozen=True)
class PottsLatticeParams:
    """Fully-specified truth for a Potts-lattice fixture.

    Parameters
    ----------
    shape : tuple[int, ...]
        Lattice extent along each dimension; length 1 is a chain, length 2
        a grid, and so on.
    boundary : BoundaryCondition
        Applied uniformly across every dimension.
    n_states : int
        Number of states per site, ``k``, >= 2.
    coupling : float
        Uniform ``J`` applied to every edge.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    seed : int
        Seed for ``np.random.default_rng``.
    n_samples : int
        Configurations to draw, each its own independent Markov chain.
    burn_in : int
        Gibbs sweeps each chain runs from its own independent random start
        before its state is recorded. Unused by the exact 1-D open-chain
        sampler, which draws i.i.d. samples directly.
    tolerance : float
        Monte Carlo tolerance a validation test checks simulated frequencies
        against their exact counterpart within.
    """

    shape: tuple[int, ...]
    boundary: BoundaryCondition
    n_states: int
    coupling: float
    field: np.ndarray
    seed: int
    n_samples: int
    burn_in: int
    tolerance: float


def _boundary(path: Path, raw: object) -> BoundaryCondition:
    """Parse a yaml boundary field, naming the file when it is unrecognized.

    Parameters
    ----------
    path : Path
        The file being loaded, for the error message.
    raw : object
        The yaml value.

    Returns
    -------
    BoundaryCondition
        The parsed boundary.

    Raises
    ------
    ValueError
        If ``raw`` is not one of the recognized boundary conditions. Caught
        here rather than in :func:`phylo.sim.graph.lattice_graph`, which
        takes the enum and so cannot be handed a bad string at all.
    """
    try:
        return BoundaryCondition(str(raw))
    except ValueError:
        recognized = sorted(condition.value for condition in BoundaryCondition)
        msg = f"{path}: boundary must be one of {recognized}, got {raw!r}"
        raise ValueError(msg) from None


def load_potts_lattice_params(path: Path) -> PottsLatticeParams:
    """Load and validate a Potts-lattice fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    PottsLatticeParams
        The parsed, validated truth.

    Raises
    ------
    ValueError
        If a required field is missing, ``field`` has the wrong shape, or a
        size is too small to identify the model.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)

    shape = tuple(int(extent) for extent in raw["shape"])
    n_states = int(raw["n_states"])
    if n_states < 2:
        msg = f"{path}: n_states must be >= 2, got {n_states}"
        raise ValueError(msg)

    field = np.asarray(raw["field"], dtype=np.float64)
    if field.shape != (n_states,):
        msg = f"{path}: field has shape {field.shape}, expected ({n_states},)"
        raise ValueError(msg)

    return PottsLatticeParams(
        shape=shape,
        boundary=_boundary(path, raw["boundary"]),
        n_states=n_states,
        coupling=float(raw["coupling"]),
        field=field,
        seed=int(raw["seed"]),
        n_samples=int(raw["n_samples"]),
        burn_in=int(raw["burn_in"]),
        tolerance=float(raw["tolerance"]),
    )


@dataclass(frozen=True)
class SimulatedPottsDataset:
    """Simulated Potts configurations together with the graph and truth.

    Parameters
    ----------
    configurations : np.ndarray
        Integer states, shape ``(n_samples, graph.n_nodes)``.
    graph : PottsGraph
        The graph the configurations were drawn on.
    field : np.ndarray
        The external field used, shape ``(n_states,)``.
    seed : int
        Seed used.
    """

    configurations: np.ndarray
    graph: PottsGraph
    field: np.ndarray
    seed: int


def simulate_potts(
    graph: PottsGraph,
    field: np.ndarray,
    seed: int,
    n_samples: int,
    burn_in: int = 500,
) -> SimulatedPottsDataset:
    """Draw ``n_samples`` Potts configurations on ``graph``.

    Parameters
    ----------
    graph : PottsGraph
        The graph to sample on.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    seed : int
        Seed for ``np.random.default_rng``.
    n_samples : int
        Configurations to draw.
    burn_in : int
        Gibbs sweeps each of the ``n_samples`` independent chains runs from
        its own random start before its state is recorded. Ignored on the
        exact open-chain path.

    Returns
    -------
    SimulatedPottsDataset
        The configurations, the graph, and the generating truth.
    """
    if graph.is_open_chain():
        configurations = _simulate_open_chain_exact(graph, field, seed, n_samples)
    else:
        configurations = _simulate_gibbs(graph, field, seed, n_samples, burn_in)
    return SimulatedPottsDataset(
        configurations=configurations, graph=graph, field=field, seed=seed
    )


def _simulate_open_chain_exact(
    graph: PottsGraph, field: np.ndarray, seed: int, n_samples: int
) -> np.ndarray:
    """Exact i.i.d. samples on an open chain, by backward messages.

    Moved verbatim (not reimplemented) from the recursion
    ``phylo.opt.potts.simulate_chains`` used before this module existed: the
    chain's backward messages give the conditional distributions directly,
    so the fixture carries no equilibration assumption (root ``CLAUDE.md``,
    "Simulate Component-Wise"). Generalized to a per-edge coupling, which
    reduces to the uniform-coupling case ``phylo.opt.potts`` needs.
    """
    rng = np.random.default_rng(seed)
    length = graph.n_nodes
    n_states = field.shape[0]

    # log_transfer[i] is the log weight of the edge between site i and i+1,
    # plus site i+1's own field -- the same shape `phylo.opt.potts.log_partition`
    # sums via transfer matrix, one matrix per edge rather than one shared.
    log_transfer = [
        coupling * np.eye(n_states) + field[np.newaxis, :]
        for coupling in graph.coupling
    ]

    # backward[i] is the log weight of everything from site i+1 onward, given
    # the state at site i; backward[-1] is empty and so zero.
    backward = np.zeros((length, n_states))
    for i in range(length - 2, -1, -1):
        backward[i] = logsumexp(
            log_transfer[i] + backward[i + 1][np.newaxis, :], axis=1
        )

    states = np.empty((n_samples, length), dtype=np.int64)
    first = _softmax(field + backward[0])
    states[:, 0] = rng.choice(n_states, size=n_samples, p=first)
    for i in range(1, length):
        conditional = _softmax(log_transfer[i - 1] + backward[i][np.newaxis, :], axis=1)
        states[:, i] = sample_rows(rng, conditional, states[:, i - 1])
    return states


def _simulate_gibbs(
    graph: PottsGraph,
    field: np.ndarray,
    seed: int,
    n_samples: int,
    burn_in: int,
) -> np.ndarray:
    """``n_samples`` independent chains, each run ``burn_in`` sweeps.

    Single-site Gibbs: each sweep visits every site in index order and
    redraws it from its exact conditional given its current neighbours,
    ``p(s_i = k | rest) proportional to exp(h_k + sum_j J_ij delta(k, s_j))``.
    The ``n_samples`` chains are independent by construction -- separate
    random starts, evolved with independent draws from the same generator --
    rather than one long chain thinned for decorrelation, so the Python-level
    loop is over sweeps, not over sweeps times samples: each site update is
    one vectorized draw across every chain.
    """
    rng = np.random.default_rng(seed)
    n_states = field.shape[0]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for (a, b), coupling in graph.weighted_edges():
        adjacency[a].append((b, coupling))
        adjacency[b].append((a, coupling))

    state = rng.integers(0, n_states, size=(n_samples, graph.n_nodes))
    chain_index = np.arange(n_samples)

    for _ in range(burn_in):
        for node in range(graph.n_nodes):
            local = np.tile(field, (n_samples, 1))
            for neighbor, coupling in adjacency[node]:
                local[chain_index, state[:, neighbor]] += coupling
            state[:, node] = sample_rows(rng, _softmax(local, axis=1), chain_index)

    return state


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    result: np.ndarray = weights / weights.sum(axis=axis, keepdims=True)
    return result
