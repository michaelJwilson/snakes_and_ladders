"""Spin configurations for a `k`-state Potts model on a general graph.

An N-D lattice (:func:`snakes_and_ladders.sim.graph.lattice_graph`) is a constructed case
of :class:`~snakes_and_ladders.sim.graph.PottsGraph`, not a separate code path, per
``docs/tex/textbook.tex``, ``sec:potts`` (Mezard &
Montanari, ch. 2; Koller & Friedman for the general framing). A graph
recognized as a 1-D chain with an open boundary is sampled exactly, by the
same backward-message recursion :func:`snakes_and_ladders.opt.potts.simulate_chains`
delegates to here; every other graph is sampled by single-site Gibbs
(heat-bath) Markov chain Monte Carlo.

**The external field is per site or shared, and one shape reaches the
kernels.** ``h`` is declared either as ``(n_states,)`` --- one field every
site shares --- or as ``(n_nodes, n_states)``, where each site carries its
own. :func:`site_field` widens the first to the second at each entry point,
so no sampler, oracle or evaluator below carries two cases; a shared field is
the broadcast of a per-site one and not a second model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.numerics import logsumexp, sample_rows
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)


def site_field(field: np.ndarray, n_nodes: int) -> np.ndarray:
    """``field`` as one row per site, whichever of the two shapes it arrives in.

    Parameters
    ----------
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` --- shared by every site
        --- or ``(n_nodes, n_states)``.
    n_nodes : int
        Sites the graph carries.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``, C-contiguous. A shared field is
        broadcast here and nowhere else, so a kernel below indexes one shape
        and a caller declaring either reaches the same code.

    Raises
    ------
    ValueError
        If ``field`` is neither shape. Refused rather than broadcast: a
        ``(n_states,)`` field on a graph whose node count equals the state
        count would otherwise be read as per-site, silently scoring a
        different model.

    Examples
    --------
    >>> site_field(np.array([0.0, 1.0]), 3).shape
    (3, 2)
    """
    if field.ndim == 1:
        return np.broadcast_to(field, (n_nodes, field.shape[0])).copy()
    if field.shape[:1] == (n_nodes,) and field.ndim == 2:
        return np.ascontiguousarray(field)
    msg = (
        f"field has shape {field.shape}; expected (n_states,) or ({n_nodes}, n_states)"
    )
    raise ValueError(msg)


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


#: The word a fixture writes where its coupling is the exact transition
#: rather than a number it chose.
CRITICAL = "critical"


def critical_coupling(n_states: int) -> float:
    """The exact ``q``-state Potts transition on a square lattice, ``ln(1 + sqrt(q))``.

    The self-dual point of the square-lattice Potts model (Baxter, ch. 12;
    ``docs/tex/textbook.tex``, ``sec:potts``), where the correlation length
    diverges and single-site updates slow critically.

    Computed rather than stored, so an instance declared *at* the transition
    cannot drift off it through a rounded literal, and keyed on the file's
    own state count rather than on the 3 the one fixture using it declares.

    Parameters
    ----------
    n_states : int
        ``q``.

    Returns
    -------
    float

    Examples
    --------
    >>> round(critical_coupling(3), 6)
    1.005052
    """
    return float(np.log(1.0 + np.sqrt(n_states)))


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
        Uniform ``J`` applied to every edge. Declared as a number, or as
        :data:`CRITICAL` for :func:`critical_coupling`.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
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
        here rather than in :func:`snakes_and_ladders.sim.graph.lattice_graph`, which
        takes the enum and so cannot be handed a bad string at all.
    """
    try:
        return BoundaryCondition(str(raw))
    except ValueError:
        recognized = sorted(condition.value for condition in BoundaryCondition)
        msg = f"{path}: boundary must be one of {recognized}, got {raw!r}"
        raise ValueError(msg) from None


def _coupling(path: Path, raw: object, n_states: int) -> float:
    """Parse a yaml coupling, resolving :data:`CRITICAL` to the closed form.

    Raises
    ------
    ValueError
        If the value is a word other than :data:`CRITICAL`. A misspelt one
        would otherwise reach ``float()`` and fail with a message naming
        neither the file nor the word the loader does know.
    """
    if isinstance(raw, str):
        if raw != CRITICAL:
            msg = f"{path}: coupling {raw!r} is not a number or {CRITICAL!r}"
            raise ValueError(msg)
        return critical_coupling(n_states)
    if not isinstance(raw, int | float):
        msg = f"{path}: coupling {raw!r} is not a number or {CRITICAL!r}"
        raise ValueError(msg)
    return float(raw)


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
    n_nodes = int(np.prod(shape))
    if field.shape not in {(n_states,), (n_nodes, n_states)}:
        msg = (
            f"{path}: field has shape {field.shape}, expected ({n_states},) "
            f"or ({n_nodes}, {n_states})"
        )
        raise ValueError(msg)

    return PottsLatticeParams(
        shape=shape,
        boundary=_boundary(path, raw["boundary"]),
        n_states=n_states,
        coupling=_coupling(path, raw["coupling"], n_states),
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
        The external field used, shape ``(n_nodes, n_states)``: widened by
        :func:`site_field` whichever shape the caller passed, so a consumer
        reading it back off the dataset has one case rather than two.
    """

    configurations: np.ndarray
    graph: PottsGraph
    field: np.ndarray


def simulate_potts(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    burn_in: int = 500,
) -> SimulatedPottsDataset:
    """Draw ``n_samples`` Potts configurations on ``graph``.

    Parameters
    ----------
    graph : PottsGraph
        The graph to sample on.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
    rng : np.random.Generator
        Passed in rather than seeded here, so a caller drawing an *ensemble*
        gets independent datasets rather than the same one repeatedly ---
        `sim/CLAUDE.md`'s standing rule (issue #240).
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
    rows = site_field(field, graph.n_nodes)
    if graph.is_open_chain():
        configurations = _simulate_open_chain_exact(graph, rows, rng, n_samples)
    else:
        configurations = _simulate_gibbs(graph, rows, rng, n_samples, burn_in)
    return SimulatedPottsDataset(configurations=configurations, graph=graph, field=rows)


def _simulate_open_chain_exact(
    graph: PottsGraph, field: np.ndarray, rng: np.random.Generator, n_samples: int
) -> np.ndarray:
    """Exact i.i.d. samples on an open chain, by backward messages.

    Moved verbatim (not reimplemented) from the recursion
    ``snakes_and_ladders.opt.potts.simulate_chains`` used before this module existed: the
    chain's backward messages give the conditional distributions directly,
    so the fixture carries no equilibration assumption (root ``CLAUDE.md``,
    "Simulate Component-Wise"). Generalized to a per-edge coupling, which
    reduces to the uniform-coupling case ``snakes_and_ladders.opt.potts`` needs.
    """
    length = graph.n_nodes
    n_states = field.shape[1]

    # log_transfer[i] is the log weight of the edge between site i and i+1,
    # plus site i+1's *own* field -- the same shape
    # `snakes_and_ladders.opt.potts.log_partition` sums via transfer matrix, one
    # matrix per edge rather than one shared, and one field row per site
    # rather than one shared. Site 0's field is not in any transfer matrix and
    # enters at `first` below, which is where a per-site field is easiest to
    # drop and why the recursion is pinned against enumeration.
    log_transfer = [
        coupling * np.eye(n_states) + field[i + 1][np.newaxis, :]
        for i, coupling in enumerate(graph.coupling)
    ]

    # backward[i] is the log weight of everything from site i+1 onward, given
    # the state at site i; backward[-1] is empty and so zero.
    backward = np.zeros((length, n_states))
    for i in range(length - 2, -1, -1):
        backward[i] = logsumexp(
            log_transfer[i] + backward[i + 1][np.newaxis, :], axis=1
        )

    states = np.empty((n_samples, length), dtype=np.int64)
    first = _softmax(field[0] + backward[0])
    states[:, 0] = rng.choice(n_states, size=n_samples, p=first)
    for i in range(1, length):
        conditional = _softmax(log_transfer[i - 1] + backward[i][np.newaxis, :], axis=1)
        states[:, i] = sample_rows(rng, conditional, states[:, i - 1])
    return states


def _simulate_gibbs(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    burn_in: int,
) -> np.ndarray:
    """``n_samples`` independent chains, each run ``burn_in`` sweeps.

    Single-site Gibbs: each sweep visits every site in index order and
    redraws it from its exact conditional given its current neighbours,
    ``p(s_i = k | rest) proportional to exp(h_ik + sum_j J_ij delta(k, s_j))``.
    The ``n_samples`` chains are independent by construction -- separate
    random starts, evolved with independent draws from the same generator --
    rather than one long chain thinned for decorrelation, so the Python-level
    loop is over sweeps, not over sweeps times samples: each site update is
    one vectorized draw across every chain.
    """
    n_states = field.shape[1]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for (a, b), coupling in graph.weighted_edges():
        adjacency[a].append((b, coupling))
        adjacency[b].append((a, coupling))

    state = rng.integers(0, n_states, size=(n_samples, graph.n_nodes))
    chain_index = np.arange(n_samples)

    for _ in range(burn_in):
        for node in range(graph.n_nodes):
            local = np.tile(field[node], (n_samples, 1))
            for neighbor, coupling in adjacency[node]:
                local[chain_index, state[:, neighbor]] += coupling
            state[:, node] = sample_rows(rng, _softmax(local, axis=1), chain_index)

    return state


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    result: np.ndarray = weights / weights.sum(axis=axis, keepdims=True)
    return result


# --- a Potts prior whose external field varies per site (issue #413) ---------

#: The lattice constructors a spots fixture may name, so the geometry is a
#: declared word rather than a loader that only ever builds a square.
_GEOMETRIES: dict[str, Callable[[Any, BoundaryCondition, float], PottsGraph]] = {
    "square": lattice_graph,
    "triangular": triangular_lattice_graph,
}

_SPOTS_REQUIRED_FIELDS = frozenset(
    {
        "geometry",
        "shape",
        "boundary",
        "n_classes",
        "coupling",
        "alpha",
        "sizes",
        "seed",
        "n_samples",
        "burn_in",
        "tolerance",
    }
)


@dataclass(frozen=True)
class PottsSpotsParams:
    """A Potts prior over class labels whose field is driven by a per-site covariate.

    The spatial half of the coupled model of
    :mod:`snakes_and_ladders.sim.spatio_sequential` with the chains and the
    gated emissions removed and a per-site field added: each site carries a
    declared size, and a class whose ``alpha`` is positive is favoured where
    the site is large. The field is

        ``h[n, m] = alpha[m] * log(size[n] / size_bar)``

    with ``size_bar`` the geometric mean of the sizes, so the covariate is
    centred and a class with ``alpha = 0`` carries no field at any site ---
    the null the other classes are measured against.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, of the declared geometry, shape, boundary and coupling.
    n_classes : int
        ``M``, the states per site.
    alpha : np.ndarray
        Per class, shape ``(M,)``: how strongly the covariate tilts it.
    sizes : np.ndarray
        The per-site covariate, shape ``(n_nodes,)``, strictly positive.
    field : np.ndarray
        ``h``, shape ``(n_nodes, M)``, from ``alpha`` and ``sizes`` above.
    seed : int
        Seed for the sampler's ``np.random.default_rng``. The sizes carry
        their own seed where they are drawn rather than declared, so a
        change to one draw does not move the other.
    n_samples : int
        Configurations to draw, each its own independent Markov chain.
    burn_in : int
        Gibbs sweeps each chain runs before its state is recorded.
    tolerance : float
        Monte Carlo tolerance a validation test checks a simulated
        single-site frequency against its exact counterpart within.
    """

    graph: PottsGraph
    n_classes: int
    alpha: np.ndarray
    sizes: np.ndarray
    field: np.ndarray
    seed: int
    n_samples: int
    burn_in: int
    tolerance: float


def spots_field(alpha: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """``alpha[m] * log(size[n] / size_bar)`` as one row per site.

    Parameters
    ----------
    alpha : np.ndarray
        Per class, shape ``(M,)``.
    sizes : np.ndarray
        Per site, shape ``(n_nodes,)``, strictly positive.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, M)``.

    Raises
    ------
    ValueError
        If a size is not positive, which has no logarithm and which a
        covariate declared as a size cannot be.

    Examples
    --------
    >>> spots_field(np.array([1.0, 0.0]), np.array([1.0, 4.0])).round(4)
    array([[-0.6931,  0.    ],
           [ 0.6931,  0.    ]])
    """
    if not bool((sizes > 0.0).all()):
        msg = f"every size must be positive, got a minimum of {sizes.min()}"
        raise ValueError(msg)
    centred = np.log(sizes) - float(np.log(sizes).mean())
    return np.asarray(centred[:, np.newaxis] * alpha[np.newaxis, :])


def _spots_sizes(path: Path, declared: Any, n_nodes: int) -> np.ndarray:
    """The per-site covariate, declared outright or drawn from a declared law.

    Two forms, because the two sizes need different things of the file. At
    nine sites the profile is written out, so a reader sees which site is
    large; at 5,041 the file states the seed and the shape parameter and the
    sizes are what they imply, as
    :mod:`snakes_and_ladders.sim.count_pairs` states its counts.

    Only ``log_sigma`` is declared for the drawn form: ``spots_field``
    divides by the geometric mean, so a location parameter would cancel and
    a fixture carrying one would invite a reader to tune a number that
    changes nothing.

    Raises
    ------
    ValueError
        If the form is not one of the two, or a declared profile does not
        carry one entry per site.
    """
    form = str(declared["form"])
    if form == "declared":
        sizes = np.asarray(declared["values"], dtype=np.float64)
        if sizes.shape != (n_nodes,):
            msg = f"{path}: sizes.values has {sizes.shape} entries for {n_nodes} sites"
            raise ValueError(msg)
        return sizes
    if form == "lognormal":
        drawn = np.random.default_rng(int(declared["seed"])).normal(
            0.0, float(declared["log_sigma"]), size=n_nodes
        )
        return np.asarray(np.exp(drawn))
    msg = f"{path}: sizes.form {form!r} is not one of ['declared', 'lognormal']"
    raise ValueError(msg)


def load_potts_spots_params(path: Path) -> PottsSpotsParams:
    """Load and validate a per-site-field Potts fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    PottsSpotsParams
        The parsed, validated instance, its field already built.

    Raises
    ------
    ValueError
        If a required field is absent, the geometry is not one this module
        builds, ``alpha`` is not one entry per class, or a size is not
        positive.
    """
    raw = load_declared(path, _SPOTS_REQUIRED_FIELDS)

    geometry = str(raw["geometry"])
    if geometry not in _GEOMETRIES:
        msg = f"{path}: geometry {geometry!r} is not one of {sorted(_GEOMETRIES)}"
        raise ValueError(msg)
    shape = tuple(int(extent) for extent in raw["shape"])
    graph = _GEOMETRIES[geometry](
        shape, _boundary(path, raw["boundary"]), float(raw["coupling"])
    )

    n_classes = int(raw["n_classes"])
    if n_classes < 2:
        msg = f"{path}: n_classes must be >= 2, got {n_classes}"
        raise ValueError(msg)
    alpha = np.asarray(raw["alpha"], dtype=np.float64)
    if alpha.shape != (n_classes,):
        msg = f"{path}: alpha has shape {alpha.shape}, expected ({n_classes},)"
        raise ValueError(msg)

    sizes = _spots_sizes(path, raw["sizes"], graph.n_nodes)
    return PottsSpotsParams(
        graph=graph,
        n_classes=n_classes,
        alpha=alpha,
        sizes=sizes,
        field=spots_field(alpha, sizes),
        seed=int(raw["seed"]),
        n_samples=int(raw["n_samples"]),
        burn_in=int(raw["burn_in"]),
        tolerance=float(raw["tolerance"]),
    )
