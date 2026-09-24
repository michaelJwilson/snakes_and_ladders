"""Spin configurations for a `k`-state Potts model on a general graph.

An N-D lattice (:func:`snakes_and_ladders.sim.graph.lattice_graph`) is a constructed case
of :class:`~snakes_and_ladders.sim.graph.PottsGraph`, not a separate code path, per
``docs/tex/textbook.tex``, ``sec:potts`` (Mezard &
Montanari, ch. 2; Koller & Friedman for the general framing). A graph
recognized as a 1-D chain with an open boundary is sampled exactly, by the
same backward-message recursion :func:`snakes_and_ladders.sim.potts_chain.simulate_chains`
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from snakes_and_ladders.numerics import logsumexp, sample_rows
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    boundary_from_declared,
    lattice_graph,
    triangular_lattice_graph,
)


def site_field(
    field: np.ndarray, n_nodes: int, *, n_states: int | None = None
) -> np.ndarray:
    """``field`` as one row per site, whichever of the two shapes it arrives in.

    The one broadcast (issue #717): `search.maxflow` and
    `search.alpha_expansion` each carried their own before, with three
    signatures between them, and a fix to one did not land in the others.

    Parameters
    ----------
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` --- shared by every site
        --- or ``(n_nodes, n_states)``.
    n_nodes : int
        Sites the graph carries.
    n_states : int | None
        Columns the caller expects, checked where the caller knows it: a
        field with the wrong number of columns would otherwise surface as an
        ``IndexError`` inside a kernel rather than at the call that got it
        wrong. A cut passes ``2``; alpha expansion the state count it was
        given.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``, C-contiguous. A shared field is
        broadcast here and nowhere else, so a kernel below indexes one shape
        and a caller declaring either reaches the same code.

    Raises
    ------
    ValueError
        If ``field`` is neither shape, or carries other than ``n_states``
        columns where that is given. Refused rather than broadcast: a
        ``(n_states,)`` field on a graph whose node count equals the state
        count would otherwise be read as per-site, silently scoring a
        different model.

    Examples
    --------
    >>> site_field(np.array([0.0, 1.0]), 3).shape
    (3, 2)
    """
    if field.ndim == 1:
        rows = np.broadcast_to(field, (n_nodes, field.shape[0])).copy()
    elif field.shape[:1] == (n_nodes,) and field.ndim == 2:
        rows = np.ascontiguousarray(field)
    else:
        msg = (
            f"field has shape {field.shape}; expected (n_states,) or "
            f"({n_nodes}, n_states)"
        )
        raise ValueError(msg)
    if n_states is not None and rows.shape[1] != n_states:
        msg = (
            f"a field for {n_states} states must have {n_states} columns, "
            f"got {rows.shape[1]}"
        )
        raise ValueError(msg)
    return rows


@dataclass(frozen=True)
class SiteField:
    """A per-site field whose sign convention is part of its type (issue #921).

    The package reads a field two ways. The samplers and the ground-state
    solvers read a **log-weight** ``h``: ``E(s) = -sum_i h_i[s_i] - sum J
    [s_i = s_j]`` (:func:`energies`), and a cluster move accepts on
    ``exp(beta sum_C [h(new) - h(old)])``. The coupled model's field ``H`` of
    the external-field equation is an **energy** the labels minimize, its
    negation. A caller declares which it holds by the constructor it calls,
    and every consumer reads :attr:`log_weight`, so the sign is decided once
    and cannot be dropped at a call site. A bare array where a
    ``SiteField`` is accepted is read as a log-weight, as before.

    Parameters
    ----------
    log_weight : np.ndarray
        ``h``, shape ``(n_nodes, n_states)``.
    """

    log_weight: np.ndarray

    def __post_init__(self) -> None:
        if self.log_weight.ndim != 2:
            msg = f"a site field is (n_nodes, n_states); got {self.log_weight.shape}"
            raise ValueError(msg)

    @classmethod
    def from_log_weight(cls, field: np.ndarray) -> SiteField:
        """The field ``h`` a sampler reads, per site, as given."""
        return cls(np.ascontiguousarray(field, dtype=np.float64))

    @classmethod
    def from_energy(cls, field: np.ndarray) -> SiteField:
        """The field of an energy ``H`` the labels minimize: ``h = -H``."""
        return cls(np.ascontiguousarray(-np.asarray(field, dtype=np.float64)))

    @classmethod
    def widened(cls, field: np.ndarray, n_nodes: int) -> SiteField:
        """A log-weight shared by every site, ``(n_states,)``, as one row per site."""
        return cls(site_field(np.asarray(field, dtype=np.float64), n_nodes))


def log_weight_of(field: SiteField | np.ndarray) -> np.ndarray:
    """The log-weight a consumer reads: a :class:`SiteField`'s, or a bare array as given."""
    return field.log_weight if isinstance(field, SiteField) else np.asarray(field)


def energies(graph: PottsGraph, field: np.ndarray, states: np.ndarray) -> np.ndarray:
    """``E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]``, per configuration.

    The negated log weight :func:`snakes_and_ladders.likelihood.potts.log_weights`
    defines, generalized to a per-node field and vectorized over a block of
    configurations. Three modules scored a labelling with their own loop
    before issue #277; this is the one they call. It lives here rather than
    beside the oracle because `likelihood/` imports :func:`site_field` from
    this module, so `sim/` cannot import back.

    ``log_weights`` stays the independent referee and is not merged into
    this: an implementation that computes its own oracle is no longer
    refereed. The two sum the edge terms in different orders --- one gather
    and a pairwise sum here, a term per edge there --- so they agree to a
    relative ``1e-12`` rather than bitwise (issue #341, pinned in
    `tests/regression/search/test_maxflow.py`).

    The edge term never enters BLAS (issue #1044). As a gemv it split
    across the host's BLAS threads: at the release-q10 rung (5,041 sites)
    one labelling cost 7.0 to 12.2 ms at the default thread count against
    130 us at one, 92 to 98% of an annealed run's wall, and its value moved
    by up to ``4.5e-13`` with the thread count. Each row here is reduced
    alone, by the same pairwise sum at any thread count and block size.

    A single labelling is the ``n_configurations = 1`` case: a caller passes
    ``state[None]`` and reads element zero. Consolidating the other way ---
    the block form looping over a scalar one --- would give every caller the
    throughput of the slowest.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Its edge order fixes which coupling applies where.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
    states : np.ndarray
        Integer states, shape ``(..., n_nodes)``.

    Returns
    -------
    np.ndarray
        Shape ``states.shape[:-1]``.

    Raises
    ------
    ValueError
        If ``states`` does not carry one column per node. A silently
        broadcast mismatch would return energies for a different model.
    """
    if states.shape[-1] != graph.n_nodes:
        msg = (
            f"states has {states.shape[-1]} columns for a graph of "
            f"{graph.n_nodes} nodes"
        )
        raise ValueError(msg)
    rows = site_field(field, graph.n_nodes)
    total = rows[np.arange(graph.n_nodes), states].sum(axis=-1)
    if graph.edges:
        # One gather over every edge rather than a Python-level term per
        # edge: issue #341 measured that loop at 92% of this function's self
        # time, and #336 found it the term left in the Rust ground state's
        # wall clock. `take` along the last axis keeps a block row-major
        # where the fancy index hands back column-major, so the sum walks
        # each row contiguously and reduces a block row as it reduces that
        # labelling alone (issue #1044).
        ends = graph.edge_index
        agree = states.take(ends[:, 0], axis=-1) == states.take(ends[:, 1], axis=-1)
        total = total + (agree * graph.edge_coupling).sum(axis=-1)
    return -np.asarray(total)


def energy(graph: PottsGraph, field: np.ndarray, labelling: np.ndarray) -> float:
    """``E(s)`` of one labelling, for any state count: :func:`energies` at ``n = 1``.

    The scalar entry point `search.maxflow` and `search.alpha_expansion`
    each defined before issue #717; one labelling is ``labelling[None]``
    through the block form, element zero read back, and the field is widened
    to ``float64`` rows on the way in so an integer field scores as the
    model it names.
    """
    values = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    return float(energies(graph, values, np.asarray(labelling)[None])[0])


@dataclass(frozen=True)
class PottsMetrics:
    """What a Potts labelling means: its energy, from :func:`energy` (issue #778).

    A :class:`snakes_and_ladders.track.Metrics` over a labelling, satisfied
    structurally --- nothing here imports the seam. It sits beside
    :func:`energy` rather than in ``opt/potts.py``: ``opt/CLAUDE.md``'s "no
    application imports" rule, enforced by
    ``tests/regression/opt/test_opt_objective.py``, forbids ``opt`` from
    importing this module, and the energy is not written a second time.

    **One metric; the second the ticket listed is left out.** Neither a
    magnetization nor an unsatisfied-edge fraction is computed anywhere in
    this module, and a metrics set restates the science rather than defining
    it.

    The series is ``state_energy`` and not ``energy``:
    :func:`snakes_and_ladders.sample.potts_mcmc.anneal_potts` records the
    *best* energy so far under ``energy``, and this is the energy of the
    labelling it is handed.

    Parameters
    ----------
    graph : PottsGraph
        The instance whose edges and couplings the energy is over.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
    """

    graph: PottsGraph
    field: np.ndarray

    names: tuple[str, ...] = ("state_energy",)

    def __call__(self, state: np.ndarray) -> dict[str, float]:
        """``{"state_energy": energy(graph, field, state)}``."""
        return {"state_energy": energy(self.graph, self.field, state)}


def heat_bath_log_weights(
    field_row: np.ndarray,
    state: np.ndarray,
    neighbours: Sequence[int],
    couplings: Sequence[float],
    start: int,
    stop: int,
    beta: float = 1.0,
    chain_index: np.ndarray | None = None,
) -> np.ndarray:
    """One site's unnormalized log conditional: its own field plus its neighbours'.

    ``log p(s_i = k | rest) + c = beta * (h_ik + sum_j J_ij [k = s_j])``, the
    arithmetic every heat-bath sweep performs and three of them each wrote
    out before issue #277: the vectorized simulator :func:`_simulate_gibbs`,
    the sequential sampler `sample.potts_mcmc.sweeps.single_site_sweep`, and the
    Rust kernel. The *loops* stay separate --- one steps a chain in time, one
    is vectorized across independent chains, one is compiled --- because they
    are separate; what they share is this expression.

    **The couplings are summed in neighbour order and not reassociated.** That
    order is the graph's edge order from each end, which
    :meth:`~snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`
    fixes, and it is what makes the extraction bitwise rather than
    approximate. Reassociating for speed is a separate change carrying its
    own measurement (issue #277).

    **The adjacency arrives as Python sequences, and that is measured rather
    than stylistic.** The compressed rows are the storage, and the arrays are
    what the compiled kernels take; a Python-level sweep slicing and
    gathering from them per site measured 1.40 us a site against 1.03 us for
    the same additions over lists converted once per sweep, which was 22% of
    a 32x32 sweep. So a caller converts `compressed_adjacency`'s rows once
    (``tolist()``) and passes them with the bounds it would have sliced by.

    Parameters
    ----------
    field_row : np.ndarray
        The site's own field, shape ``(n_states,)``.
    state : np.ndarray
        The current configuration: ``(n_nodes,)`` for one chain, or
        ``(n_chains, n_nodes)`` for a block of independent chains. The
        returned shape follows it.
    neighbours : Sequence[int]
        The compressed neighbour rows,
        ``compressed_adjacency().neighbours.tolist()``.
    couplings : Sequence[float]
        The coupling per entry of ``neighbours``, in the same order.
    start, stop : int
        The site's row: ``offsets[i]`` and ``offsets[i + 1]``.
    beta : float
        Inverse temperature, applied to the whole conditional --- the model
        scaling `sample.potts_mcmc.tempered` states. At 1.0 the
        multiplication is the identity, so it is skipped rather than applied.
    chain_index : np.ndarray | None
        ``arange(n_chains)``, for the block form only. Passed in so the
        vectorized caller hoists it out of its site loop rather than
        rebuilding it per site; built here when omitted.

    Returns
    -------
    np.ndarray
        Shape ``(n_states,)`` or ``(n_chains, n_states)``, following
        ``state``. Unnormalized: the caller shifts, exponentiates and samples
        in whatever way its own loop needs.
    """
    if state.ndim == 1:
        local = field_row.copy()
        for position in range(start, stop):
            local[state[neighbours[position]]] += couplings[position]
    else:
        if chain_index is None:
            chain_index = np.arange(state.shape[0])
        local = np.tile(field_row, (state.shape[0], 1))
        for position in range(start, stop):
            local[chain_index, state[:, neighbours[position]]] += couplings[position]
    if beta != 1.0:
        local *= beta
    return local


def owner_rows(offsets: np.ndarray) -> np.ndarray:
    """Which site each entry of the compressed adjacency belongs to.

    ``offsets`` bounds site ``i``'s neighbours at ``offsets[i]:offsets[i + 1]``,
    so this is each site repeated by its degree. Built once per run and handed
    to :func:`local_fields`, which is called once per proposal (the allocation
    rule).
    """
    degree = np.diff(offsets)
    owner: np.ndarray = np.repeat(np.arange(degree.shape[0]), degree)
    return owner


def local_fields(
    rows: np.ndarray,
    state: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    owner: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Every site's unnormalized log conditional at once.

    :func:`heat_bath_log_weights` for one site, over the whole lattice in one
    reduction: a gradient-informed proposal reads every site's conditional per
    step (:mod:`snakes_and_ladders.sample.balanced`), and a Python loop over
    sites to build it would cost a heat-bath sweep per proposal.

    The couplings reach the right cell by one ``bincount`` over the compressed
    rows, keyed on ``site * n_states + state[neighbour]``. The entries are
    walked in the order :func:`heat_bath_log_weights` walks them, but the field
    is added **after** their sum rather than accumulated into, so the two agree
    to a relative ``1e-12`` and not bitwise
    (`tests/regression/sim/test_potts_simulate.py`). The site's own label does
    not appear: a lattice has no self-coupling, which is what makes a
    single-site energy difference exact from this array alone.

    Parameters
    ----------
    rows : np.ndarray
        The field as one row per site, shape ``(n_nodes, n_states)``, from
        :func:`site_field`.
    state : np.ndarray
        The current configuration, shape ``(n_nodes,)``.
    neighbours, couplings : np.ndarray
        The compressed adjacency's second and third rows.
    owner : np.ndarray
        :func:`owner_rows` of the first, hoisted by the caller.
    beta : float
        Inverse temperature, applied to the whole conditional as
        :func:`heat_bath_log_weights` applies it. At 1.0 the multiplication is
        the identity and is skipped.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``.

    Examples
    --------
    >>> offsets = np.array([0, 1, 2])
    >>> local_fields(
    ...     np.zeros((2, 2)),
    ...     np.array([0, 1]),
    ...     np.array([1, 0]),
    ...     np.array([0.5, 0.5]),
    ...     owner_rows(offsets),
    ... )
    array([[0. , 0.5],
           [0.5, 0. ]])
    """
    n_nodes, n_states = rows.shape
    counted = np.bincount(
        owner * n_states + state[neighbours],
        weights=couplings,
        minlength=n_nodes * n_states,
    )
    local: np.ndarray = rows + counted.reshape(n_nodes, n_states)
    if beta != 1.0:
        local *= beta
    return local


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
        A fixture may declare it as a size tilt instead of a table (issue
        #906): a mapping of ``alpha`` (one per state), ``size_spread`` and
        ``size_seed``, the per-site sizes drawn lognormally at that spread
        from ``np.random.default_rng(size_seed)`` and the field built by
        :func:`spatio_only_field`, as
        :func:`snakes_and_ladders.search.ground_state.lattice_rung` builds it.
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
    alpha : np.ndarray | None
        The class ladder of a size-tilted field, shape ``(n_states,)``;
        ``None`` for a field declared as a table.
    sizes : np.ndarray | None
        The per-site sizes of a size-tilted field, shape ``(n_nodes,)``;
        ``None`` for a field declared as a table.
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
    alpha: np.ndarray | None = None
    sizes: np.ndarray | None = None

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a Potts-lattice fixture's declared mapping.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        Raises
        ------
        ValueError
            If a required field is missing, ``field`` has the wrong shape, or a
            size is too small to identify the model.
        """
        shape = tuple(int(extent) for extent in declared["shape"])
        n_states = int(declared["n_states"])
        if n_states < 2:
            msg = f"{path}: n_states must be >= 2, got {n_states}"
            raise ValueError(msg)

        n_nodes = int(np.prod(shape))
        alpha: np.ndarray | None = None
        sizes: np.ndarray | None = None
        raw = declared["field"]
        if isinstance(raw, Mapping):
            alpha, sizes = _size_tilt(path, raw, n_states, n_nodes)
            field = spatio_only_field(alpha, sizes)
        else:
            field = np.asarray(raw, dtype=np.float64)
        if field.shape not in {(n_states,), (n_nodes, n_states)}:
            msg = (
                f"{path}: field has shape {field.shape}, expected ({n_states},) "
                f"or ({n_nodes}, {n_states})"
            )
            raise ValueError(msg)

        return cls(
            shape=shape,
            boundary=boundary_from_declared(path, declared["boundary"]),
            n_states=n_states,
            coupling=_coupling(path, declared["coupling"], n_states),
            field=field,
            seed=int(declared["seed"]),
            n_samples=int(declared["n_samples"]),
            burn_in=int(declared["burn_in"]),
            tolerance=float(declared["tolerance"]),
            alpha=alpha,
            sizes=sizes,
        )


#: The keys a size-tilted field declares.
SIZE_TILT = frozenset({"alpha", "size_spread", "size_seed"})


def _size_tilt(
    path: Path, raw: Mapping[str, Any], n_states: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """The class ladder and the per-site sizes a size-tilted field declares.

    Raises
    ------
    ValueError
        If a key is missing or unknown, the ladder is not one per state, or
        the spread is not positive.
    """
    if set(raw) != SIZE_TILT:
        msg = (
            f"{path}: a size-tilted field declares exactly {sorted(SIZE_TILT)}, "
            f"got {sorted(raw)}"
        )
        raise ValueError(msg)
    alpha = np.asarray(raw["alpha"], dtype=np.float64)
    if alpha.shape != (n_states,):
        msg = f"{path}: alpha has shape {alpha.shape}, expected ({n_states},)"
        raise ValueError(msg)
    spread = float(raw["size_spread"])
    if not spread > 0.0:
        msg = f"{path}: size_spread must be positive, got {spread}"
        raise ValueError(msg)
    sizes = np.random.default_rng(int(raw["size_seed"])).lognormal(
        0.0, spread, size=n_nodes
    )
    return alpha, sizes


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
    ``snakes_and_ladders.sim.potts_chain.simulate_chains`` used before this module existed: the
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

    The conditional is :func:`heat_bath_log_weights` and the neighbours are
    the graph's compressed rows, which is what a site's update indexes rather
    than a list of Python tuples per node (issue #277).
    """
    n_states = field.shape[1]
    adjacency = graph.compressed_adjacency()
    bounds = adjacency.offsets.tolist()
    neighbours = adjacency.neighbours.tolist()
    couplings = adjacency.couplings.tolist()

    state = rng.integers(0, n_states, size=(n_samples, graph.n_nodes))
    chain_index = np.arange(n_samples)

    for _ in range(burn_in):
        for node in range(graph.n_nodes):
            local = heat_bath_log_weights(
                field[node],
                state,
                neighbours,
                couplings,
                bounds[node],
                bounds[node + 1],
                chain_index=chain_index,
            )
            state[:, node] = sample_rows(rng, _softmax(local, axis=1), chain_index)

    return state


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    result: np.ndarray = weights / weights.sum(axis=axis, keepdims=True)
    return result


# --- a Potts prior whose external field varies per site (issue #413) ---------

#: The lattice constructors a spatio_only fixture may name, so the geometry is a
#: declared word rather than a loader that only ever builds a square.
_GEOMETRIES: dict[str, Callable[[Any, BoundaryCondition, float], PottsGraph]] = {
    "square": lattice_graph,
    "triangular": triangular_lattice_graph,
}

_SPATIO_ONLY_REQUIRED_FIELDS = frozenset(
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
class SpatioOnlyParams:
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

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _SPATIO_ONLY_REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a per-site-field Potts fixture's declared mapping.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The parsed, validated instance, its field already built.

        Raises
        ------
        ValueError
            If a required field is absent, the geometry is not one this module
            builds, ``alpha`` is not one entry per class, or a size is not
            positive.
        """
        geometry = str(declared["geometry"])
        if geometry not in _GEOMETRIES:
            msg = f"{path}: geometry {geometry!r} is not one of {sorted(_GEOMETRIES)}"
            raise ValueError(msg)
        shape = tuple(int(extent) for extent in declared["shape"])
        graph = _GEOMETRIES[geometry](
            shape,
            boundary_from_declared(path, declared["boundary"]),
            float(declared["coupling"]),
        )

        n_classes = int(declared["n_classes"])
        if n_classes < 2:
            msg = f"{path}: n_classes must be >= 2, got {n_classes}"
            raise ValueError(msg)
        alpha = np.asarray(declared["alpha"], dtype=np.float64)
        if alpha.shape != (n_classes,):
            msg = f"{path}: alpha has shape {alpha.shape}, expected ({n_classes},)"
            raise ValueError(msg)

        sizes = _spatio_only_sizes(path, declared["sizes"], graph.n_nodes)
        return cls(
            graph=graph,
            n_classes=n_classes,
            alpha=alpha,
            sizes=sizes,
            field=spatio_only_field(alpha, sizes),
            seed=int(declared["seed"]),
            n_samples=int(declared["n_samples"]),
            burn_in=int(declared["burn_in"]),
            tolerance=float(declared["tolerance"]),
        )


def spatio_only_field(alpha: np.ndarray, sizes: np.ndarray) -> np.ndarray:
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
    >>> spatio_only_field(np.array([1.0, 0.0]), np.array([1.0, 4.0])).round(4)
    array([[-0.6931,  0.    ],
           [ 0.6931,  0.    ]])
    """
    if not bool((sizes > 0.0).all()):
        msg = f"every size must be positive, got a minimum of {sizes.min()}"
        raise ValueError(msg)
    centred = np.log(sizes) - float(np.log(sizes).mean())
    return np.asarray(centred[:, np.newaxis] * alpha[np.newaxis, :])


def _spatio_only_sizes(path: Path, declared: Any, n_nodes: int) -> np.ndarray:
    """The per-site covariate, declared outright or drawn from a declared law.

    Two forms, because the two sizes need different things of the file. At
    nine sites the profile is written out, so a reader sees which site is
    large; at 5,041 the file states the seed and the shape parameter and the
    sizes are what they imply, as
    :mod:`snakes_and_ladders.sim.count_pairs` states its counts.

    Only ``log_sigma`` is declared for the drawn form: ``spatio_only_field``
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


# --- a Potts prior whose field favours one state per tile (issue #1050) -----

_SPATIO_TILING_REQUIRED_FIELDS = frozenset(
    {
        "geometry",
        "shape",
        "boundary",
        "n_states",
        "coupling",
        "tiles",
        "states",
        "strengths",
    }
)

#: The keys a tiling fixture's ``tiles`` declares: the seed its centres are
#: drawn from and ``k``, the tile count.
_TILES_KEYS = frozenset({"seed", "k"})


def tile_partition(graph: PottsGraph, k: int, rng: np.random.Generator) -> np.ndarray:
    """Every node assigned to the nearest of ``k`` centres drawn from ``rng``: a seeded Voronoi tiling.

    The centres are ``k`` distinct nodes drawn uniformly without replacement,
    in draw order; a node joins the centre nearest it in graph distance
    (hops), and a tie goes to the centre drawn first. Tile ``t`` holds centre
    ``t``.

    **Every tile is connected, and the tie rule is why.** Take a node ``v``
    joined to centre ``c`` at distance ``d``, and its predecessor ``u`` on a
    shortest path to ``c``, at ``d - 1``. A centre ``c'`` nearer ``u`` than
    ``d - 1`` would be nearer ``v`` than ``d``; one at exactly ``d - 1`` would
    be at most ``d`` from ``v``, so it ties with ``c`` there and was drawn
    after it. So ``u`` joins ``c`` too, and induction on ``d`` walks ``v``
    back to ``c`` inside the tile. A tie broken by anything other than one
    order over the centres --- a random draw per node --- can strand a node.

    Parameters
    ----------
    graph : PottsGraph
        Connected.
    k : int
        The tile count, ``1 <= k <= graph.n_nodes``.
    rng : np.random.Generator
        Draws the centres, and nothing else.

    Returns
    -------
    np.ndarray
        ``int64``, shape ``(n_nodes,)``, the tile of each node.

    Raises
    ------
    ValueError
        If ``k`` is out of range, or a node is reached from no centre, which a
        disconnected graph allows.

    Examples
    --------
    >>> from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
    >>> chain = lattice_graph((6,), BoundaryCondition.OPEN, 1.0)

    The generator draws centres 0 and 4; node 2 is two hops from each and
    joins the first drawn.

    >>> tile_partition(chain, 2, np.random.default_rng(3))
    array([0, 0, 0, 1, 1, 1])
    """
    if not 1 <= k <= graph.n_nodes:
        msg = f"k must be in [1, {graph.n_nodes}], got {k}"
        raise ValueError(msg)
    centres = rng.choice(graph.n_nodes, size=k, replace=False)
    offsets, neighbours, _ = graph.compressed_adjacency()
    # Hop counts from each centre, one row per centre in draw order, by a
    # breadth-first search whose frontier expands one layer per step.
    distances = np.full((k, graph.n_nodes), np.inf)
    for row, centre in enumerate(centres):
        reached = distances[row]
        reached[centre] = 0.0
        frontier = np.array([centre])
        hops = 0
        while frontier.size:
            hops += 1
            # Every neighbour of the frontier, gathered through the offsets.
            starts = offsets[frontier]
            counts = offsets[frontier + 1] - starts
            first = np.cumsum(counts) - counts
            gathered = np.arange(int(counts.sum())) - np.repeat(first - starts, counts)
            found = np.unique(neighbours[gathered])
            frontier = found[np.isinf(reached[found])]
            reached[frontier] = hops
    if not bool(np.isfinite(distances.min(axis=0)).all()):
        msg = "a node is reached from no centre; the graph is not connected"
        raise ValueError(msg)
    # `argmin` returns the first row at the minimum, which is the tie rule above.
    return np.asarray(np.argmin(distances, axis=0), dtype=np.int64)


def tiling_field(
    tiles: np.ndarray,
    states: np.ndarray,
    strengths: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """``h[i, m] = s_t [m = a_t]`` for ``i`` in tile ``t``: one favoured state per tile.

    Not linear in the state, where :func:`spatio_only_field` is: a tile
    rewards its own state and no other, so no two-state reduction holds and
    the optimum at ``q >= 3`` can use every state the tiling plants.

    Parameters
    ----------
    tiles : np.ndarray
        Tile per node, ``int``, shape ``(n_nodes,)``, values in ``[0, k)``.
    states : np.ndarray
        ``a_t``, the favoured state per tile, ``int``, shape ``(k,)``, values
        in ``[0, n_states)``.
    strengths : np.ndarray
        ``s_t`` per tile, shape ``(k,)``, each positive.
    n_states : int
        ``q``.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``: ``s_t`` at each node's favoured state,
        zero elsewhere.

    Raises
    ------
    ValueError
        If the lengths disagree, a favoured state is out of range, a tile is
        out of range, or a strength is not positive.

    Examples
    --------
    >>> tiling_field(np.array([0, 0, 1]), np.array([2, 0]), np.array([1.5, 0.5]), 3)
    array([[0. , 0. , 1.5],
           [0. , 0. , 1.5],
           [0.5, 0. , 0. ]])
    """
    tiles = np.asarray(tiles, dtype=np.int64)
    states = np.asarray(states, dtype=np.int64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if states.shape != strengths.shape or states.ndim != 1:
        msg = (
            f"one state and one strength per tile, got {states.shape} "
            f"and {strengths.shape}"
        )
        raise ValueError(msg)
    if states.size and not bool(((states >= 0) & (states < n_states)).all()):
        msg = f"every favoured state must be in [0, {n_states}), got {states}"
        raise ValueError(msg)
    if not bool(((tiles >= 0) & (tiles < states.size)).all()):
        msg = f"every tile must be in [0, {states.size})"
        raise ValueError(msg)
    if not bool((strengths > 0.0).all()):
        msg = f"every strength must be positive, got a minimum of {strengths.min()}"
        raise ValueError(msg)
    field = np.zeros((tiles.size, n_states))
    field[np.arange(tiles.size), states[tiles]] = strengths[tiles]
    return field


@dataclass(frozen=True)
class SpatioTilingParams:
    """A Potts prior whose field favours one planted state per tile.

    The lattice of :class:`SpatioOnlyParams` with a field that is not linear
    in the state: the sites are split into ``k`` connected tiles by
    :func:`tile_partition`, and tile ``t`` rewards state ``a_t`` by ``s_t``
    (:func:`tiling_field`). The strengths are declared across the coupling,
    so some tiles are held by their field and others give way to a
    neighbour's state.

    An instance and not a draw: the planted states are the truth a ground
    state is read against, and no sampler reads this fixture.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, of the declared geometry, shape, boundary and coupling.
    n_states : int
        ``q``.
    tiles : np.ndarray
        Tile per site, shape ``(n_nodes,)``.
    states : np.ndarray
        ``a_t``, the favoured state per tile, shape ``(k,)``.
    strengths : np.ndarray
        ``s_t``, shape ``(k,)``.
    field : np.ndarray
        ``h``, shape ``(n_nodes, q)``, from the three above.
    tiling_seed : int
        Seed of the generator the tiles' centres are drawn from.
    """

    graph: PottsGraph
    n_states: int
    tiles: np.ndarray
    states: np.ndarray
    strengths: np.ndarray
    field: np.ndarray
    tiling_seed: int

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _SPATIO_TILING_REQUIRED_FIELDS

    @property
    def n_tiles(self) -> int:
        """``k``."""
        return int(self.states.size)

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a tiling-field Potts fixture's declared mapping.

        Raises
        ------
        ValueError
            If the geometry is not one this module builds, ``tiles`` does not
            declare exactly a seed and ``k``, or the states and strengths are
            not one per tile.
        """
        geometry = str(declared["geometry"])
        if geometry not in _GEOMETRIES:
            msg = f"{path}: geometry {geometry!r} is not one of {sorted(_GEOMETRIES)}"
            raise ValueError(msg)
        shape = tuple(int(extent) for extent in declared["shape"])
        graph = _GEOMETRIES[geometry](
            shape,
            boundary_from_declared(path, declared["boundary"]),
            float(declared["coupling"]),
        )
        n_states = int(declared["n_states"])
        if n_states < 3:
            msg = (
                f"{path}: n_states must be >= 3, got {n_states}; at two the "
                "field is linear in the state and the instance is spatio_only's"
            )
            raise ValueError(msg)
        raw = declared["tiles"]
        if not isinstance(raw, Mapping) or set(raw) != _TILES_KEYS:
            msg = f"{path}: tiles declares exactly {sorted(_TILES_KEYS)}"
            raise ValueError(msg)
        k = int(raw["k"])
        states = np.asarray(declared["states"], dtype=np.int64)
        strengths = np.asarray(declared["strengths"], dtype=np.float64)
        if states.shape != (k,) or strengths.shape != (k,):
            msg = (
                f"{path}: states {states.shape} and strengths {strengths.shape} "
                f"must each be one per tile, ({k},)"
            )
            raise ValueError(msg)
        seed = int(raw["seed"])
        tiles = tile_partition(graph, k, np.random.default_rng(seed))
        try:
            field = tiling_field(tiles, states, strengths, n_states)
        except ValueError as error:
            msg = f"{path}: {error}"
            raise ValueError(msg) from error
        return cls(
            graph=graph,
            n_states=n_states,
            tiles=tiles,
            states=states,
            strengths=strengths,
            field=field,
            tiling_seed=seed,
        )
