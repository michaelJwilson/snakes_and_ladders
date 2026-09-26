"""Undirected graphs for the Potts model, and N-D lattice constructors.

An N-D lattice is a constructed case of a general graph, not a second
representation: :func:`lattice_graph` builds a :class:`PottsGraph`, and
nothing downstream distinguishes a lattice from a hand-built graph except
the ``shape``/``boundary`` metadata a lattice happens to carry.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np

from sal.incidence import SparseIncidence


class BoundaryCondition(StrEnum):
    """How a lattice's outermost sites connect.

    A ``StrEnum`` rather than a bare string so ``mypy --strict`` rejects an
    unrecognized boundary at the call site rather than the constructor
    raising at run time, and so the yaml loader has one place to parse it.
    The same choice ``sal.search.infer.MoveSet`` and
    ``sal.learn.tree.RewardModel`` make.
    """

    OPEN = "open"
    PERIODIC = "periodic"


def boundary_from_declared(path: Path, raw: object) -> BoundaryCondition:
    """Parse a yaml boundary field, naming the file when it is unrecognized.

    The five fixture loaders that read a boundary parsed it two ways: two
    through this check and three through ``BoundaryCondition(str(raw))``,
    whose message names neither the file nor the recognized values (issue
    #862). It lives beside the enum so every loader reaches it without
    reaching into another model's module.

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
        here rather than in :func:`lattice_graph`, which takes the enum and
        so cannot be handed a bad string at all.
    """
    try:
        return BoundaryCondition(str(raw))
    except ValueError:
        recognized = sorted(condition.value for condition in BoundaryCondition)
        msg = f"{path}: boundary must be one of {recognized}, got {raw!r}"
        raise ValueError(msg) from None


def _read_only(values: np.ndarray) -> np.ndarray:
    """Mark a cached derived array unwritable, and return it.

    A `cached_property` hands the *same* array to every caller, so a consumer
    that writes through it corrupts the graph for every later call --- and the
    graph is frozen, so nothing else about it can change under a caller. The
    flag turns that from silent corruption into a `ValueError` at the write.

    It also decides a real case: `torch.as_tensor` on a writable NumPy array
    shares its buffer, so an in-place tensor op would reach back into the
    cache. Against an unwritable one PyTorch copies instead (with a warning
    the callers here avoid by copying explicitly), which is the behaviour the
    consumers want.
    """
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class CompressedAdjacency:
    """A graph's neighbour rows in compressed-row form, couplings beside them.

    The package's one neighbour structure (issue #277), carried by name
    rather than as three positional arrays (issue #865). Built once per
    graph, not once per call: :attr:`PottsGraph.incidence` caches it and
    :meth:`PottsGraph.compressed_adjacency` returns that instance, so the
    Potts cluster move issue #586 measured at 2.755 ms of 3.425 ms allocates
    nothing it did not before and reads a field where it unpacked a tuple.

    Parameters
    ----------
    offsets : np.ndarray
        ``int64`` of length ``n_nodes + 1``. Node ``i``'s neighbours are
        ``neighbours[offsets[i]:offsets[i + 1]]``.
    neighbours : np.ndarray
        ``int64`` of length ``2 * n_edges``, each node's neighbours in the
        graph's edge order.
    couplings : np.ndarray
        ``float64`` of length ``2 * n_edges``, the coupling on the edge the
        entry of ``neighbours`` at the same position names.
    """

    offsets: np.ndarray
    neighbours: np.ndarray
    couplings: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(offsets, neighbours, couplings)``: the order callers unpack."""
        yield from (self.offsets, self.neighbours, self.couplings)


@dataclass(frozen=True)
class Endpoints:
    """The edges as three columns in edge order: the coordinate form.

    The companion to :class:`CompressedAdjacency` for the consumer that
    scores every edge at once rather than walking one node's row.

    Parameters
    ----------
    first, second : np.ndarray
        ``int64`` of length ``n_edges``, the two ends of each edge.
    coupling : np.ndarray
        ``float64`` of length ``n_edges``, the coupling on that edge.
    """

    first: np.ndarray
    second: np.ndarray
    coupling: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(first, second, coupling)``: the order callers unpack."""
        yield from (self.first, self.second, self.coupling)


def _transposes(
    forward: np.ndarray,
    forward_weights: np.ndarray,
    backward: np.ndarray,
    backward_weights: np.ndarray,
    n_nodes: int,
) -> bool:
    """Whether the upper entries and the transposed lower ones are one multiset.

    Compared on one ``int64`` key per pair, ``i * n_nodes + j``, which a
    single sort orders; the weights are compared in that order, and only a
    repeated key --- a doubled bond --- needs the second sort key.
    """
    if forward.shape != backward.shape:
        return False
    forward_key = forward[:, 0] * n_nodes + forward[:, 1]
    backward_key = backward[:, 0] * n_nodes + backward[:, 1]
    forward_order = np.argsort(forward_key, kind="stable")
    backward_order = np.argsort(backward_key, kind="stable")
    if not np.array_equal(forward_key[forward_order], backward_key[backward_order]):
        return False
    if np.array_equal(forward_weights[forward_order], backward_weights[backward_order]):
        return True
    forward_order = np.lexsort((forward_weights, forward_key))
    backward_order = np.lexsort((backward_weights, backward_key))
    return bool(
        np.array_equal(forward_weights[forward_order], backward_weights[backward_order])
    )


@dataclass(frozen=True)
class PottsGraph:
    """An undirected graph carrying a per-edge Potts coupling.

    Parameters
    ----------
    n_nodes : int
        Number of sites, indexed ``0`` to ``n_nodes - 1``.
    edges : tuple[tuple[int, int], ...]
        Undirected edges as ``(i, j)`` pairs. Not deduplicated: a periodic
        lattice of extent 2 along some dimension legitimately produces the
        same pair twice (the "+1" and "-1" neighbour coincide), which reads
        as a doubled bond rather than a single one -- the edge count formula
        `lattice_graph` is validated against expects exactly this.
    coupling : tuple[float, ...]
        Coupling ``J`` per edge, same order and length as ``edges``.
    shape : tuple[int, ...] | None
        The lattice extent along each dimension, if this graph was built by
        :func:`lattice_graph`; ``None`` for a graph built directly.
    boundary : BoundaryCondition | None
        The boundary condition, if this graph was built by
        :func:`lattice_graph`; ``None`` for a graph built directly.

    Raises
    ------
    ValueError
        If ``coupling`` does not carry one entry per edge, or an edge names a
        node outside ``[0, n_nodes)``. Both are invariants every consumer
        assumes -- :func:`sal.sim.potts.simulate_potts` indexes the spin
        array by node and the coupling array by edge position -- so they are
        checked where the graph is built rather than where it is used.
    """

    n_nodes: int
    edges: tuple[tuple[int, int], ...]
    coupling: tuple[float, ...]
    shape: tuple[int, ...] | None = None
    boundary: BoundaryCondition | None = None

    def __post_init__(self) -> None:
        if len(self.coupling) != len(self.edges):
            msg = (
                f"coupling has {len(self.coupling)} entries for "
                f"{len(self.edges)} edges -- one per edge is required"
            )
            raise ValueError(msg)
        index = self.edge_index
        outside = (index < 0) | (index >= self.n_nodes)
        if outside.any():
            edge = tuple(
                int(node) for node in index[int(np.argmax(outside.any(axis=1)))]
            )
            msg = f"edge {edge} names a node outside [0, {self.n_nodes})"
            raise ValueError(msg)

    @classmethod
    def from_csr(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        coupling: np.ndarray | float,
    ) -> PottsGraph:
        """The graph a symmetric CSR adjacency describes, each undirected edge once (issue #1081).

        A caller holding a sparse adjacency --- ``scipy.sparse``'s, or a
        k-nearest-neighbour graph's --- built ``edges`` as Python tuples to
        get here. This reads the three CSR arrays with NumPy alone and keeps
        the entry ``(i, j)`` with ``i < j`` of each symmetric pair, in
        row-major order, so a graph of 10^6 sites is built without a Python
        loop in the validation. :attr:`edge_coupling` is seeded with the array
        it built rather than rebuilt from the tuple.

        Parameters
        ----------
        indptr : np.ndarray
            Row pointers, shape ``(n_nodes + 1,)``.
        indices : np.ndarray
            Column indices, shape ``(indptr[-1],)``.
        coupling : np.ndarray | float
            ``J`` per stored entry, shape ``(indptr[-1],)``, or one ``J`` for
            every edge. Entry ``(i, j)`` and entry ``(j, i)`` must carry the
            same value.

        Raises
        ------
        ValueError
            If the adjacency is not symmetric --- an entry without its
            transpose, or a transpose of another coupling --- or has a self
            loop, which a Potts coupling does not define.
        """
        indptr = np.asarray(indptr, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        n_nodes = indptr.shape[0] - 1
        rows = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
        weights = np.broadcast_to(np.asarray(coupling, dtype=np.float64), indices.shape)
        if np.any(rows == indices):
            node = int(rows[np.argmax(rows == indices)])
            msg = f"node {node} has a self loop, which a Potts coupling does not define"
            raise ValueError(msg)
        upper, lower = rows < indices, rows > indices
        forward = np.stack([rows[upper], indices[upper]], axis=1)
        backward = np.stack([indices[lower], rows[lower]], axis=1)
        if not _transposes(forward, weights[upper], backward, weights[lower], n_nodes):
            msg = "the CSR adjacency is not symmetric: every (i, j, J) needs its (j, i, J)"
            raise ValueError(msg)
        edge_coupling = np.ascontiguousarray(weights[upper])
        graph = cls(
            n_nodes,
            tuple(zip(forward[:, 0].tolist(), forward[:, 1].tolist(), strict=True)),
            tuple(edge_coupling.tolist()),
        )
        # The couplings this built are the array the cached property would
        # rebuild from the tuple; seeding it skips that conversion.
        graph.__dict__["edge_coupling"] = _read_only(edge_coupling)
        return graph

    @cached_property
    def edge_index(self) -> np.ndarray:
        """The edges as an ``(n_edges, 2)`` array of node indices, derived once.

        `edges` is a tuple of pairs, which is what a fixture declares and what
        `__post_init__` validates, and is not what any consumer uses: every one
        of them indexes an array by it. Converting is ``O(n_edges)`` and
        allocates, so a consumer that writes ``np.asarray(graph.edges)`` inside
        a function pays that on every call --- 604 of those conversions were
        81% of a 200-step `anneal_potts` at 32x32 (issue #623).

        Empty is ``(0, 2)`` rather than ``(0,)``, so a caller may index the
        columns of a graph with no edges without a special case.
        """
        if not self.edges:
            return _read_only(np.empty((0, 2), dtype=np.int64))

        return _read_only(np.asarray(self.edges, dtype=np.int64).reshape(-1, 2))

    @cached_property
    def edge_coupling(self) -> np.ndarray:
        """The couplings as a ``float64`` array, in the graph's edge order.

        The companion to :attr:`edge_index`, and derived for the same reason:
        one coupling per edge is this class's invariant, so the array form is
        the class's to hold rather than each consumer's to rebuild.
        """
        return _read_only(np.asarray(self.coupling, dtype=np.float64))

    def weighted_edges(self) -> Iterator[tuple[tuple[int, int], float]]:
        """Yield each edge with the coupling on it, in the graph's edge order.

        Eight modules walked these two tuples in step with a hand-written
        ``zip(graph.edges, graph.coupling, strict=True)``, twelve times
        (issue #230). The pairing is an invariant of this class -- one
        coupling per edge, checked in ``__post_init__`` -- so it belongs to
        the class rather than being restated by every consumer, and a
        consumer that forgets ``strict=True`` silently truncates to the
        shorter tuple.

        Yields
        ------
        tuple[tuple[int, int], float]
            ``((first, second), coupling)`` per edge.
        """
        yield from zip(self.edges, self.coupling, strict=True)

    def compressed_adjacency(self) -> CompressedAdjacency:
        """The adjacency in compressed-row form: offsets, neighbours, couplings.

        Neighbour ``j`` of node ``i`` sits at ``neighbours[offsets[i]:offsets[i + 1]]``
        with the coupling on that edge at the same position of ``couplings``,
        in the graph's edge order from each end. That order is part of the
        contract: a sweep consumes one draw per site in it, so a permutation
        would change which draw a site sees without changing any distribution
        a chi-square could catch.

        **This is the package's only adjacency.** Six builders returned this
        object in two shapes until issue #277 retired five of them; one
        contiguous array instead of a list of Python lists is the layout rule
        root ``CLAUDE.md`` states, so a neighbour walk is a stride rather
        than a pointer chase, and it is what a compiled kernel takes without
        marshalling per node. `tests/regression/test_duplication_guards.py`
        fails a second builder.

        The arrays are :attr:`incidence`'s, shared by every caller and
        read-only; a caller that needs to write takes a copy.

        Returns
        -------
        CompressedAdjacency
            ``offsets`` (``int64``, length ``n_nodes + 1``), ``neighbours``
            (``int64``) and ``couplings`` (``float64``), the last two of length
            ``2 * n_edges``. Read-only, and the same arrays on every call; it
            iterates in that order, so a caller unpacking it is unchanged.
        """
        return self.incidence

    @cached_property
    def incidence(self) -> CompressedAdjacency:
        """The compressed adjacency, derived once per graph and then reused.

        Each edge contributes its two directed entries in edge order, so a
        *stable* sort by owning node leaves each row in edge order -- the
        order :meth:`compressed_adjacency`'s contract fixes.

        Derived once because the survey issue #586 ran found it derived per
        call: every caller asked the graph to rebuild it, and
        :func:`sal.search.spatio_sequential._wolff_update` asks
        once per cluster move, where a move touches one cluster. At a 64x64
        periodic lattice the build was 2.755 ms of a 3.425 ms move.

        The three arrays are returned to every caller, so they are read-only:
        an aliased buffer a consumer wrote into would be a second graph that
        nothing declares, and NumPy refuses the write where it happens rather
        than leaving it to be found in a distribution.
        """
        ends = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        incidence = SparseIncidence.from_pairs(
            self.n_nodes, self.n_nodes, ends.reshape(-1), ends[:, ::-1].reshape(-1)
        )
        couplings = incidence.gather(
            np.repeat(np.asarray(self.coupling, dtype=np.float64), 2)
        )
        arrays = (incidence.offsets, incidence.indices, couplings)
        for array in arrays:
            array.flags.writeable = False
        return CompressedAdjacency(*arrays)

    @cached_property
    def endpoints(self) -> Endpoints:
        """The edges as three arrays in edge order: first, second, coupling.

        The coordinate form beside :attr:`incidence`'s compressed one, and
        for the other kind of consumer. A sweep walks a node's neighbours and
        wants the compressed rows; a scorer evaluates every edge at once and
        wants the two endpoint columns, which is what
        :func:`sal.likelihood.potts.log_weights` gathers with.

        Derived once for the reason :attr:`incidence` is (issue #586's
        recompute-or-store rule): the alternative is
        ``np.asarray(graph.edges)`` inside a loop, and
        :func:`sal.sample.potts_mcmc.anneal_potts` scores once
        per sweep.

        Read-only, as :attr:`incidence` is and for the same reason.

        Returns
        -------
        Endpoints
            ``first`` and ``second`` of dtype ``int64`` and ``coupling`` of
            ``float64``, each of length ``len(self.edges)``. It iterates in
            that order, so a caller unpacking it is unchanged.
        """
        ends = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        arrays = (
            np.ascontiguousarray(ends[:, 0]),
            np.ascontiguousarray(ends[:, 1]),
            np.asarray(self.coupling, dtype=np.float64),
        )
        for array in arrays:
            array.flags.writeable = False
        return Endpoints(*arrays)

    def is_open_chain(self) -> bool:
        """Whether this graph is a 1-D lattice with an open boundary.

        The one case with a cheap exact sampler (:mod:`sal.sim.potts`'s
        backward-message recursion, the same one
        :func:`sal.opt.potts.log_partition` sums via transfer matrix) --
        a periodic ring is a different, cyclic recursion, so it is excluded
        here rather than approximated by the open-chain code.
        """
        return (
            self.shape is not None
            and len(self.shape) == 1
            and self.boundary is BoundaryCondition.OPEN
        )


def _lattice(
    shape: tuple[int, ...],
    offsets: tuple[tuple[int, ...], ...],
    boundary: BoundaryCondition,
    coupling: float,
) -> PottsGraph:
    """A lattice over ``shape``, one edge per offset per site, wrapped or dropped at the edge.

    The core :func:`lattice_graph` and :func:`triangular_lattice_graph` each
    wrote out (issue #862): the extent guard, the row-major index, the sweep
    that wraps a target under a periodic boundary and skips it under an open
    one, and the graph built from what the sweep collected. The two differ in
    their offsets alone --- the unit vectors for the square lattice, those
    plus the cell diagonal for the triangular one --- so the offsets are the
    argument.

    Parameters
    ----------
    shape : tuple[int, ...]
        Extent along each dimension, each ``>= 2``.
    offsets : tuple[tuple[int, ...], ...]
        The neighbour each site is joined to, as a displacement per
        dimension. Applied in the order given, so the edge order is the
        caller's.
    boundary : BoundaryCondition
        ``PERIODIC`` wraps a target that leaves the grid; ``OPEN`` drops it.
    coupling : float
        Uniform ``J`` applied to every edge.

    Returns
    -------
    PottsGraph
        Nodes indexed by the row-major unraveling of ``shape``, carrying the
        shape and the boundary.

    Raises
    ------
    ValueError
        If ``shape`` is empty or any extent is below 2.
    """
    if not shape:
        msg = "shape must have at least one dimension"
        raise ValueError(msg)
    if any(extent < 2 for extent in shape):
        msg = f"every extent in shape must be >= 2, got {shape}"
        raise ValueError(msg)
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    def index(coordinate: tuple[int, ...]) -> int:
        return sum(c * s for c, s in zip(coordinate, strides, strict=True))

    periodic = boundary is BoundaryCondition.PERIODIC
    edges: list[tuple[int, int]] = []
    for coordinate in product(*(range(extent) for extent in shape)):
        for offset in offsets:
            target = tuple(c + step for c, step in zip(coordinate, offset, strict=True))
            if periodic:
                target = tuple(
                    c % extent for c, extent in zip(target, shape, strict=True)
                )
            elif any(c >= extent for c, extent in zip(target, shape, strict=True)):
                continue
            edges.append((index(coordinate), index(target)))

    n_nodes = 1
    for extent in shape:
        n_nodes *= extent

    return PottsGraph(
        n_nodes=n_nodes,
        edges=tuple(edges),
        coupling=(coupling,) * len(edges),
        shape=shape,
        boundary=boundary,
    )


def lattice_graph(
    shape: tuple[int, ...], boundary: BoundaryCondition, coupling: float
) -> PottsGraph:
    """Build an N-D lattice as a :class:`PottsGraph`, with a uniform coupling.

    A 1-D chain is ``lattice_graph((length,), boundary, coupling)``, not a
    separate type.

    Parameters
    ----------
    shape : tuple[int, ...]
        Extent along each of ``N`` dimensions, each ``>= 2``.
    boundary : BoundaryCondition
        ``OPEN``: no wraparound edges, so a boundary node has fewer
        neighbours. ``PERIODIC``: every dimension wraps.
    coupling : float
        Uniform ``J`` applied to every edge.

    Returns
    -------
    PottsGraph
        ``n_nodes = prod(shape)``, nodes indexed by the standard row-major
        (C-order) unraveling of ``shape``, matching
        ``numpy.ravel_multi_index``.

    Raises
    ------
    ValueError
        If ``shape`` is empty or any extent is below 2. The boundary is a
        :class:`BoundaryCondition`, so an unrecognized one is a type error
        rather than a run-time check.
    """
    offsets = tuple(
        tuple(1 if other == dim else 0 for other in range(len(shape)))
        for dim in range(len(shape))
    )
    return _lattice(shape, offsets, boundary, coupling)


def erdos_renyi_graph(
    n_nodes: int, probability: float, coupling: float, rng: np.random.Generator
) -> PottsGraph:
    """A `G(n, p)` random graph as a :class:`PottsGraph`, with a uniform coupling.

    Every unordered pair is an edge independently with probability
    ``probability``. A lattice is the *regular* extreme of a graph and this is
    the disordered one; between them they exercise the two structures a
    message-passing evaluator behaves differently on.

    **What this is for, and what it is not.** Sparse `G(n, p)` is locally
    tree-like: at ``p = c / n`` the expected number of triangles tends to a
    constant, ``c**3 / 6``, so a vanishing fraction of vertices lie on a short
    cycle. That is the regime belief propagation is asymptotically exact in,
    and the complement of the lattice, where every vertex sits on four
    4-cycles and the Bethe approximation is at its worst (issue #172).

    None of the famous `G(n, p)` results are usable here. The giant-component
    threshold at ``p = 1 / n`` and the connectivity threshold at
    ``ln(n) / n`` hold in the limit, and at the ``n <= 10`` cap
    ``infra/CLAUDE.md`` sets so enumeration stays affordable they mean
    nothing. Nothing in this repository tests them, and a test that did would
    be measuring a limit at a size where it does not hold.

    Parameters
    ----------
    n_nodes : int
        Number of nodes, ``>= 1``.
    probability : float
        Edge probability, in ``[0, 1]``.
    coupling : float
        Uniform ``J`` applied to every edge drawn.
    rng : np.random.Generator
        Passed in rather than seeded here, so a caller drawing an *ensemble*
        gets independent graphs rather than the same one repeatedly --- the
        mistake a `seed` parameter invites and which
        `sal.qa.rl_reward_surface` already made once.

    Returns
    -------
    PottsGraph
        With ``shape`` and ``boundary`` both ``None``: this is not a lattice
        and must not be mistaken for one by
        :meth:`PottsGraph.is_open_chain`.

    Raises
    ------
    ValueError
        If ``n_nodes`` is below 1 or ``probability`` is outside ``[0, 1]``.
    """
    if n_nodes < 1:
        msg = f"n_nodes must be at least 1, got {n_nodes}"
        raise ValueError(msg)
    if not 0.0 <= probability <= 1.0:
        msg = f"probability must lie in [0, 1], got {probability}"
        raise ValueError(msg)

    pairs = [
        (first, second)
        for first in range(n_nodes)
        for second in range(first + 1, n_nodes)
    ]
    drawn = rng.random(len(pairs)) < probability
    edges = tuple(pair for pair, keep in zip(pairs, drawn, strict=True) if keep)
    return PottsGraph(n_nodes=n_nodes, edges=edges, coupling=(coupling,) * len(edges))


def triangular_lattice_graph(
    shape: tuple[int, int], boundary: BoundaryCondition, coupling: float
) -> PottsGraph:
    """A 2-D triangular lattice: the square lattice plus one diagonal per cell.

    Rows and columns are joined as in :func:`lattice_graph`, and each unit
    cell gains the ``(row, column) -- (row + 1, column + 1)`` diagonal. Every
    cell is then split into two triangles, so the graph contains 3-cycles and
    is **not bipartite** --- which is the whole point of it.

    **Why an odd cycle matters.** A two-state antiferromagnet wants every edge
    to disagree. That is possible exactly when the graph is 2-colourable, so
    on a bipartite graph --- a chain, a square lattice --- the ground state is
    unfrustrated and the minimum energy is zero. A triangle cannot be
    2-coloured, so at least one edge of it must agree, and the ground state
    pays for it. That is geometric frustration, and it is the reason the
    triangular Ising antiferromagnet retains entropy at zero temperature
    (Wannier 1950).

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, columns)``, each ``>= 2``.
    boundary : BoundaryCondition
        ``OPEN`` drops the edges that would leave the grid; ``PERIODIC`` wraps
        both dimensions, giving every node degree 6.
    coupling : float
        Uniform ``J``. Negative is the antiferromagnet this exists for; the
        sign is the caller's, because the same graph read ferromagnetically is
        a perfectly ordinary unfrustrated instance and the contrast is worth
        being able to draw.

    Returns
    -------
    PottsGraph
        Carrying ``shape`` and ``boundary``, as :func:`lattice_graph` does.
        Note it is *not* an open chain even at ``shape = (2, 2)``, so
        :meth:`PottsGraph.is_open_chain` correctly refuses it.

    Raises
    ------
    ValueError
        If either extent is below 2.
    """
    return _lattice(shape, ((0, 1), (1, 0), (1, 1)), boundary, coupling)
