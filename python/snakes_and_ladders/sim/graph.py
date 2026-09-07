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
from itertools import product

import numpy as np


class BoundaryCondition(StrEnum):
    """How a lattice's outermost sites connect.

    A ``StrEnum`` rather than a bare string so ``mypy --strict`` rejects an
    unrecognized boundary at the call site rather than the constructor
    raising at run time, and so the yaml loader has one place to parse it.
    The same choice ``snakes_and_ladders.search.infer.MoveSet`` and
    ``snakes_and_ladders.search.rl.RewardModel`` make.
    """

    OPEN = "open"
    PERIODIC = "periodic"


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
        assumes -- :func:`snakes_and_ladders.sim.potts.simulate_potts` indexes the spin
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
        for edge in self.edges:
            first, second = edge
            if not (0 <= first < self.n_nodes and 0 <= second < self.n_nodes):
                msg = f"edge {edge} names a node outside [0, {self.n_nodes})"
                raise ValueError(msg)

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

    def is_open_chain(self) -> bool:
        """Whether this graph is a 1-D lattice with an open boundary.

        The one case with a cheap exact sampler (:mod:`snakes_and_ladders.sim.potts`'s
        backward-message recursion, the same one
        :func:`snakes_and_ladders.opt.potts.log_partition` sums via transfer matrix) --
        a periodic ring is a different, cyclic recursion, so it is excluded
        here rather than approximated by the open-chain code.
        """
        return (
            self.shape is not None
            and len(self.shape) == 1
            and self.boundary is BoundaryCondition.OPEN
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

    edges: list[tuple[int, int]] = []
    for coordinate in product(*(range(extent) for extent in shape)):
        node = index(coordinate)
        for dim, extent in enumerate(shape):
            if boundary is BoundaryCondition.PERIODIC:
                neighbor = list(coordinate)
                neighbor[dim] = (coordinate[dim] + 1) % extent
                edges.append((node, index(tuple(neighbor))))
            elif coordinate[dim] + 1 < extent:
                neighbor = list(coordinate)
                neighbor[dim] += 1
                edges.append((node, index(tuple(neighbor))))

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
        `snakes_and_ladders.qa.rl_reward_surface` already made once.

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
    rows, columns = shape
    if rows < 2 or columns < 2:
        msg = f"every extent in shape must be >= 2, got {shape}"
        raise ValueError(msg)

    def index(row: int, column: int) -> int:
        return row * columns + column

    periodic = boundary is BoundaryCondition.PERIODIC
    offsets = ((0, 1), (1, 0), (1, 1))
    edges: list[tuple[int, int]] = []
    for row, column in product(range(rows), range(columns)):
        for row_step, column_step in offsets:
            target_row, target_column = row + row_step, column + column_step
            if periodic:
                target_row, target_column = target_row % rows, target_column % columns
            elif target_row >= rows or target_column >= columns:
                continue
            edges.append((index(row, column), index(target_row, target_column)))

    return PottsGraph(
        n_nodes=rows * columns,
        edges=tuple(edges),
        coupling=(coupling,) * len(edges),
        shape=shape,
        boundary=boundary,
    )
