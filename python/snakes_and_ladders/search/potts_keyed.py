"""The cluster moves, as something a deterministic ``step`` can call.

Issue #706. :class:`~snakes_and_ladders.learn.keyed.KeyedMove` is declared in
``learn`` and implemented here, which is the direction the package layering
allows: ``search`` may import ``learn``, and
`tests/regression/learn/test_learn_environment.py` refuses the reverse. So a
Wolff step and a Swendsen-Wang pass reach
:class:`~snakes_and_ladders.learn.potts_nd.PottsNDEnvironment` without that
module knowing this one exists.

**These are the moves ``potts_mcmc`` already runs, not second copies of them.**
Above zero temperature each delegates to ``_wolff_sweep`` or
``_swendsen_wang_sweep``, which is why this module adds no kernel to keep in
step with an oracle --- the oracle *is* what runs. What it adds is two things
those functions did not have: the choice of root and colour moved out of the
sweep's generator and into the action (#706's Wolff action names both), and the
``T = 0`` limit.

**Zero temperature is written out rather than passed through**, because
``beta = 1 / 0`` is not a number the sweeps can carry: ``_recolour`` scores a
proposal as ``(beta * h)[new].sum() - (beta * h)[old].sum()``, which is
``inf - inf`` and so ``nan``, and every proposal would be rejected by an
arithmetic accident. The limit itself is well defined and is the sharpest thing
here to check: the bond probability ``1 - exp(-J / T)`` goes to one, so the
clusters become exactly the monochromatic connected components, and the accept
step keeps a recolouring only if it does not lower the score. Both are
independently computable, which is what
`tests/regression/learn/test_cluster_arms.py` pins them against.
"""

from __future__ import annotations

import math

import numpy as np

from snakes_and_ladders.search.potts_mcmc import (
    MoveKind,
    _find,
    _niedermayer_sweep,
    _swendsen_wang_sweep,
    _union,
    _wolff_sweep,
    niedermayer_threshold,
)
from snakes_and_ladders.sim.graph import PottsGraph


def monochrome_partition(
    labels: np.ndarray, offsets: np.ndarray, neighbours: np.ndarray
) -> np.ndarray:
    """Component index per site over the bonds between like-coloured sites.

    The ``T = 0`` cluster partition of both moves: every like-coloured bond is
    active with probability ``1 - exp(-J / T) -> 1``, so a cluster is a
    connected component of the like-coloured subgraph. Uses
    ``potts_mcmc``'s own union-find, so the partition is not a second reading
    of what a component is. Walked over the compressed rows the move built
    once rather than the graph's edge tuples, which `CLAUDE.md`'s layout rule
    asks and `infra/appraise_structures.py` reported (2026-09-17 review).

    Parameters
    ----------
    labels : np.ndarray
        The labelling, ``(n_nodes,)``.
    offsets, neighbours : np.ndarray
        The compressed adjacency, as
        :meth:`~snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`
        lays it out. Each bond appears twice and is joined once, from its
        lower end.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` of component roots, as :func:`_find` reports them.
    """
    n_nodes = int(offsets.shape[0]) - 1
    parent = np.arange(n_nodes)
    bounds, incident = offsets.tolist(), neighbours.tolist()
    for node in range(n_nodes):
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if neighbour > node and labels[neighbour] == labels[node]:
                _union(parent, node, neighbour)
    return np.array([_find(parent, node) for node in range(n_nodes)])


def _recolour_at_zero(
    labels: np.ndarray,
    members: np.ndarray,
    field: np.ndarray,
    proposed: int,
) -> None:
    """Keep a cluster's recolouring only if it does not lower the score.

    The ``beta -> inf`` limit of ``potts_mcmc._recolour``: the Metropolis
    acceptance ``exp(beta * difference)`` vanishes for a negative difference,
    so a decrease is refused outright and no uniform is consumed. A zero
    difference is accepted, as it is there --- a move that changes nothing
    about the score is not a move that failed.
    """
    current = int(labels[members[0]])
    if proposed == current:
        return
    difference = float(field[members, proposed].sum() - field[members, current].sum())
    if difference >= 0.0:
        labels[members] = proposed


class WolffMove:
    """One cluster grown from the action's root, recoloured to its label.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, whose couplings must be non-negative --- the bond
        probability ``1 - exp(-J / T)`` is not a probability below zero, which
        ``potts_mcmc`` refuses for the same reason.
    field : np.ndarray
        ``(n_nodes, n_states)``, in the score convention
        ``search.ground_state`` uses: larger is better, and the energy is its
        negation.

    Raises
    ------
    ValueError
        If the field's first axis is not the graph's site count.
    """

    def __init__(self, graph: PottsGraph, field: np.ndarray) -> None:
        field = np.asarray(field, dtype=np.float64)
        if field.shape[0] != graph.n_nodes:
            msg = (
                f"field has {field.shape[0]} rows for {graph.n_nodes} sites: "
                "a move and its environment score one problem or neither is "
                "measured (issue #706)"
            )
            raise ValueError(msg)
        self._n_nodes = graph.n_nodes
        self._field = field
        # Built once and held: the T = 0 partition and the T > 0 sweep both
        # read these rows, and `infra/appraise_structures.py` reads the store
        # from these assignments (recompute or store, decided as store).
        offsets, neighbours, couplings = graph.compressed_adjacency()
        self._offsets = offsets
        self._neighbours = neighbours
        self._couplings = couplings
        # `potts_mcmc._anneal`'s charge, to the integer division: a cluster
        # member's neighbours are read and the member is written, and the mean
        # degree is what a heat-bath sweep is charged per site.
        self._per_member = 1 + 2 * len(graph.edges) // graph.n_nodes

    @property
    def n_nodes(self) -> int:
        """Sites the move expects."""
        return self._n_nodes

    @property
    def n_states(self) -> int:
        """Labels the move expects."""
        return int(self._field.shape[1])

    @property
    def parametric(self) -> bool:
        """A Wolff action names its root and the colour to recolour to."""
        return True

    def propose(
        self,
        state: np.ndarray,
        *,
        temperature: float,
        site: int,
        label: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Grow the cluster at ``site``, recolour it to ``label``, and charge it."""
        labels = np.ascontiguousarray(state, dtype=np.int64).copy()
        if temperature == 0.0:
            partition = monochrome_partition(labels, self._offsets, self._neighbours)
            members = np.flatnonzero(partition == partition[site])
            _recolour_at_zero(labels, members, self._field, label)
            size = int(members.size)
        else:
            size = _wolff_sweep(
                labels,
                self._field,
                self._offsets,
                self._neighbours,
                self._couplings,
                rng,
                beta=1.0 / temperature,
                root=site,
                proposed=label,
            )
        return labels, size * self._per_member


class NiedermayerMove:
    """One cluster grown under Niedermayer's bond rule, two colours transposed on it.

    Wolff's arm generalized (issue #756):
    :func:`~snakes_and_ladders.search.potts_mcmc._niedermayer_sweep` activates a
    bond on its energy relative to a threshold ``E_0`` rather than on its
    endpoints agreeing, so the arm runs on a coupling of either sign, and at
    :func:`~snakes_and_ladders.search.potts_mcmc.niedermayer_threshold`'s value
    it *is* Wolff on a ferromagnet --- same bonds, same acceptance of one.

    The action names a root and a colour, as :class:`WolffMove`'s does. The
    colour is read as the one the root's own colour is **transposed** with
    rather than the one the cluster is recoloured to: above ``E_0 = 0`` a
    cluster is not monochromatic and a recolouring would move its interior
    bonds, which is the one thing the construction needs left alone.

    **Zero temperature is passed through rather than written out**, which is
    where this differs from :class:`WolffMove`. ``beta = inf`` is a number the
    sweep carries: a bond of positive energy margin is certain rather than
    drawn, and a step that lowers the score is refused without consuming a
    uniform, so the limit is the sweep's own rather than a second kernel
    beside it.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings of either sign, unlike :class:`WolffMove`'s.
    field : np.ndarray
        ``(n_nodes, n_states)``, as :class:`WolffMove` takes it.

    Raises
    ------
    ValueError
        If the field's first axis is not the graph's site count.
    """

    def __init__(self, graph: PottsGraph, field: np.ndarray) -> None:
        field = np.asarray(field, dtype=np.float64)
        if field.shape[0] != graph.n_nodes:
            msg = (
                f"field has {field.shape[0]} rows for {graph.n_nodes} sites: "
                "a move and its environment score one problem or neither is "
                "measured (issue #706)"
            )
            raise ValueError(msg)
        self._n_nodes = graph.n_nodes
        self._field = field
        offsets, neighbours, couplings = graph.compressed_adjacency()
        self._offsets = offsets
        self._neighbours = neighbours
        self._couplings = couplings
        self._threshold = niedermayer_threshold(couplings)
        self._per_member = 1 + 2 * len(graph.edges) // graph.n_nodes

    @property
    def n_nodes(self) -> int:
        """Sites the move expects."""
        return self._n_nodes

    @property
    def n_states(self) -> int:
        """Labels the move expects."""
        return int(self._field.shape[1])

    @property
    def parametric(self) -> bool:
        """A Niedermayer action names its root and the colour to transpose with."""
        return True

    @property
    def threshold(self) -> float:
        """``E_0``, the bond rule's threshold on this lattice."""
        return self._threshold

    def propose(
        self,
        state: np.ndarray,
        *,
        temperature: float,
        site: int,
        label: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Grow the cluster at ``site``, transpose its colour with ``label``, charge it."""
        labels = np.ascontiguousarray(state, dtype=np.int64).copy()
        size = _niedermayer_sweep(
            labels,
            self._field,
            self._offsets,
            self._neighbours,
            self._couplings,
            rng,
            beta=math.inf if temperature == 0.0 else 1.0 / temperature,
            threshold=self._threshold,
            root=site,
            partner=label,
        )
        return labels, size * self._per_member


class SwendsenWangMove:
    """Every bond drawn, every cluster recoloured, once per action.

    The action names a temperature and nothing else, so this move contributes
    one candidate per rung --- the whole reason #706 runs it first: six
    candidates a decision is the smallest environment in which "learn the
    schedule" is the entire question.

    Parameters
    ----------
    graph, field
        As :class:`WolffMove`.

    Raises
    ------
    ValueError
        As :class:`WolffMove`.
    """

    def __init__(self, graph: PottsGraph, field: np.ndarray) -> None:
        field = np.asarray(field, dtype=np.float64)
        if field.shape[0] != graph.n_nodes:
            msg = (
                f"field has {field.shape[0]} rows for {graph.n_nodes} sites: "
                "a move and its environment score one problem or neither is "
                "measured (issue #706)"
            )
            raise ValueError(msg)
        self._graph = graph
        self._field = field
        offsets, neighbours, _ = graph.compressed_adjacency()
        self._offsets = offsets
        self._neighbours = neighbours
        #: `search.ground_state.Rung.visits_per_sweep`: the bond pass reads
        #: both labels of every edge and writes every site.
        self._visits = graph.n_nodes + 2 * len(graph.edges)

    @property
    def n_nodes(self) -> int:
        """Sites the move expects."""
        return self._graph.n_nodes

    @property
    def n_states(self) -> int:
        """Labels the move expects."""
        return int(self._field.shape[1])

    @property
    def parametric(self) -> bool:
        """A Swendsen-Wang action names a temperature and nothing else."""
        return False

    def propose(
        self,
        state: np.ndarray,
        *,
        temperature: float,
        site: int,
        label: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Recolour every cluster once, and charge a sweep."""
        del site, label
        labels = np.ascontiguousarray(state, dtype=np.int64).copy()
        if temperature == 0.0:
            partition = monochrome_partition(labels, self._offsets, self._neighbours)
            for root in np.unique(partition):
                members = np.flatnonzero(partition == root)
                proposed = int(rng.integers(self.n_states))
                _recolour_at_zero(labels, members, self._field, proposed)
        else:
            _swendsen_wang_sweep(
                labels, self._graph, self._field, rng, None, 1.0 / temperature
            )
        return labels, self._visits


def cluster_moves(
    graph: PottsGraph, field: np.ndarray
) -> dict[MoveKind, WolffMove | SwendsenWangMove | NiedermayerMove]:
    """Every cluster move on one lattice, keyed as the environment expects them.

    One call rather than three, so an arm cannot be built with a Wolff move on
    one field and a Swendsen-Wang move on another.

    Houdayer's move is not here and cannot be: it acts on a *pair* of replicas
    and an arm's action carries one labelling to one labelling, so there is no
    action of this environment that names it.
    :func:`~snakes_and_ladders.search.potts_mcmc.sample_potts_pair` and
    :func:`~snakes_and_ladders.search.tempered.tempered_potts_pair` are where
    it is offered (issue #756).
    """
    return {
        MoveKind.WOLFF: WolffMove(graph, field),
        MoveKind.SWENDSEN_WANG: SwendsenWangMove(graph, field),
        MoveKind.NIEDERMAYER: NiedermayerMove(graph, field),
    }
