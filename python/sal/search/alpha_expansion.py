"""Alpha expansion: approximate MAP on a Potts MRF, with a proved bound.

`k`-state MAP is NP-hard, so :mod:`sal.search.maxflow`'s exact minimum cut
stops at two states. Alpha expansion recovers the general case as a sequence
of binary cuts: for each label ``alpha`` in turn, every site is offered the
choice of keeping its current label or switching to ``alpha``, and *that*
binary problem is submodular for a Potts pairwise term, so the exact solver
handles it unchanged.

**The guarantee is the point.** The Potts pairwise term ``V(a, b) = J [a != b]``
is a metric for ``J >= 0``, and for a metric the algorithm's local minimum is
within ``2 * c_max / c_min`` of the *global* one --- exactly **2** for a
uniform coupling (Boykov, Veksler & Zabih 2001). No other method here states a
bound: belief propagation (issue #172) reports a measured deviation, the
samplers (#174) a distribution, and enumeration stops at nine sites. A bound
holds at every size, so a result can be checked where the algorithm runs.

It is a local minimum with respect to moves that change *arbitrarily many*
sites at once, which is what makes it beat single-site descent: no sequence of
single flips crosses a barrier that one expansion crosses in a step.

**Two moves, one body each.** The expansion and the alpha-beta swap differ in
the label set a cycle offers and in the network one move builds; the widening,
the solver pick, the accept, the cycle and the two refusals are the same text,
so they are written once and parameterised by a :class:`_Move` (issue #858).
Single-site descent, the baseline the expansion has to beat, is
:mod:`sal.search.icm` (issue #1055); it returns this module's
:class:`Labelling`.

See Boykov, Veksler & Zabih (2001); Kolmogorov & Zabih (2004) for which
energies a cut can represent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, NamedTuple

import numpy as np

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.opt.termination import Termination
from sal.search.maxflow import (
    FlowNetwork,
    check_non_negative_couplings,
    max_flow,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
    energy,
    log_weight_of,
    site_field,
    states_of,
)

# The bound is `2 * c_max / c_min` for a metric pairwise term; with a uniform
# coupling the ratio is 1 and the factor is exactly 2.
UNIFORM_POTTS_BOUND = 2.0

DEFAULT_MAX_CYCLES = 50


@dataclass(frozen=True)
class Labelling:
    """A state per node, and the energy it scores.

    One type for the three moves: :func:`expand`, :func:`swap` and
    :func:`~sal.search.icm.iterated_conditional_modes` report the same two quantities with
    the same meaning, so a caller that exchanges one for another reads the
    same fields (issue #865).

    Parameters
    ----------
    labelling : np.ndarray
        One state per node.
    energy : float
        Its energy under :func:`~sal.sim.potts.energy`.
    sweeps : int
        Sweeps the descent ran, the clean one that stopped it included, so a
        caller charges what was spent without rerunning one sweep at a time
        (issue #1059). Zero for a single :func:`expand` or :func:`swap`, which
        runs no sweep.
    termination : Termination | None
        For a descent: converged where the labelling returned is one a sweep
        leaves unchanged --- a local minimum at the floor --- and the budget
        otherwise, after ``sweeps``. ``None`` for a single move.
    """

    labelling: np.ndarray
    energy: float
    sweeps: int = 0
    termination: Termination = dataclass_field(kw_only=True)

    def __iter__(self) -> Iterator[Any]:
        """``(labelling, energy, sweeps, termination)``: the declared order (#865).

        ``Any`` and not a union: an unpacking gives every name the element
        type, so a union would mistype each of them.
        """
        yield from (self.labelling, self.energy, self.sweeps, self.termination)


@dataclass(frozen=True, kw_only=True)
class BoundedLabelling(Labelling):
    """A labelling, and a lower bound on the minimum it is measured against (issue #1081).

    What the bounded solvers return: TRW-S and the tightened dual decomposition
    did the same job with two types, different field orders and an optional
    termination. ``energy`` is an **upper** bound on the minimum --- some
    labelling attains it --- and ``bound`` a **lower** one, both in
    :func:`~sal.sim.potts.energy`'s sign.

    Parameters
    ----------
    bound : float
        A lower bound on ``min_x E(x)``, valid at every iteration.
    """

    bound: float

    @property
    def gap(self) -> float:
        """``energy - bound``: what is not established. Zero certifies the labelling optimal."""
        return self.energy - self.bound

    @property
    def optimal(self) -> bool:
        """Whether the gap has closed to floating-point noise.

        A real certificate: nothing can beat the bound, and this labelling
        attains it.
        """
        return bool(self.gap <= 1e-9 * max(1.0, abs(self.bound)))


@dataclass(frozen=True)
class ExpansionResult:
    """A labelling, and what reaching it cost.

    Parameters
    ----------
    labelling : np.ndarray
        One state per node.
    energy : float
        Its energy under :func:`energy`.
    cycles : int
        Complete sweeps over the label set. The loop stops on the first sweep
        that lowers nothing, so this is one more than the number that helped.
    moves : int
        Expansions that strictly lowered the energy. Zero means the starting
        labelling was already expansion-optimal, which is information rather
        than a failure.
    termination : Termination | None
        Converged after ``cycles`` where a cycle lowered nothing. Where
        ``max_cycles`` ran out first it is the budget after ``max_cycles``,
        and a warning says so (issue #1059): the default cap is a defect
        guard, since monotonicity over a finite state space bounds the
        cycles, but a caller's cap is a budget (`search.ground_state`
        derives one from site visits), and a budget returns what it holds.
    """

    labelling: np.ndarray
    energy: float
    cycles: int
    moves: int
    termination: Termination = dataclass_field(kw_only=True)


class _Arcs(NamedTuple):
    """A move's network as arrays, one row per edge, in ``add_edge`` order (issue #935).

    :meth:`~sal.search.maxflow.FlowNetwork.from_arcs` builds
    the Python solver's list store from these, the oracle route. The Rust
    route refills a :class:`LatticeCut` instead and is pinned to this
    network's minimal minimum cut.
    """

    n_nodes: int
    tail: np.ndarray
    head: np.ndarray
    capacity: np.ndarray
    reverse: np.ndarray

    def network(self) -> FlowNetwork:
        """The same network as a :class:`FlowNetwork`, for the Python solver."""
        return FlowNetwork.from_arcs(
            self.n_nodes, self.tail, self.head, self.capacity, self.reverse
        )


class _CutMove(NamedTuple):
    """One binary move, built: its cut and where the cut lands.

    Parameters
    ----------
    source_side : Callable[[], np.ndarray]
        Cuts the move's network and returns the source side of its minimal
        minimum cut: the Python solver on the arcs :func:`_expansion_arcs`
        or :func:`_swap_arcs` lays out, or a :class:`LatticeCut` refilled
        in place (issue #935).
    place : Callable[[np.ndarray], np.ndarray]
        The source-side mask to the labelling it proposes.
    """

    source_side: Callable[[], np.ndarray]
    place: Callable[[np.ndarray], np.ndarray]


def _python_source_side(arcs: _Arcs, source: int, sink: int) -> np.ndarray:
    """The Python solver's source side on ``arcs``: the oracle route."""
    return max_flow(arcs.network(), source, sink).source_side


def _check_cut_backend(backend: Backend, move: str) -> None:
    """Refuse a backend neither move has a minimum cut for."""
    refuse_backend(f"{move}'s minimum cut", backend, (Backend.PYTHON, Backend.RUST))


def _terminal_capacities(
    source_side: np.ndarray, sink_side: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The two terminal arcs of each site, shifted by the cheaper branch.

    A cut pays one of the two whatever it does, so subtracting the smaller
    from both leaves every cut's capacity lower by the same constant and the
    minimizing cut where it was. Subtracting it keeps the capacities
    non-negative, which the flow requires.

    Parameters
    ----------
    source_side : np.ndarray
        Each site's cost of landing on the source side.
    sink_side : np.ndarray
        Each site's cost of landing on the sink side.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The capacity of ``source -> site``, which is cut when the site lands
        on the sink side, and of ``site -> sink``, which is cut when it lands
        on the source side.
    """
    offset = np.minimum(source_side, sink_side)
    return sink_side - offset, source_side - offset


class _Carried(NamedTuple):
    """What a cycle carries from one move to the next (issue #935).

    Parameters
    ----------
    cut : LatticeCut | None
        The Rust route's network, laid out once for the run; ``None`` on the
        Python route.
    energy : float
        The energy of the labelling the move starts from, as
        :func:`~sal.sim.potts.energy` scored it on the last
        move, so the next does not score it again.
    """

    cut: oxisal.LatticeCut | None
    energy: float


#: One binary move by one minimum cut: a single iteration, run to completion.
_ONE_MOVE = Termination.after(1, converged=True)


def _lowest_by_cut(
    graph: PottsGraph,
    field: np.ndarray,
    labelling: np.ndarray,
    build: Callable[[np.ndarray], _CutMove | None],
    held: float | None,
) -> Labelling:
    """A binary move by one minimum cut, taken only where it lowers the energy.

    The body :func:`expand` and :func:`swap` share (issue #858): widen the
    field, build the move's network, cut it with the chosen solver, and
    accept the proposal only against the energy. They differ in the network,
    which is ``build``'s, and in nothing else. ``build`` returning ``None``
    is a move with no site to make it on, which is the labelling unchanged.
    ``held`` is the input labelling's energy where the caller has it.
    """
    values = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    built = build(values)
    current = energy(graph, values, labelling) if held is None else held
    if built is None:
        return Labelling(labelling, current, termination=_ONE_MOVE)

    proposed = built.place(built.source_side())

    # Only the changed sites' terms move, so the proposal is scored by their
    # difference: a full `energy` was 0.18 s of a 0.92 s swap at 142^2
    # (issue #997). The cycle re-scores its result in full once.
    candidate = current + _energy_change(graph, values, labelling, proposed)
    if candidate < current:
        return Labelling(proposed, candidate, termination=_ONE_MOVE)
    return Labelling(labelling, current, termination=_ONE_MOVE)


def _energy_change(
    graph: PottsGraph,
    values: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    """``energy(after) - energy(before)`` from the sites that differ and their edges."""
    changed = before != after
    sites = np.flatnonzero(changed)
    if sites.size == 0:
        return 0.0
    field = values[sites, after[sites]].sum() - values[sites, before[sites]].sum()
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    touched = np.flatnonzero(changed[first] | changed[second])
    a, b = first[touched], second[touched]
    coupling = graph.edge_coupling[touched]
    agree = (after[a] == after[b]).astype(float) - (before[a] == before[b])
    return float(-field - coupling @ agree)


@dataclass(frozen=True)
class _Move:
    """A move set the cycle iterates, and what the refusals call it.

    Parameters
    ----------
    name : str
        The subject of this move's refusals, e.g. ``"alpha expansion"``.
    reason : str
        What non-negative couplings buy this move, passed to
        :func:`~sal.search.maxflow.check_non_negative_couplings`.
    label_sets : Callable[[int], Iterator[tuple[int, ...]]]
        The label sets one cycle offers, in the order it offers them: one
        label per move for the expansion, an unordered pair for the swap.
    apply : Callable[..., Labelling]
        The move itself, as :func:`expand` and :func:`swap` implement it.
    """

    name: str
    reason: str
    label_sets: Callable[[int], Iterator[tuple[int, ...]]]
    apply: Callable[
        [
            PottsGraph,
            np.ndarray,
            np.ndarray,
            tuple[int, ...],
            Backend,
            _Carried | None,
        ],
        Labelling,
    ]


def _cycle_to_a_local_minimum(
    graph: PottsGraph,
    field: np.ndarray,
    n_states: int,
    move: _Move,
    *,
    start: np.ndarray | None,
    max_cycles: int,
    backend: Backend,
) -> ExpansionResult:
    """Cycle over ``move``'s label sets until a full sweep lowers nothing.

    The body :func:`alpha_expansion` and :func:`alpha_beta_swap` share
    (issue #858). The loop, the accept and the refusal are identical
    between them; what differs is the label set a cycle iterates and the move
    it applies to each, which is what ``move`` carries. Monotonicity over a
    finite state space makes the default cap unreachable; a cap the caller
    derived from a budget can be reached, and then the labelling held is
    returned with a warning and a termination recording the cap (issue #1059).
    """
    check_non_negative_couplings(graph, move.reason)

    values = site_field(
        np.asarray(field, dtype=float), graph.n_nodes, n_states=n_states
    )
    labelling = (
        values.argmax(axis=1).astype(np.int64)
        if start is None
        else check_labelling(start, graph.n_nodes, n_states)
    )
    current = held = energy(graph, values, labelling)
    # One network for every move of the run: the lattice's arcs are laid out
    # once and each cut refills their capacities (issue #935).
    cut = _lattice_cut(graph) if backend is Backend.RUST else None

    moves = 0
    for cycle in range(1, max_cycles + 1):
        improved = False
        for labels in move.label_sets(n_states):
            moved = move.apply(
                graph, values, labelling, labels, backend, _Carried(cut, held)
            )
            labelling, held = moved.labelling, moved.energy
            if moved.energy < current - 1e-12:
                current = moved.energy
                improved = True
                moves += 1
        if not improved:
            return ExpansionResult(
                labelling=labelling,
                # In full: the moves carried it by differences (issue #997).
                energy=energy(graph, values, labelling),
                cycles=cycle,
                moves=moves,
                termination=Termination.after(cycle, converged=True),
            )

    # The cap is a termination, not a warning (issue #1089): the result says
    # it ran to `max_cycles`, which is all a caller needs to decide.
    return ExpansionResult(
        labelling=labelling,
        energy=energy(graph, values, labelling),
        cycles=max_cycles,
        moves=moves,
        termination=Termination.after(max_cycles, converged=False),
    )


def _expansion_network(
    graph: PottsGraph, values: np.ndarray, labelling: np.ndarray, alpha: int
) -> FlowNetwork:
    """:func:`_expansion_arcs` as a :class:`FlowNetwork`, the Python solver's form."""
    return _expansion_arcs(graph, values, labelling, alpha).network()


def _expansion_arcs(
    graph: PottsGraph, values: np.ndarray, labelling: np.ndarray, alpha: int
) -> _Arcs:
    """The expansion network of :func:`expand`, built without a call per arc.

    Arc for arc and in the same order as the ``add_edge`` loop it replaces,
    which `tests/regression/search/test_alpha_expansion.py` asserts against a
    transcription of that loop: the order is the contract, since a minimum cut
    need not be unique and the two solvers are pinned to one labelling.

    Issue #598 measured the loop at 76,928 calls and 53.0 per cent of a 32x32
    run at four labels, against the Rust cut kernel's 12.1 --- the network was
    built in Python and solved in Rust.

    Parameters
    ----------
    graph : PottsGraph
        The graph.
    values : np.ndarray
        Per-site field, shape ``(n_nodes, n_states)``, as
        :func:`sal.sim.potts.site_field` widens it.
    labelling : np.ndarray
        The labelling to expand, one label per node.
    alpha : int
        The label every node may move to.

    Returns
    -------
    FlowNetwork
        Spanning ``n_nodes + 2`` nodes plus one auxiliary per disagreeing
        edge, with the source at ``n_nodes`` and the sink at ``n_nodes + 1``.
    """
    n_nodes = graph.n_nodes
    source, sink = n_nodes, n_nodes + 1
    first, second, coupling = graph.endpoints
    labels = np.asarray(labelling)
    disagree = labels[first] != labels[second]
    n_auxiliary = int(np.count_nonzero(disagree))

    # The keep branch of a node already labelled alpha is unaffordable rather
    # than absent, so the node stays in the network (see :func:`expand`).
    switch = -values[:, alpha].astype(np.float64)
    keep = np.where(
        labels == alpha,
        _infinite_capacity(graph, values),
        -values[np.arange(n_nodes), labels].astype(np.float64),
    )
    from_source, to_sink = _terminal_capacities(keep, switch)

    node_tail = np.empty(2 * n_nodes, dtype=np.int64)
    node_head = np.empty(2 * n_nodes, dtype=np.int64)
    node_capacity = np.empty(2 * n_nodes, dtype=np.float64)
    nodes = np.arange(n_nodes, dtype=np.int64)
    node_tail[0::2], node_tail[1::2] = source, nodes
    node_head[0::2], node_head[1::2] = nodes, sink
    node_capacity[0::2], node_capacity[1::2] = from_source, to_sink

    # One arc per agreeing edge and three per disagreeing one, laid out in
    # edge order so the auxiliaries are numbered as the loop numbered them.
    width = np.where(disagree, 3, 1)
    start = np.concatenate(([0], np.cumsum(width)[:-1]))
    edge_tail = np.empty(int(width.sum()), dtype=np.int64)
    edge_head = np.empty_like(edge_tail)
    edge_capacity = np.zeros(edge_tail.size, dtype=np.float64)
    edge_reverse = np.zeros(edge_tail.size, dtype=np.float64)

    first_differs = np.where(labels[first] != alpha, coupling, 0.0)
    second_differs = np.where(labels[second] != alpha, coupling, 0.0)

    agreeing = start[~disagree]
    edge_tail[agreeing], edge_head[agreeing] = first[~disagree], second[~disagree]
    edge_capacity[agreeing] = edge_reverse[agreeing] = first_differs[~disagree]

    split = np.flatnonzero(disagree)
    at = start[split]
    auxiliary = np.arange(n_auxiliary, dtype=np.int64) + n_nodes + 2
    edge_tail[at], edge_head[at] = first[split], auxiliary
    edge_capacity[at] = edge_reverse[at] = first_differs[split]
    edge_tail[at + 1], edge_head[at + 1] = second[split], auxiliary
    edge_capacity[at + 1] = edge_reverse[at + 1] = second_differs[split]
    edge_tail[at + 2], edge_head[at + 2] = auxiliary, sink
    edge_capacity[at + 2] = coupling[split]

    return _Arcs(
        n_nodes + 2 + n_auxiliary,
        np.concatenate((node_tail, edge_tail)),
        np.concatenate((node_head, edge_head)),
        np.concatenate((node_capacity, edge_capacity)),
        np.concatenate((np.zeros(2 * n_nodes), edge_reverse)),
    )


def expand(
    graph: PottsGraph,
    field: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    *,
    backend: Backend = Backend.RUST,
) -> Labelling:
    """The optimal ``alpha``-expansion of ``labelling``, by one minimum cut.

    Every node chooses between keeping its current label and taking ``alpha``.
    The orientation is the half of this that is easy to get backwards, so it
    is stated rather than implied: **the source side keeps, the sink side
    takes ``alpha``**. An edge from the source to a node is therefore cut
    exactly when that node switches, and carries the data cost of switching.

    The pairwise term needs an auxiliary node wherever the two endpoints
    currently disagree, because three distinct costs have to be representable
    on one edge --- ``V(f_p, alpha)``, ``V(alpha, f_q)`` and ``V(f_p, f_q)``
    --- and a single arc can express only two. With ``V(a, b) = J [a != b]``:

    * endpoints agreeing: one arc of capacity ``V(f_p, alpha)``, which is
      ``J`` unless they are already ``alpha``, in which case it is zero;
    * endpoints disagreeing: an auxiliary ``a`` with ``p--a`` at
      ``V(f_p, alpha)``, ``q--a`` at ``V(f_q, alpha)``, and ``a -> sink`` at
      ``V(f_p, f_q) = J``.

    Getting those capacities wrong does not break loudly. It produces a
    labelling that is merely *worse*, which is indistinguishable from the
    algorithm working on a problem where it does badly --- which is why the
    `k = 2` reduction to :mod:`sal.search.maxflow` is the test that matters.

    A node already labelled ``alpha`` cannot move away, since an expansion
    moves labels *to* ``alpha`` only. It is pinned by making its keep branch
    unaffordable rather than by dropping it, so the edge terms around it stay
    in the same network.

    ``backend`` chooses the network and its solver.
    :data:`~sal.backend.Backend.PYTHON` builds the network
    above and cuts it with the Python Dinic: the oracle.
    :data:`~sal.backend.Backend.RUST`, the **default** since
    #935, refills a :class:`~sal.oxisal.LatticeCut`
    laid out once over the lattice, with no auxiliary node: Kolmogorov &
    Zabih's (2004) arc for the pairwise term, so the layout does not change
    with the labelling. Within :func:`alpha_expansion` each label's cut also
    starts from the flow its last cut ended on. A minimum cut need not be
    unique, but both routes read the *minimal* one --- the set reachable from
    the source in the residual graph, which every maximum flow of every
    network encoding the move's energy shares --- at one saturation floor
    (:data:`~sal.search.maxflow.SATURATED`), so rounding in
    their different sums does not move a site across the cut: 120 of 120
    random cut moves on decimal-valued fields agree bitwise
    (`test_the_cut_moves_agree_across_solvers_on_tied_fields`).

    Returns
    -------
    Labelling
        The expanded labelling and its energy. When no expansion helps, the
        input is returned unchanged.

    Raises
    ------
    ValueError
        If ``backend`` names an implementation this function does not have,
        or ``labelling`` is not one integer state in range per node
        (:func:`~sal.sim.potts.check_labelling`).
    """
    checked = check_labelling(
        labelling, graph.n_nodes, int(np.shape(field)[-1]), name="labelling"
    )
    return _expand(graph, field, checked, alpha, backend, None)


def _expand(
    graph: PottsGraph,
    field: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    backend: Backend,
    carried: _Carried | None,
) -> Labelling:
    """:func:`expand` with what a cycle carries across its moves."""
    _check_cut_backend(backend, "alpha expansion")

    def build(values: np.ndarray) -> _CutMove | None:
        # The sink side switched, so it takes alpha and the rest is held.
        def place(source_side: np.ndarray) -> np.ndarray:
            return np.where(~source_side[: graph.n_nodes], alpha, labelling)

        if backend is Backend.RUST:
            workspace = (
                _lattice_cut(graph)
                if carried is None or carried.cut is None
                else carried.cut
            )

            def solved() -> np.ndarray:
                return np.asarray(
                    workspace.expansion_source_side(
                        np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
                        values.shape[1],
                        np.ascontiguousarray(labelling, dtype=np.int64),
                        alpha,
                        _infinite_capacity(graph, values),
                    ),
                    dtype=bool,
                )

            return _CutMove(solved, place)
        arcs = _expansion_arcs(graph, values, labelling, alpha)
        return _CutMove(
            lambda: _python_source_side(arcs, graph.n_nodes, graph.n_nodes + 1),
            place,
        )

    return _lowest_by_cut(
        graph,
        field,
        labelling,
        build,
        None if carried is None else carried.energy,
    )


def _expansion_label_sets(n_states: int) -> Iterator[tuple[int, ...]]:
    """One label per move: every site is offered ``alpha``, for each label in turn."""
    return ((alpha,) for alpha in range(n_states))


def _apply_expansion(
    graph: PottsGraph,
    values: np.ndarray,
    labelling: np.ndarray,
    labels: tuple[int, ...],
    backend: Backend,
    carried: _Carried | None,
) -> Labelling:
    """:func:`expand` in the shape :func:`_cycle_to_a_local_minimum` calls."""
    return _expand(graph, values, labelling, labels[0], backend, carried)


EXPANSION = _Move(
    name="alpha expansion",
    reason=(
        "the Potts pairwise term is a metric only then, and the factor-2 bound "
        "rests on it"
    ),
    label_sets=_expansion_label_sets,
    apply=_apply_expansion,
)
"""The expansion move set: one cut per label, and the factor-2 bound."""


def alpha_expansion(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    *,
    n_states: int | None = None,
    start: np.ndarray | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    backend: Backend = Backend.RUST,
) -> ExpansionResult:
    """Cycle over labels until a full sweep lowers nothing.

    Two invariants make this checkable without an oracle, and both are
    asserted by the tests: the energy is **monotonically non-increasing**
    across every move, which a sign error in :func:`expand` breaks
    immediately; and the loop **terminates**, which follows from monotonicity
    over a finite state space.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative --- the metric condition the
        bound rests on.
    field : SiteField | np.ndarray
        ``(n_states,)`` or ``(n_nodes, n_states)``.
    n_states : int | None
        Label count, read from the field's state axis; given, it is checked
        against it (issue #1091).
    start : np.ndarray | None
        Initial labelling; the per-node data optimum when omitted, which is
        the labelling ignoring every coupling.
    max_cycles : int
        Cycles to run at most. The default, :data:`DEFAULT_MAX_CYCLES`, is a
        defect guard: monotonicity makes a correct run settle inside it. A
        caller's cap is a budget, and reaching it returns the labelling held.
    backend : Backend
        Which network and minimum-cut solver each :func:`expand` runs; see
        :func:`expand` for the two and why the Rust one is the default.

    Raises
    ------
    ValueError
        If a coupling is negative, or ``start`` is not one integer state in
        range per node (:func:`~sal.sim.potts.check_labelling`).

    Warns
    -----
    UserWarning
        Where ``max_cycles`` runs out before a cycle lowers nothing; the
        result then carries ``cycles = max_cycles`` and a termination
        recording the budget (issue #1059).
    """
    field = log_weight_of(field)
    return _cycle_to_a_local_minimum(
        graph,
        field,
        states_of(field, graph.n_nodes, n_states),
        EXPANSION,
        start=start,
        max_cycles=max_cycles,
        backend=backend,
    )


def _lattice_cut(graph: PottsGraph) -> oxisal.LatticeCut:
    """The Rust cut's network over ``graph``'s edges, laid out once (issue #935)."""
    first, second, coupling = graph.endpoints
    return oxisal.LatticeCut(
        graph.n_nodes,
        np.ascontiguousarray(first, dtype=np.int64),
        np.ascontiguousarray(second, dtype=np.int64),
        np.ascontiguousarray(coupling, dtype=np.float64),
    )


def _infinite_capacity(graph: PottsGraph, values: np.ndarray) -> float:
    """A capacity no cut would ever pay, scaled to this problem.

    ``float('inf')`` would work arithmetically and destroy the flow
    bookkeeping, since subtracting it leaves a residual of ``nan``. This is
    larger than every alternative cut by construction and stays finite.
    """
    return 1.0 + float(np.abs(values).sum()) + float(graph.edge_coupling.sum())


def _swap_arcs(
    graph: PottsGraph,
    values: np.ndarray,
    moving: np.ndarray,
    alpha: int,
    beta: int,
) -> _Arcs:
    """The network of :func:`swap` over the ``moving`` sites, built without a call per arc (issue #935).

    Arc for arc and in the order of the ``add_edge`` loop it replaces, which
    `tests/regression/search/test_alpha_expansion.py` asserts against a
    transcription of that loop. At 80,656 sites and ten labels that loop was
    18.8 s of a 20.8 s swap under ``cProfile``, the cut itself 0.7 s.

    The data term of a moving site is its own field and nothing else. A
    *held* neighbour carries neither alpha nor beta --- those are exactly the
    labels that move --- so it agrees with the moving site under neither
    choice and contributes the same constant to both. That is why this move
    needs no auxiliary node and why the expansion does: there, a held
    neighbour can already be alpha.

    Returns
    -------
    _Arcs
        Spanning the moving sites plus a source at ``moving.size`` and a sink
        after it.
    """
    source, sink = moving.size, moving.size + 1
    # Cut source -> index when the site lands on the sink side, taking beta,
    # so that arc carries the cost of beta.
    from_source, to_sink = _terminal_capacities(
        -values[moving, alpha].astype(float), -values[moving, beta].astype(float)
    )
    # Each site's source and sink arc in turn, then every edge with both ends
    # moving, in edge order, at its coupling both ways.
    indices = np.arange(moving.size, dtype=np.int64)
    node_tail = np.empty(2 * moving.size, dtype=np.int64)
    node_head = np.empty(2 * moving.size, dtype=np.int64)
    node_capacity = np.empty(2 * moving.size, dtype=np.float64)
    node_tail[0::2], node_tail[1::2] = source, indices
    node_head[0::2], node_head[1::2] = indices, sink
    node_capacity[0::2], node_capacity[1::2] = from_source, to_sink
    position = np.full(graph.n_nodes, -1, dtype=np.int64)
    position[moving] = indices
    first, second, coupling = graph.endpoints
    inside = (position[first] >= 0) & (position[second] >= 0)
    edge_capacity = np.asarray(coupling[inside], dtype=np.float64)
    return _Arcs(
        moving.size + 2,
        np.concatenate((node_tail, position[first[inside]])),
        np.concatenate((node_head, position[second[inside]])),
        np.concatenate((node_capacity, edge_capacity)),
        np.concatenate((np.zeros(2 * moving.size), edge_capacity)),
    )


def swap(
    graph: PottsGraph,
    field: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    beta: int,
    *,
    backend: Backend = Backend.RUST,
) -> Labelling:
    """The optimal ``alpha``-``beta`` swap of ``labelling``, by one minimum cut.

    Only the sites currently labelled ``alpha`` or ``beta`` move, and each
    chooses between the two; every other site is held. The binary problem is
    therefore over that subset alone, and it needs **no auxiliary node**: both
    endpoints of an edge inside the subset end at ``alpha`` or ``beta``, so
    the Potts term takes two values and one arc carries it. That is the whole
    difference from :func:`expand` --- a smaller network, a cheaper cut, and
    **no bound**, since the factor-2 guarantee of Boykov, Veksler & Zabih is a
    property of the expansion move and not of this one.

    The source side takes ``alpha`` and the sink side ``beta``. A held
    neighbour carries neither label --- those are exactly the ones that move
    --- so it agrees with the moving site under neither choice and drops out
    of the data term entirely.

    Returns
    -------
    Labelling
        The swapped labelling and its energy; the input unchanged where no
        swap lowers it.

    Raises
    ------
    ValueError
        If ``backend`` names an implementation this function does not have,
        or ``alpha`` and ``beta`` are the same label, where the move is the
        identity and a caller asking for it has a bug rather than a no-op, or
        ``labelling`` is not one integer state in range per node
        (:func:`~sal.sim.potts.check_labelling`).
    """
    checked = check_labelling(
        labelling, graph.n_nodes, int(np.shape(field)[-1]), name="labelling"
    )
    return _swap(graph, field, checked, alpha, beta, backend, None)


def _swap(
    graph: PottsGraph,
    field: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    beta: int,
    backend: Backend,
    carried: _Carried | None,
) -> Labelling:
    """:func:`swap` with what a cycle carries across its moves."""
    _check_cut_backend(backend, "the alpha-beta swap")
    if alpha == beta:
        msg = f"a swap needs two distinct labels, got {alpha} twice"
        raise ValueError(msg)

    def build(values: np.ndarray) -> _CutMove | None:
        moving = np.flatnonzero((labelling == alpha) | (labelling == beta))
        if moving.size == 0:
            return None

        def place(source_side: np.ndarray) -> np.ndarray:
            proposed = labelling.copy()
            proposed[moving] = np.where(source_side[: moving.size], alpha, beta)
            return proposed

        if backend is Backend.RUST:
            # The kernel cuts the whole lattice with every held site at zero
            # capacity, so its side is read at the moving sites' own indices.
            workspace = (
                _lattice_cut(graph)
                if carried is None or carried.cut is None
                else carried.cut
            )

            def solved() -> np.ndarray:
                side = workspace.swap_source_side(
                    np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
                    values.shape[1],
                    np.ascontiguousarray(labelling, dtype=np.int64),
                    alpha,
                    beta,
                )
                return np.asarray(side, dtype=bool)[moving]

            return _CutMove(solved, place)
        arcs = _swap_arcs(graph, values, moving, alpha, beta)
        return _CutMove(
            lambda: _python_source_side(arcs, moving.size, moving.size + 1), place
        )

    return _lowest_by_cut(
        graph,
        field,
        labelling,
        build,
        None if carried is None else carried.energy,
    )


def _swap_label_sets(n_states: int) -> Iterator[tuple[int, ...]]:
    """Every unordered pair of labels, the first index outermost."""
    return (
        (alpha, beta)
        for alpha in range(n_states)
        for beta in range(alpha + 1, n_states)
    )


def _apply_swap(
    graph: PottsGraph,
    values: np.ndarray,
    labelling: np.ndarray,
    labels: tuple[int, ...],
    backend: Backend,
    carried: _Carried | None,
) -> Labelling:
    """:func:`swap` in the shape :func:`_cycle_to_a_local_minimum` calls."""
    return _swap(graph, values, labelling, labels[0], labels[1], backend, carried)


SWAP = _Move(
    name="the alpha-beta swap",
    reason="the swap's binary sub-problem is submodular only then",
    label_sets=_swap_label_sets,
    apply=_apply_swap,
)
"""The swap move set: one cut per label pair, no auxiliary node, and no bound."""


def alpha_beta_swap(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    *,
    n_states: int | None = None,
    start: np.ndarray | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    backend: Backend = Backend.RUST,
) -> ExpansionResult:
    """Cycle over every label pair until a full sweep lowers nothing.

    The same loop as :func:`alpha_expansion` over a different move set, so a
    difference between the two is a statement about the move and not about
    the model or the code path. It is the cheaper move --- one cut over the
    sites carrying two labels rather than over the whole lattice, and no
    auxiliary node --- and it carries **no bound**, which is the trade issue
    #551 measures. A cycle is ``n_states * (n_states - 1) / 2`` cuts against
    the expansion's ``n_states``, so "cheaper per move" is not "cheaper per
    cycle" and the comparison is run at equal budget rather than equal cycles.

    Raises
    ------
    ValueError
        If a coupling is negative, or ``start`` is not one integer state in
        range per node.

    Warns
    -----
    UserWarning
        Where ``max_cycles`` runs out first, as :func:`alpha_expansion` does.
    """
    field = log_weight_of(field)
    return _cycle_to_a_local_minimum(
        graph,
        field,
        states_of(field, graph.n_nodes, n_states),
        SWAP,
        start=start,
        max_cycles=max_cycles,
        backend=backend,
    )
