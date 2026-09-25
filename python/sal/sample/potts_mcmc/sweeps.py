"""The kernels: one sweep of each move set on a Potts lattice, the cluster constructions the sweeps share, and the compiled routes they dispatch to.

A kernel advances a state in place from a generator; none owns a chain, a
schedule or a record, which are
:mod:`~sal.sample.potts_mcmc.chains`'. The route between the
Python oracle and the Rust sweep is chosen here, per kernel, by the backend
the caller names. Imports :mod:`~sal.sample.potts_mcmc.moves`
alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from typing import Any, NamedTuple

import numpy as np

from sal.backend import Backend, refuse_backend
from sal.sample.accept import accept, accept_at, accept_drawn
from sal.sample.balanced import (
    draw_change,
    log_balanced_weights,
    log_metropolis_ratio,
    log_normalizer,
    log_ratios,
)
from sal.sample.potts_mcmc.moves import PottsMove
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    heat_bath_log_weights,
    local_fields,
    log_weight_of,
    owner_rows,
)

#: How far from a cumulative boundary a draw must land for the Rust sweep to
#: decide a site itself, in units of the last place per state. NumPy's ``exp``
#: and ``libm``'s differ by at most one such unit, the cumulative sum carries
#: that difference across at most ``n_states`` additions, and scaling the draw
#: by the last entry carries it once more: four units per state bounds it, and
#: this is four times that. The same width, on the same derivation, as
#: `gibbs._GUARD` (issues #561, #599); two consumers, so it is stated in each
#: rather than made a seam.
GUARD = 16.0


class Recolour(NamedTuple):
    """What one cluster recolouring proposed, and whether the field accepted it.

    Two booleans rather than one, because a proposal that drew the colour the
    cluster already has is not a rejection: counting it as one understates the
    accept rate by ``1 / q`` and would make the cluster moves look worse than
    they are at exactly the point issue #551 measures them.
    """

    proposed: bool
    accepted: bool


@dataclass
class ClusterCounter:
    """Cluster sizes and field acceptances, accumulated over a run.

    The instrumentation issue #551 exists to collect. Mutable and passed in
    rather than returned, because a sweep recolours a variable number of
    clusters and threading a growing tuple back through every call would cost
    more than the sweep.

    Attributes
    ----------
    sizes : list[int]
        One entry per cluster recoloured.
    proposals, accepts : int
        Colour changes proposed, and of those accepted. The accept rate is
        the ratio, and is undefined rather than 1.0 when nothing was proposed.
    spanning : int
        Clusters reaching from one edge of the lattice to the opposite one.
        Counted only where the graph declares a 2-D ``shape``; a graph without
        one has no sides to span and this stays zero.
    """

    sizes: list[int] = dataclass_field(default_factory=list)
    proposals: int = 0
    accepts: int = 0
    spanning: int = 0

    def record(
        self, members: np.ndarray, outcome: Recolour, graph: PottsGraph | None
    ) -> None:
        """Add one recoloured cluster."""
        self.sizes.append(int(members.shape[0]))
        self.proposals += int(outcome.proposed)
        self.accepts += int(outcome.accepted)
        self.spanning += int(_spans(members, graph))

    @property
    def accept_rate(self) -> float:
        """Accepted over proposed; ``nan`` where nothing was proposed."""
        return self.accepts / self.proposals if self.proposals else float("nan")

    @property
    def mean_size(self) -> float:
        """Sites per recoloured cluster; ``nan`` where none was built."""
        return float(np.mean(self.sizes)) if self.sizes else float("nan")

    @property
    def max_size(self) -> int:
        """The largest cluster recoloured; zero where none was built."""
        return max(self.sizes) if self.sizes else 0

    @property
    def spanning_fraction(self) -> float:
        """Clusters that spanned the lattice, as a fraction of those built."""
        return self.spanning / len(self.sizes) if self.sizes else float("nan")


def _spans(members: np.ndarray, graph: PottsGraph | None) -> bool:
    """Whether a cluster reaches both opposite sides of a 2-D lattice.

    The node index of a lattice built by
    :func:`sal.sim.graph.triangular_lattice_graph` is
    ``row * columns + column``, so the coordinates are a ``divmod``. A
    spanning cluster is what makes a cluster move a global move rather than a
    large local one, which is the distinction issue #551 reports against
    temperature.
    """
    if graph is None or graph.shape is None or len(graph.shape) != 2:
        return False
    rows, columns = graph.shape
    row, column = np.divmod(members, columns)
    return bool(
        (row.min() == 0 and row.max() == rows - 1)
        or (column.min() == 0 and column.max() == columns - 1)
    )


def sweep_at(
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    backend: Backend,
) -> Callable[[np.ndarray, np.random.Generator, float], None]:
    """One tempered heat-bath sweep, on the backend the caller named.

    Both closures consume exactly ``n_nodes`` uniforms per sweep from the
    generator they are handed, so switching backend changes which arithmetic
    evaluates the conditional and nothing about the stream. Tempering reaches
    the Rust kernel as ``beta`` itself, applied to the accumulated local field
    where the Python sweep applies it, so the two agree bitwise at every
    temperature rather than only at 1.0 (issue #571).

    **The two produce the same chain, not a chain of the same law**, which is
    why :data:`~sal.backend.Backend.RUST` is the default
    (issue #599). Every step of the conditional is the arithmetic NumPy
    performs, operation for operation, but ``exp`` is not: NumPy computes it
    by its own SIMD polynomial and the kernel by ``libm``, and one draw across
    a boundary that moved in the last place sends two chains apart. So the
    kernel decides a site only where the draw clears every cumulative boundary
    by :data:`GUARD` units of the last place per state, returns the position
    of the first site it declines, and :func:`site_update` --- the oracle's
    own update --- decides that one before the kernel resumes.

    The adjacency arrives as the compressed rows both backends read, built
    once by the caller: the Python sweep indexes them and the kernel takes
    them across the boundary without marshalling (issue #277).
    """
    if backend is Backend.PYTHON:

        def python_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            single_site_sweep(
                state, rows, offsets, neighbours, couplings, rng, beta=beta
            )

        return python_sweep
    if backend is Backend.RUST:
        from sal import oxisal

        # The field crosses as one row per site, which is the shape `rows`
        # already has: `sim.potts.site_field` widened it at the entry point,
        # so a shared field is rows that are all equal and there is one code
        # path rather than two (issue #551, #571). The adjacency is the
        # caller's, built once (issue #277); both arrays are made contiguous
        # here rather than per sweep.
        contiguous_field = np.ascontiguousarray(rows, dtype=np.float64)
        contiguous_couplings = np.ascontiguousarray(couplings, dtype=np.float64)
        # The lists `site_update` indexes on a hand-back, converted once per
        # run rather than per sweep: the kernel hands a site back so rarely
        # that a per-sweep conversion would cost more than the sweep.
        bounds = offsets.tolist()
        incident, weights = neighbours.tolist(), contiguous_couplings.tolist()

        def rust_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            n_nodes = state.shape[0]
            draws = np.ascontiguousarray(rng.random(n_nodes), dtype=np.float64)
            node = 0
            while node < n_nodes:
                # `beta` is passed rather than multiplied into the arguments.
                # The Python sweep scales the accumulated local field, so
                # scaling the parts instead computes `beta * h + sum (beta *
                # J)` against its `(h + sum J) * beta` -- equal in real
                # arithmetic, not bitwise, which cost agreement at every
                # temperature but 1.0 (issue #571). It also drops two
                # whole-array temporaries per sweep, so issue #651 re-opened it
                # under `CLAUDE.md`'s rule that bitwise may back off to a
                # declared tolerance.
                #
                # **Measured and refused.** The reassociation is not a
                # last-place difference: `h + sum J` cancels, so the
                # scaled-parts form's absolute error is set by the magnitudes
                # of the parts while the result is near zero. Over 50,000
                # sites the median is 1.00 unit of the last place and the
                # maximum is 201,145, with 2.9% past `GUARD` -- and those are
                # the cancelling sites, whose accumulated field is a median
                # 7.4e-02 against 2.8e+00 for the rest. A near-zero field is a
                # near-uniform conditional, so the error concentrates on the
                # sites whose decision it is likeliest to flip. Neither the
                # guard nor a relative tolerance reaches it.
                # `tests/regression/search/test_potts_sweep_reassociation.py`
                # holds those numbers.
                node = oxisal.single_site_sweeps(
                    state,
                    contiguous_field,
                    offsets,
                    neighbours,
                    contiguous_couplings,
                    draws,
                    1,
                    beta,
                    GUARD,
                    node,
                )
                if node < n_nodes:
                    site_update(
                        state,
                        rows,
                        incident,
                        weights,
                        bounds,
                        node,
                        float(draws[node]),
                        beta,
                    )
                    node += 1

        return rust_sweep
    refuse_backend("the heat-bath sweep", backend, (Backend.PYTHON, Backend.RUST))
    raise AssertionError(backend)  # pragma: no cover - both routes returned


def single_site_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    beta: float = 1.0,
) -> None:
    """One heat-bath sweep: every site redrawn from its exact conditional.

    The baseline the cluster algorithms are measured against. The conditional
    is :func:`sal.sim.potts.heat_bath_log_weights`, shared with
    the vectorized simulator; the loop is not, since that one runs many
    independent chains at once and this one steps a single chain in time
    (issue #277).

    ``rows`` is the field as one row per site, widened by
    :func:`sal.sim.potts.site_field` at the entry point. A
    shared field reaches here as rows that are all equal, so there is one
    code path rather than two, on `sim/potts.py`'s rule (issue #551).

    ``offsets``, ``neighbours`` and ``couplings`` are the graph's compressed
    rows: site ``i``'s neighbours are ``neighbours[offsets[i]:offsets[i + 1]]``
    and their couplings sit at the same positions.

    ``beta`` tempers the conditional in place, for :func:`anneal_potts`, whose
    temperature changes every sweep and would otherwise rebuild the adjacency
    each time. At 1.0 the multiplication is the identity bitwise, and is
    skipped.
    """
    draws = np.asarray(rng.random(state.shape[0]))
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    for node in range(state.shape[0]):
        site_update(
            state, rows, incident, weights, bounds, node, float(draws[node]), beta
        )


def site_update(
    state: np.ndarray,
    rows: np.ndarray,
    incident: Sequence[int],
    weights: Sequence[float],
    bounds: Sequence[int],
    node: int,
    draw: float,
    beta: float,
) -> None:
    """One site redrawn from its exact conditional, in place.

    The whole of the oracle's update, factored out so the Rust sweep's
    hand-back path decides its site by calling this rather than a copy of it
    (issue #599). ``incident``, ``weights`` and ``bounds`` are the compressed
    rows as the lists `sim.potts.heat_bath_log_weights` measured as cheaper to
    index than array rows.

    One uniform and a search, rather than ``rng.choice`` per site: this is the
    baseline the cluster algorithms are timed against, so its constant factor
    decides how large a lattice the comparison reaches.
    """
    local = heat_bath_log_weights(
        rows[node], state, incident, weights, bounds[node], bounds[node + 1], beta
    )
    local -= local.max()
    cumulative = np.cumsum(np.exp(local))
    state[node] = np.searchsorted(cumulative, draw * cumulative[-1])


@dataclass(frozen=True)
class Clusters:
    """A labelling grouped into its clusters, as one sort rather than a scan.

    Parameters
    ----------
    order : np.ndarray
        The nodes, sorted by label and by index within a label.
    bounds : np.ndarray
        Where each cluster starts in ``order``, with ``labels.size`` last, so
        cluster ``k`` is ``order[bounds[k]:bounds[k + 1]]``.
    """

    order: np.ndarray
    bounds: np.ndarray

    def __iter__(self) -> Iterator[Any]:
        """``(order, bounds)``: the order callers unpack.

        ``Any`` for :meth:`TemperedModel.__iter__`'s reason.
        """
        yield from (self.order, self.bounds)


def cluster_members(labels: np.ndarray) -> Clusters:
    """Group a labelling into its clusters: the members, and where each starts.

    ``labels[order[bounds[k]:bounds[k + 1]]]`` is the ``k``-th root in
    increasing order, and the slice of ``order`` is that cluster's members in
    increasing node order --- which is exactly what
    ``np.flatnonzero(labels == root)`` returns for ``root`` walked over
    ``np.unique(labels)``, since a stable sort keeps equal keys in index
    order. So this changes the grouping's cost and not one member of one
    cluster.

    The cost is the point. The form it replaces is ``O(n_clusters * n_nodes)``
    --- one full comparison of the labelling per cluster --- and the bond pass
    at the transition makes a cluster for every 1.7 sites, so the quadratic
    term *is* the pass: at 64x64 it was 10.5 ms of a 41.2 ms
    ``SwendsenWangMove.propose`` against 0.5 ms for this one sort (#754).
    Root ``CLAUDE.md``'s rule that an algorithmic cut outranks a mechanical
    one, taken before the port below rather than ported around.
    """
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
    )
    return Clusters(order=order, bounds=np.concatenate((starts, [labels.size])))


def taylor_log_ratios(
    rows: np.ndarray,
    state: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    owner: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Gibbs-with-gradients' first-order estimate of every single-flip change.

    Grathwohl et al. (2021) relax the state to the simplex --- the one-hot
    matrix `learn.relaxed.one_hot` writes --- and estimate
    ``log pi(s') - log pi(s)`` by ``grad(log pi)(x) . (x' - x)``, one gradient
    for the whole neighbourhood against one energy per neighbour.

    **On a Potts energy the estimate is exact.** The relaxed log weight
    ``sum_i h_i . x_i + sum_(ij) J_ij x_i . x_j`` is affine in each site's row
    --- a lattice has no self-coupling, so no term carries ``x_i`` twice ---
    and a single-flip change moves one row, so the first-order term is the
    whole difference. The gradient at a one-hot is then
    :func:`sal.sim.potts.local_fields`, which is what this
    computes: the tape returns the same numbers
    (:func:`autodiff_log_ratios`, pinned at ``1e-12``) for a tape's cost per
    proposal. Where the estimate is exact this equals
    :func:`~sal.sample.balanced.log_ratios`, and the two stay
    separate functions because that equality is a property of *this* energy
    and is pinned rather than assumed.

    Parameters
    ----------
    rows, state, neighbours, couplings, owner, beta
        As :func:`sal.sim.potts.local_fields`.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``.
    """
    return log_ratios(
        local_fields(rows, state, neighbours, couplings, owner, beta), state
    )


def autodiff_log_ratios(
    graph: PottsGraph, rows: np.ndarray, state: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """:func:`taylor_log_ratios` from the tape: the definition, not the route.

    The relaxed log weight is built in ``torch`` on the one-hot state and
    differentiated by ``torch.autograd.grad``, which is what Gibbs-with-
    gradients *is*. It is the oracle rather than the sampler's route: the
    closed form :func:`taylor_log_ratios` takes reproduces it to ``1e-12``
    (`tests/regression/search/test_potts_mcmc.py`) and costs one ``bincount``
    against a tape built and walked per proposal.

    ``torch`` is imported here rather than at module scope: it is the
    heaviest import in the package and no chain this module runs needs it.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, read for its edge list and per-edge couplings.
    rows : np.ndarray
        The field as one row per site, shape ``(n_nodes, n_states)``.
    state : np.ndarray
        The current configuration.
    beta : float
        Inverse temperature, scaling the whole log weight.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``.
    """
    import torch

    # `torch.tensor` rather than `from_numpy`: a graph hands out read-only
    # views of its arrays (#623), which `from_numpy` takes with a warning
    # about undefined behaviour on write. These are read and never written.
    labels = np.asarray(state, dtype=np.int64)
    probabilities = torch.zeros(rows.shape, dtype=torch.float64)
    probabilities[torch.arange(rows.shape[0]), torch.tensor(labels)] = 1.0
    probabilities.requires_grad_(True)

    value = (probabilities * torch.tensor(rows, dtype=torch.float64)).sum()
    if graph.edges:
        ends = graph.edge_index
        first = probabilities[torch.tensor(ends[:, 0], dtype=torch.long)]
        second = probabilities[torch.tensor(ends[:, 1], dtype=torch.long)]
        coupling = torch.tensor(graph.edge_coupling, dtype=torch.float64)
        value = value + (coupling * (first * second).sum(dim=1)).sum()
    (gradient,) = torch.autograd.grad(beta * value, probabilities)
    return log_ratios(gradient.detach().numpy(), labels)


def balanced_sweep_at(
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    move: PottsMove,
) -> Callable[[np.ndarray, np.random.Generator, float], None]:
    """``n_nodes`` locally balanced proposals per sweep, each Metropolis-corrected.

    The move set Zanella (2020) defines and Grathwohl et al. (2021) take the
    gradient form of; :mod:`sal.sample.balanced` holds the
    kernel both share and derives the correction. A sweep is ``n_nodes``
    proposals, the heat bath's sweep size, so an autocorrelation time in
    sweeps compares the two without a normalization.

    **The conditional is maintained, not rebuilt.** Every proposal reads every
    site's conditional, and rebuilding it from the adjacency per proposal
    would make a sweep quadratic in the lattice for no new information: a flip
    at ``i`` moves only the rows of ``i``'s neighbours, by the incident
    coupling. Those rows are copied before the flip and restored on a
    rejection, so a rejected proposal leaves the array it found rather than a
    value that has been added to and subtracted from. The array is rebuilt
    once per sweep, which is also what lets ``beta`` change between sweeps
    (:func:`anneal_potts`).

    The two move sets differ in one expression --- which estimate of the
    single-flip differences weights the neighbourhood --- and share the accept
    step, which takes the *exact* difference whichever proposed it.
    """
    owner = owner_rows(offsets)
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    gradient_informed = move is PottsMove.GIBBS_WITH_GRADIENTS

    def sweep(state: np.ndarray, rng: np.random.Generator, beta: float) -> None:
        local = local_fields(rows, state, neighbours, couplings, owner, beta)
        for _ in range(state.shape[0]):
            exact = log_ratios(local, state)
            estimate = (
                taylor_log_ratios(rows, state, neighbours, couplings, owner, beta)
                if gradient_informed
                else exact
            )
            forward_weights = log_balanced_weights(estimate, state)
            forward_total = log_normalizer(forward_weights)
            change = draw_change(forward_weights, forward_total, rng)
            node, colour = change.variable, change.value

            previous = int(state[node])
            log_ratio = float(exact[node, colour])
            forward = float(forward_weights[node, colour])

            # Advanced indexing already copies, so this is the restore buffer
            # and not a view of the rows about to change.
            touched = incident[bounds[node] : bounds[node + 1]]
            restored = local[touched]
            _apply_flip(local, state, node, colour, incident, weights, bounds, beta)
            reverse_estimate = (
                taylor_log_ratios(rows, state, neighbours, couplings, owner, beta)
                if gradient_informed
                else log_ratios(local, state)
            )
            reverse_weights = log_balanced_weights(reverse_estimate, state)
            reverse_total = log_normalizer(reverse_weights)
            reverse = float(reverse_weights[node, previous])

            log_alpha = log_metropolis_ratio(
                log_ratio, forward, forward_total, reverse, reverse_total
            )
            if not accept(log_alpha, rng):
                local[touched] = restored
                state[node] = previous

    return sweep


def _apply_flip(
    local: np.ndarray,
    state: np.ndarray,
    node: int,
    colour: int,
    incident: Sequence[int],
    weights: Sequence[float],
    bounds: Sequence[int],
    beta: float,
) -> None:
    """Relabel one site and carry the change into its neighbours' conditionals.

    The site's own row does not move: its conditional is built from its
    neighbours' labels and its own field, neither of which this touches.
    """
    previous = int(state[node])
    for position in range(bounds[node], bounds[node + 1]):
        neighbour, coupling = incident[position], beta * weights[position]
        local[neighbour, previous] -= coupling
        local[neighbour, colour] += coupling
    state[node] = colour


def swendsen_wang_sweep(
    state: np.ndarray,
    graph: PottsGraph,
    rows: SiteField | np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    beta: float = 1.0,
    backend: Backend = Backend.PYTHON,
) -> None:
    """Activate bonds, find clusters, recolour each one.

    Every cluster is recoloured independently, so in a field each needs its own
    accept step --- Wolff flips one cluster and needs one. Hence two code paths
    rather than one rule assumed to cover both.

    ``counter``, when given, records every cluster's size and whether its
    field accept step passed; issue #551 measures the acceptance against
    temperature and that is the quantity it reads.

    ``beta`` tempers the bond probability and the accept step together, the
    model scaling :func:`tempered` states, applied here rather than by
    rebuilding the graph per schedule step. At 1.0 it is the identity.

    ``backend`` names which implementation runs the pass.
    :data:`~sal.backend.Backend.PYTHON` is this one, the
    oracle, and is the default for the reason :func:`_cluster_pass_rust`
    states: the two draw the same uniforms in a different order, so the Rust
    pass is a chain of the same law and not the same chain. A ``counter`` is
    refused on the Rust route rather than silently ignored --- the
    instrumentation reads each cluster's members, which is the gather the
    port removes.

    The edge ends are :attr:`~sal.sim.graph.PottsGraph.edge_index`'s
    rather than two ``np.fromiter`` passes over ``graph.edges``: the graph has
    held the array form since #623, and rebuilding it per sweep was 0.900 ms
    of a 42.2 ms pass against 0.001 ms to read the store (#754). Recompute or
    store, decided as store, and the same ``int64`` indices either way.
    """
    rows = log_weight_of(rows)
    if backend is Backend.RUST:
        if counter is not None:
            msg = (
                "the Rust cluster pass takes no counter: the instrumentation "
                "reads every cluster's members, which is the gather the port "
                "removes (issue #551, #754)"
            )
            raise ValueError(msg)
        _cluster_pass_rust(state, graph, rows, rng, beta)
        return
    refuse_backend("the Swendsen-Wang pass", backend, (Backend.PYTHON, Backend.RUST))

    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < bond_probability(graph, beta))

    parent = np.arange(graph.n_nodes)
    for edge in np.flatnonzero(active):
        union_roots(parent, int(first[edge]), int(second[edge]))

    labels = np.array([find_root(parent, node) for node in range(graph.n_nodes)])
    # Scaled once rather than per cluster: the multiply is over the whole
    # field and there are as many clusters as sites at the transition, which
    # made it 10.5 ms of the same 41.2 ms pass. Every entry is the value the
    # per-cluster form produced, so no recolouring moves.
    scaled = beta * rows
    clusters = cluster_members(labels)
    # Bound once: the pass walks a cluster per 1.7 sites, and the lookups
    # would be paid per cluster (#754).
    order, bounds = clusters.order, clusters.bounds
    for cluster in range(bounds.size - 1):
        members = order[bounds[cluster] : bounds[cluster + 1]]
        outcome = _recolour(state, members, scaled, rng)
        if counter is not None:
            counter.record(members, outcome, graph)


def bond_probability(graph: PottsGraph, beta: float) -> np.ndarray:
    """``1 - exp(-beta J)`` per edge, in the graph's edge order.

    Factored out because both routes evaluate it and only one may: ``exp`` is
    a threshold the bond draw is compared against, so a second evaluation in
    Rust would decide an edge differently in the last place. The kernel takes
    this array and compares against it, which leaves the bond pass the
    oracle's arithmetic exactly (#754).
    """
    return np.asarray(1.0 - np.exp(-(beta * graph.edge_coupling)))


def _cluster_pass_rust(
    state: np.ndarray,
    graph: PottsGraph,
    rows: np.ndarray,
    rng: np.random.Generator,
    beta: float,
) -> None:
    """:func:`swendsen_wang_sweep`'s pass, on the extension.

    **The same law, in a different order of draws.** The oracle draws a
    cluster's colour and then, only where the field difference is negative,
    its accept uniform --- a lazy stream no array can replay, since what the
    next draw *is* depends on the last one's outcome. So this route draws the
    bond uniforms the oracle draws (one array, the same call), and then one
    colour and one uniform per cluster in bulk, each independent and
    identically distributed as the oracle's own: the chain is of the same law
    and is not the same chain, which is why
    :data:`~sal.backend.Backend.PYTHON` stays the default and
    why the pin is the enumerated law rather than the oracle's stream
    (`tests/regression/search/test_potts_mcmc_cluster_rust.py`).

    **Given the same draws it is the oracle bitwise, by construction.** The
    bond probability and the scaled field cross as arrays NumPy evaluated, so
    the only arithmetic the kernel adds is a cluster's field sum and ``exp``
    of the difference. Both are thresholds, and both are guarded by
    :data:`GUARD`: a cluster whose decision sits inside the width two
    summation orders and two ``exp`` implementations can move it is handed
    back, and decided here by :func:`_recolour_drawn` --- the oracle's own
    recolouring --- before the kernel resumes. The construction is
    :func:`sweep_at`'s, per cluster rather than per site.

    One crossing per call, and the boundary carries eight contiguous arrays:
    state and labels out, the flattened edge ends, the bond probabilities and
    their draws, the colour and accept draws, and the scaled field.
    """
    from sal import oxisal

    n_nodes, n_states = graph.n_nodes, int(rows.shape[1])
    scaled = np.ascontiguousarray(beta * rows, dtype=np.float64)
    edges = np.ascontiguousarray(graph.edge_index, dtype=np.int64).reshape(-1)
    probability = np.ascontiguousarray(bond_probability(graph, beta), dtype=np.float64)
    bond_draws = np.ascontiguousarray(rng.random(len(graph.edges)), dtype=np.float64)
    # One per cluster, indexed by the cluster's rank in increasing root order.
    # A pass builds at most one cluster per site, so the site count is the
    # bound the caller can know before the bond pass --- and learning the
    # real count would cost a second crossing.
    colour_draws = np.ascontiguousarray(
        rng.integers(0, n_states, size=n_nodes), dtype=np.int64
    )
    accept_draws = np.ascontiguousarray(rng.random(n_nodes), dtype=np.float64)
    labels = np.empty(n_nodes, dtype=np.int64)

    cluster = 0
    while True:
        n_clusters, cluster = oxisal.swendsen_wang_sweep(
            state,
            scaled,
            edges,
            probability,
            bond_draws,
            colour_draws,
            accept_draws,
            labels,
            GUARD,
            cluster,
        )
        if cluster >= n_clusters:
            return
        clusters = cluster_members(labels)
        order, bounds = clusters.order, clusters.bounds
        members = order[bounds[cluster] : bounds[cluster + 1]]
        # The draw behind a call rather than as a value, for the reason
        # `_recolour_drawn` gives: it is read only where the field difference
        # is negative, and `partial` is what says so without a closure over
        # the loop.
        _recolour_drawn(
            state,
            members,
            scaled,
            int(colour_draws[cluster]),
            partial(float, accept_draws[cluster]),
        )
        cluster += 1


@dataclass(frozen=True)
class AdjacencyLists:
    """The compressed adjacency as Python lists, converted once per chain (issue #919).

    A cluster grown site by site in Python indexes the adjacency one entry
    at a time, which is faster on lists than on NumPy scalars; converting
    the arrays on every step cost O(n_nodes + n_edges) before the first bond,
    which at small clusters was the step. A chain builds this once and hands
    it to every :func:`wolff_sweep` and :func:`niedermayer_sweep`.

    Parameters
    ----------
    bounds : list[int]
        ``offsets``: the incident edges of site ``i`` are ``bounds[i]`` to
        ``bounds[i + 1]``.
    incident : list[int]
        ``neighbours``, the far site of each incident edge.
    weights : list[float]
        ``couplings``, each incident edge's coupling.
    """

    bounds: list[int]
    incident: list[int]
    weights: list[float]


def adjacency_lists(
    offsets: np.ndarray, neighbours: np.ndarray, couplings: np.ndarray
) -> AdjacencyLists:
    """The compressed adjacency as the lists a cluster move walks."""
    return AdjacencyLists(offsets.tolist(), neighbours.tolist(), couplings.tolist())


def wolff_sweep(
    state: np.ndarray,
    rows: SiteField | np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    graph: PottsGraph | None = None,
    beta: float = 1.0,
    root: int | None = None,
    proposed: int | None = None,
    lists: AdjacencyLists | None = None,
) -> int:
    """Grow one cluster from a random seed, recolour it, and stop.

    Exactly one cluster per sweep, and the "exactly" is load-bearing. An earlier
    version ran clusters until their cumulative size reached ``n_nodes``, to
    spend the same budget as the other two move sets. That is a
    *state-dependent* stopping rule: an aligned configuration makes large
    clusters, so it reached the budget in fewer steps and received less
    randomization than a disordered one. Each step is correct, but stopping on
    the outcome biases the composition --- measured on a two-site chain at
    ``J = 0.7``, it put 0.384 on each aligned state against an exact 0.334, and
    the chi-square rejected it.

    One flip per sweep is therefore not one single-site sweep's work, and
    :func:`sample_potts` returns the mean cluster size so a comparison can be
    normalized.

    ``root`` and ``proposed``, when given, are the cluster's seed site and the
    colour to recolour it to rather than draws made here. That is issue #706's
    Wolff *action* --- a root, a label and a temperature --- so the choice
    belongs to the policy taking it, leaving the bond construction as the only
    randomness. Omitted, both are drawn as before and every existing caller's
    stream is bitwise unchanged.

    ``lists`` is the adjacency converted once by :func:`adjacency_lists`;
    omitted, it is converted here, O(n_nodes + n_edges) per step (#919). The
    arrays and the lists are the same adjacency, so the stream is bitwise
    the same either way.

    Returns
    -------
    int
        The size of the cluster this step built.
    """
    rows = log_weight_of(rows)
    walk = adjacency_lists(offsets, neighbours, couplings) if lists is None else lists
    bounds, incident, weights = walk.bounds, walk.incident, walk.weights
    seed_node = int(rng.integers(state.shape[0])) if root is None else int(root)
    colour = int(state[seed_node])
    cluster = [seed_node]
    in_cluster = np.zeros(state.shape[0], dtype=bool)
    in_cluster[seed_node] = True
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour, coupling = incident[position], weights[position]
            if in_cluster[neighbour] or state[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * coupling):
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)
    members = np.array(cluster, dtype=np.int64)
    outcome = _recolour(state, members, beta * rows, rng, proposed)
    if counter is not None:
        counter.record(members, outcome, graph)
    return len(cluster)


def niedermayer_threshold(couplings: np.ndarray) -> float:
    """Niedermayer's ``E_0`` for a graph, in the energy units of :func:`energies`.

    ``max(0, -min J)``: the smallest threshold at which every bond probability
    of :func:`niedermayer_sweep` is defined on this graph. On a ferromagnet it
    is 0 and the rule *is* Wolff's, bond for bond and to the last bit; on the
    uniform antiferromagnet it is ``|J|``, where bonds form between unlike
    sites --- which is what an antiferromagnet's satisfied bonds are --- and
    the construction is again exact, the accept step carrying the field alone.

    **It is the smallest, and that is the point.** The bond probabilities rise
    with ``E_0``, so a larger threshold buys nothing and costs the cluster:
    where the couplings are mixed the cluster already percolates here --- 8.94
    sites of 9 on the instance
    `tests/regression/search/test_potts_mcmc.py` measures --- and a single
    cluster that is the whole lattice is a global spin reversal, which is the
    reason Houdayer's pair move exists beside this one.

    Parameters
    ----------
    couplings : np.ndarray
        Edge couplings, as
        :meth:`~sal.sim.graph.PottsGraph.compressed_adjacency`
        lays them out or in any other order --- only the minimum is read.

    Returns
    -------
    float
        ``E_0``, never negative.
    """
    return max(0.0, -float(np.min(couplings))) if couplings.size else 0.0


def niedermayer_sweep(
    state: np.ndarray,
    rows: SiteField | np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    graph: PottsGraph | None = None,
    beta: float = 1.0,
    threshold: float = 0.0,
    root: int | None = None,
    partner: int | None = None,
    lists: AdjacencyLists | None = None,
) -> int:
    """Grow one cluster under Niedermayer's bond rule, transpose two colours on it.

    Niedermayer (1988) generalizes the Fortuin-Kasteleyn construction by
    activating a bond on its *energy relative to a threshold* ``E_0`` rather
    than on its endpoints agreeing. In the energy convention of
    :func:`energies` --- ``E = -h[s] - sum J [s_i = s_j]`` --- a bond's energy
    is ``-J`` where its endpoints agree and ``0`` where they do not, so the
    rule is

    ``p(bond) = 1 - exp(-beta * max(0, E_0 + J [s_i = s_j]))``,

    the ``max`` being what keeps it a probability for any ``E_0`` and any sign
    of ``J``. **Wolff is the case ``E_0 = 0`` on a ferromagnet**: the unlike
    bonds get ``p = 0`` and the like ones ``1 - exp(-beta J)``, which is
    :func:`wolff_sweep`'s own probability.

    ``lists`` is the adjacency converted once, as :func:`wolff_sweep` takes it.

    ``E_0`` is the one knob, and it runs between two algorithms. At or above
    :func:`niedermayer_threshold` every ``max`` is on its linear branch, the
    boundary terms below cancel exactly, and what is left is a
    Fortuin-Kasteleyn construction for a coupling of *either* sign whose only
    accept step is the field's --- Wolff's, where Wolff runs. Below it the
    like bonds of an antiferromagnet fall to ``p = 0``, the terms survive, and
    at ``E_0 = 0`` on an antiferromagnet no bond forms at all: the cluster is
    its seed and the step is a single-site Metropolis flip on the exact
    difference. The default is the threshold, so the default is the cluster
    algorithm.

    The cluster is then changed by the *transposition* of two colours rather
    than by a recolouring to one. Above ``E_0 = 0`` the cluster is no longer
    monochromatic, and a recolouring would change the agreement of its interior
    bonds --- the one thing the construction needs left alone, since those are
    the factors that cancel between the forward move and the reverse. A
    permutation of the colours cannot change an agreement, which is why it is
    the move that generalizes.

    What does not cancel is the boundary and the field, and that is the accept
    step. Writing ``x`` for ``[s_i = s_j]`` on a boundary bond before the
    transposition and ``x'`` for it after,

    ``delta = sum_C (h[new] - h[old]) + sum_boundary (g(x') - g(x))``,
    ``g(x) = J x - max(0, E_0 + J x)``,

    and the move is accepted with probability ``min(1, exp(beta * delta))``.
    ``g`` is constant in ``x`` wherever ``E_0`` and ``E_0 + J`` are both
    non-negative --- both are then ``-E_0`` --- so at or above the threshold
    every boundary term cancels and the accept step is the field's alone,
    which is what :func:`_recolour` applies for Wolff. Below the threshold
    they do not cancel, and they are what keeps the step a valid
    Metropolis-Hastings move where it is no longer a Fortuin-Kasteleyn one.

    Parameters
    ----------
    state, rows, offsets, neighbours, couplings, rng, counter, graph
        As :func:`wolff_sweep`. ``rows`` is the field at one row per site,
        untempered; ``beta`` scales it here rather than at the call site,
        because the bond rule and the accept step must carry the same one.
    beta : float
        Inverse temperature. ``math.inf`` is admitted and is the ``T = 0``
        limit taken exactly: a bond of positive energy margin is certain
        rather than drawn, and a ``delta`` below zero is refused without
        consuming a uniform.
    threshold : float
        ``E_0``. :func:`niedermayer_threshold` is the value that makes the
        rule Wolff's where Wolff runs.
    root : int | None
        The cluster's seed, or ``None`` to draw it uniformly.
    partner : int | None
        The colour the seed's own colour is transposed with, or ``None`` to
        draw it uniformly from the other ``n_states - 1``. Drawn from the
        colours rather than from the state, so the proposal is symmetric: the
        reverse move must be able to name the same unordered pair.

    Returns
    -------
    int
        The size of the cluster this step built.
    """
    rows = log_weight_of(rows)
    n_nodes = int(state.shape[0])
    n_states = int(rows.shape[1])
    walk = adjacency_lists(offsets, neighbours, couplings) if lists is None else lists
    bounds, incident, weights = walk.bounds, walk.incident, walk.weights
    labels = state.tolist()

    seed_node = int(rng.integers(n_nodes)) if root is None else int(root)
    held = int(labels[seed_node])
    if partner is None:
        swapped = (held + 1 + int(rng.integers(n_states - 1))) % n_states
    else:
        swapped = int(partner)

    in_cluster = np.zeros(n_nodes, dtype=bool)
    in_cluster[seed_node] = True
    cluster = [seed_node]
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if in_cluster[neighbour]:
                continue
            margin = threshold + (
                weights[position] if labels[neighbour] == labels[node] else 0.0
            )
            if margin <= 0.0:
                continue
            probability = 1.0 - np.exp(-beta * margin)
            # Only `beta = inf` reaches one, and there the bond is certain
            # rather than drawn: the T = 0 limit without a second branch.
            if probability >= 1.0 or rng.random() < probability:
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)

    members = np.array(cluster, dtype=np.int64)
    delta = 0.0
    for node in cluster:
        current = labels[node]
        if current == held:
            moved = swapped
        elif current == swapped:
            moved = held
        else:
            continue
        delta += float(rows[node, moved] - rows[node, current])
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if in_cluster[neighbour]:
                continue
            # `g(x') - g(x)` for this boundary bond, with `beta` divided out:
            # the log-density's own term, then the bond probabilities' one.
            coupling = weights[position]
            before = 1.0 if labels[neighbour] == current else 0.0
            after = 1.0 if labels[neighbour] == moved else 0.0
            delta += coupling * (after - before)
            delta -= max(0.0, threshold + coupling * after)
            delta += max(0.0, threshold + coupling * before)

    accepted = _niedermayer_accept(delta, beta, rng)
    if accepted:
        held_members = members[state[members] == held]
        swapped_members = members[state[members] == swapped]
        state[held_members] = swapped
        state[swapped_members] = held
    if counter is not None:
        counter.record(members, Recolour(proposed=True, accepted=accepted), graph)
    return len(cluster)


def _niedermayer_accept(delta: float, beta: float, rng: np.random.Generator) -> bool:
    """:func:`~sal.sample.accept.accept_at` on ``delta``, under a name.

    A separate function because it is what an ablation replaces:
    `tests/regression/search/test_potts_mcmc.py` swaps in an unconditional
    accept and asserts the enumerated chi-square rejects, which is the
    evidence that the tests above have the power they claim. The ``beta = inf``
    limit it takes is stated where the accept step lives (issue #857).
    """
    return accept_at(beta, delta, rng)


def _like_bonds(
    state: np.ndarray, graph: PottsGraph, rng: np.random.Generator, beta: float
) -> np.ndarray:
    """The Fortuin-Kasteleyn bonds of one pass, as ``(n_bonds, 2)`` site pairs.

    One uniform per edge, drawn as :func:`swendsen_wang_sweep` draws them, and
    compared against :func:`bond_probability` on the edges whose ends agree.
    """
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < bond_probability(graph, beta))
    return np.asarray(graph.edge_index[active])


def ghost_couplings(rows: SiteField | np.ndarray) -> np.ndarray:
    """``K[i, a] = h[i, a] - min_b h[i, b]``, each site's couplings to the ghosts of :func:`ghost_spin_sweep`.

    A function so the ablation in
    `tests/regression/sample/test_potts_cluster_field.py` can replace the
    shift with ``max(0, h)`` alone, which drops the negative part of the
    field, and show the enumeration refutes it.
    """
    rows = log_weight_of(rows)
    return np.asarray(rows - rows.min(axis=1, keepdims=True))


def ghost_spin_sweep(
    state: np.ndarray,
    graph: PottsGraph,
    rows: SiteField | np.ndarray,
    rng: np.random.Generator,
    beta: float = 1.0,
    backend: Backend = Backend.RUST,
    ghost: np.ndarray | None = None,
) -> None:
    """Swendsen-Wang with the field as bonds to one ghost site per label (issue #1041).

    The per-site field is a coupling to ``q`` ghost sites, ghost ``a`` fixed
    at label ``a``: ``-h[i, s_i] = -sum_a h[i, a] [s_i = a]``. A per-site
    constant leaves the law unchanged, so the row is shifted to
    ``K[i, a] = h[i, a] - min_b h[i, b] >= 0``, every ghost coupling is
    ferromagnetic, and the bond between site ``i`` and the ghost of its own
    label forms with probability ``1 - exp(-beta K[i, s_i])``: the ticket's
    ``1 - exp(-beta max(0, h_a))`` on the shifted row, where the ``max`` is
    then the identity. Two ghosts carry different labels, so no cluster
    reaches both.

    **No accept step.** The field is inside the Fortuin-Kasteleyn measure, so
    given the bonds a cluster bonded to a ghost keeps that ghost's label and
    every other cluster takes a uniform label: the heat bath on the joint
    measure, as Swendsen-Wang is in zero field. A site whose field prefers
    another label rarely bonds to its current label's ghost, so its cluster is
    the one that moves.

    ``backend`` merges the bonds (:func:`bond_roots`); the two give the same
    roots, so the same chain. ``ghost`` is :func:`ghost_couplings` computed
    once by a caller running many passes: recomputed per pass, its row
    minimum was 0.41 s of a 1.41 s release-size anneal of 873 passes.

    Draws: one uniform per edge, one per site, one label per site.
    """
    rows = log_weight_of(rows)
    n_nodes, n_states = graph.n_nodes, int(rows.shape[1])
    bonds = _like_bonds(state, graph, rng, beta)
    couplings = ghost_couplings(rows) if ghost is None else ghost
    own = couplings[np.arange(n_nodes), state]
    ghosted = rng.random(n_nodes) < 1.0 - np.exp(-beta * own)
    roots = bond_roots(n_nodes, bonds, backend=backend)
    frozen = np.zeros(n_nodes, dtype=bool)
    frozen[roots[ghosted]] = True
    # One label per site, read at each free cluster's root.
    labels = rng.integers(0, n_states, size=n_nodes)
    free = ~frozen[roots]
    state[free] = labels[roots[free]]


def _label_hastings(n_states: int) -> float:
    """``log q``, the Hastings term of :func:`label_directed_sweep`, under a name.

    A function so the ablation in
    `tests/regression/sample/test_potts_cluster_field.py` can set it to zero
    and show the enumeration refutes the kernel without it.
    """
    return float(np.log(n_states))


def label_directed_sweep(
    state: np.ndarray,
    graph: PottsGraph,
    rows: SiteField | np.ndarray,
    rng: np.random.Generator,
    target: int,
    beta: float = 1.0,
    backend: Backend = Backend.RUST,
) -> None:
    """Fortuin-Kasteleyn clusters, each proposed onto one label, by Metropolis-Hastings (issue #1041).

    The bonds are :func:`swendsen_wang_sweep`'s. Given them, the clusters are
    independent and cluster ``C`` at label ``c`` has weight
    ``exp(beta sum_C h[i, c])``, the coupling having cancelled. A cluster off
    ``target`` proposes ``target``; a cluster at ``target`` proposes a label
    drawn uniformly from all ``q``, its own included. The proposal is not
    symmetric, so the log ratio carries its Hastings term, ``-log q`` onto
    ``target`` and ``+log q`` off it (:func:`_label_hastings`). The kernel at
    one ``target`` keeps the Boltzmann law, so a caller cycling ``target``
    through the labels keeps it too.

    **The draw from all ``q`` labels is what makes the chain aperiodic.**
    Drawn from the other ``q - 1``, at ``q = 2`` in zero field every
    proposal is accepted and every cluster changes label each pass: the
    lattice's global flip, the same two configurations in turn, which the
    two-label chi-square of `tests/regression/sample/test_potts_mcmc.py`
    rejected at p = 0.0.

    ``backend`` merges the bonds, as in :func:`ghost_spin_sweep`.

    Draws: one uniform per edge, one label per site and one uniform per site,
    each read at a cluster's root.
    """
    rows = log_weight_of(rows)
    n_nodes, n_states = graph.n_nodes, int(rows.shape[1])
    bonds = _like_bonds(state, graph, rng, beta)
    roots = bond_roots(n_nodes, bonds, backend=backend)
    partners = rng.integers(0, n_states, size=n_nodes)
    # A cluster is monochromatic, so each site's `at_target` is its cluster's.
    at_target = state == target
    proposed = np.where(at_target, partners[roots], target)
    sites = np.arange(n_nodes)
    gain = beta * (rows[sites, proposed] - rows[sites, state])
    # Summed per cluster at its root's index; only roots are read below.
    log_ratio = np.bincount(roots, weights=gain, minlength=n_nodes)
    hastings = _label_hastings(n_states)
    log_ratio += np.where(at_target, hastings, -hastings)
    uniforms = rng.random(n_nodes)
    moved = (uniforms < np.exp(np.minimum(log_ratio, 0.0)))[roots]
    state[moved] = proposed[moved]


def houdayer_cluster(
    first: np.ndarray, second: np.ndarray, offsets: np.ndarray, neighbours: np.ndarray
) -> np.ndarray:
    """Component index per site over the bonds joining two sites where the replicas disagree.

    The overlap ``q_i = s_i s'_i`` of Houdayer (2001) reads ``-1`` exactly where
    the two replicas disagree, and his cluster is a connected component of that
    region. Sites where the replicas agree are their own singletons here, so
    one array answers both questions a caller has --- which sites are in the
    defect region, and which component each of them is in.

    Uses :func:`find_root` and :func:`union_roots`, so the components are not a second
    reading of what a component is, and walks the compressed rows rather than
    the graph's edge tuples (root `CLAUDE.md`'s layout rule).

    Parameters
    ----------
    first, second : np.ndarray
        The two replicas' labellings, ``(n_nodes,)``.
    offsets, neighbours : np.ndarray
        The compressed adjacency, as
        :meth:`~sal.sim.graph.PottsGraph.compressed_adjacency`
        lays it out.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` of component roots, as :func:`find_root` reports them.
    """
    n_nodes = int(offsets.shape[0]) - 1
    parent = np.arange(n_nodes)
    defect = (first != second).tolist()
    bounds, incident = offsets.tolist(), neighbours.tolist()
    for node in range(n_nodes):
        if not defect[node]:
            continue
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if neighbour > node and defect[neighbour]:
                union_roots(parent, node, neighbour)
    return np.array([find_root(parent, node) for node in range(n_nodes)])


def houdayer_move(
    first: np.ndarray,
    second: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Swap the two replicas' labels on one component of their overlap defect.

    Houdayer's isoenergetic cluster move (2001), in the form his paper states
    for two replicas at one temperature. The cluster is a connected component
    of ``{i : s_i != s'_i}``, and on it the two replicas exchange labels.

    **The acceptance is 1, and it is an identity rather than a cancellation.**
    A bond with both ends in the cluster has its pair of agreements exchanged
    between the replicas, so the pair's energy is unchanged; a bond with one
    end in it has its other end where the replicas agree --- a neighbour that
    disagreed would be in the same component --- so the two terms are again
    exchanged. The field term is exchanged site by site for the same reason.
    So ``E(s) + E(s')`` is invariant, the move is an involution, and the
    defect region it is built from is what the swap leaves alone, which makes
    the reverse proposal exactly as likely as the forward one.

    Parameters
    ----------
    first, second : np.ndarray
        The two replicas, mutated in place.
    offsets, neighbours : np.ndarray
        The compressed adjacency.
    rng : np.random.Generator
        Draws the defect site the component is grown from.

    Returns
    -------
    int
        The size of the cluster swapped; ``0`` where the replicas agree
        everywhere and there is no defect to move.
    """
    defects = np.flatnonzero(first != second)
    if defects.size == 0:
        return 0
    partition = houdayer_cluster(first, second, offsets, neighbours)
    seed_node = int(defects[rng.integers(defects.size)])
    members = np.flatnonzero(partition == partition[seed_node])
    held = first[members].copy()
    first[members] = second[members]
    second[members] = held
    return int(members.size)


def _recolour(
    state: np.ndarray,
    members: np.ndarray,
    rows: np.ndarray,
    rng: np.random.Generator,
    proposed: int | None = None,
) -> Recolour:
    """Propose one colour for a whole cluster, accepting on the field alone.

    The proposal is uniform over every colour including the current one, which
    makes it symmetric and leaves the acceptance ratio as the field term alone.
    Drawing from the ``k - 1`` other colours would mix marginally faster and
    would need the proposal ratio carried through the acceptance.

    The bond construction contributes nothing to the ratio: bonds live only
    between like-coloured sites, and every site in the cluster changes colour
    together, so the cluster is exactly as likely to be built in the proposed
    configuration as in the current one. What does not cancel is the field.
    For a field shared by every site that difference is ``|C| * (h[new] -
    h[old])``; for the per-site field of `spatio_only` it is the sum of that
    difference over the cluster's own members, which the shared case is the
    special case of.

    ``proposed``, when given, is the colour to try rather than one drawn here:
    issue #706's Wolff action names the colour it recolours to, so the draw
    belongs to the action rather than to this function. Omitted, the draw is
    unchanged, which is what keeps every existing caller's stream bitwise as it
    was.

    Returns
    -------
    Recolour
        Whether a colour change was proposed at all, and whether it was
        accepted. Issue #551 reads the acceptance against temperature, and a
        run that never proposed is not a run that was rejected.
    """
    if proposed is None:
        proposed = int(rng.integers(rows.shape[1]))
    return _recolour_drawn(state, members, rows, proposed, rng.random)


def _recolour_drawn(
    state: np.ndarray,
    members: np.ndarray,
    rows: np.ndarray,
    proposed: int,
    draw: Callable[[], float],
) -> Recolour:
    """:func:`_recolour` with the colour already chosen and the uniform behind a call.

    The whole of the oracle's recolouring, factored out so the Rust pass's
    hand-back path decides its cluster by calling this rather than a copy of
    it --- :func:`site_update`'s place in :func:`sweep_at`, one level up
    (issues #599, #754).

    ``draw`` is a callable and not a float because the acceptance is
    *conditional*: the oracle consumes a uniform only where the field
    difference is negative, and taking one eagerly would advance the
    generator on a cluster that never needed it and move every chain after
    it. The caller passes ``rng.random``; the hand-back passes the draw the
    kernel was given.
    """
    current = int(state[members[0]])
    if proposed == current:
        return Recolour(proposed=False, accepted=False)
    difference = float(rows[members, proposed].sum() - rows[members, current].sum())
    if accept_drawn(difference, draw):
        state[members] = proposed
        return Recolour(proposed=True, accepted=True)
    return Recolour(proposed=True, accepted=False)


def bond_roots(
    n_nodes: int, bonds: np.ndarray, *, backend: Backend = Backend.RUST
) -> np.ndarray:
    """Every node's union-find root once ``bonds`` are merged in order.

    The labelling the Swendsen--Wang pass forms: two nodes share a root
    exactly when a chain of bonds joins them, and which node is the root is
    :func:`union_roots`' rule, keeping the first bond end's. Issue #986:
    :data:`~sal.backend.Backend.RUST`, the default, is
    ``oxisal.bond_roots``, the same :func:`find_root` and
    :func:`union_roots` compiled; :data:`~sal.backend.Backend.PYTHON`
    is the loop over them and the oracle that pins the roots bitwise.

    Parameters
    ----------
    n_nodes : int
        Nodes, labelled ``0 .. n_nodes - 1``.
    bonds : np.ndarray
        ``(n_bonds, 2)`` node pairs, merged in row order.
    backend : Backend
        Which implementation merges them.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` ``int64`` roots.
    """
    pairs = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    if backend is Backend.RUST:
        from sal import oxisal

        return oxisal.bond_roots(
            n_nodes,
            np.ascontiguousarray(pairs[:, 0]),
            np.ascontiguousarray(pairs[:, 1]),
        )
    parent = np.arange(n_nodes)
    for first, second in pairs.tolist():
        union_roots(parent, first, second)
    return np.array(
        [find_root(parent, node) for node in range(n_nodes)], dtype=np.int64
    )


def find_root(parent: np.ndarray, node: int) -> int:
    """Union-find root, with path compression."""
    root = node
    while parent[root] != root:
        root = int(parent[root])
    while parent[node] != root:
        parent[node], node = root, int(parent[node])
    return root


def union_roots(parent: np.ndarray, first: int, second: int) -> None:
    """Merge two components."""
    first_root, second_root = find_root(parent, first), find_root(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root
