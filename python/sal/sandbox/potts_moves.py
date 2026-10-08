"""The Niedermayer, ghost-spin and label-directed Potts moves, no longer supported (issue #1365).

Moved here on the sandbox's "unsupported" rule, by the owner's decision
alone, with no Rust port: the three cluster moves that left
:class:`~sal.sample.potts_mcmc.PottsMove` as :class:`SandboxMove`, their
pure-Python kernels (:func:`niedermayer_sweep`, :func:`ghost_spin_sweep`,
:func:`label_directed_sweep` and their helpers), the keyed
:class:`NiedermayerMove` that left ``sample.potts_keyed`` with
``MoveKind.NIEDERMAYER``, and the drivers that run them:
:func:`sample_potts`, :func:`sample_potts_pair`, :func:`anneal_potts` and
:func:`run_annealed`, each its supported namesake on a lattice that also
runs these moves. None has a heat-bath recolouring, so each runs under the
uniform proposal and none takes ``recolour`` (#1322).

**Admitted private imports**, each because the driver is the supported one
and must record, charge and anneal alike: ``chains._Lattice``,
``_record_chain``, ``_record_pair`` and ``_anneal_lattice`` (the loops,
issue #1365); ``sweeps._like_bonds`` and ``sweeps._recolour_drawn`` (the
bonds and the forbidden-label rule the kernels share with heat-bath
Swendsen-Wang and Wolff); ``ground_state._problem`` (the instance a run
reads).

The tests moved with them, under ``tests/regression/sandbox/``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from sal.backend import Backend
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.sample.accept import accept_at
from sal.sample.potts_mcmc import (
    AdjacencyLists,
    AnnealedPotts,
    ClusterCounter,
    PottsChain,
    PottsMove,
    PottsPair,
    RecolourOutcome,
    adjacency_lists,
    bond_roots,
    refuse_negative_coupling,
    tempered,
)
from sal.sample.potts_mcmc.chains import (
    _anneal_lattice,
    _Lattice,
    _record_chain,
    _record_pair,
    step_visits,
)
from sal.sample.potts_mcmc.sweeps import _like_bonds
from sal.sample.schedule import ScheduleParams, TempSchedule
from sal.search.ground_state import (
    ANNEAL_SCHEDULE,
    MethodRun,
    Problem,
    Rung,
    _problem,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, check_labelling, log_weight_of, site_field


class SandboxMove(StrEnum):
    """The cluster moves that left :class:`~sal.sample.potts_mcmc.PottsMove` (issue #1365)."""

    NIEDERMAYER = "niedermayer"
    # Issue #1041: the field as bonds to a ghost site per label, and
    # Fortuin-Kasteleyn clusters proposed onto one label at a time.
    GHOST_SPIN = "ghost-spin"
    LABEL_DIRECTED = "label-directed"


#: The moves built on the Fortuin-Kasteleyn bonds, refused on a negative
#: coupling as the supported cluster moves are. Niedermayer's rule builds
#: clusters on a coupling of either sign and is not here (issue #756).
_FK_MOVES = frozenset({SandboxMove.GHOST_SPIN, SandboxMove.LABEL_DIRECTED})

#: A move set as these drivers take it: one move, or a sequence run in order.
Moves = SandboxMove | PottsMove | Sequence[SandboxMove | PottsMove]


def _moves(move: Moves, graph: PottsGraph) -> tuple[SandboxMove | PottsMove, ...]:
    """``move`` as a tuple run in order, refusing a Fortuin-Kasteleyn move on a negative coupling."""
    moves = (move,) if isinstance(move, SandboxMove | PottsMove) else tuple(move)
    if not moves:
        msg = "move is a SandboxMove or PottsMove, or a non-empty sequence of them"
        raise ValueError(msg)
    for each in moves:
        if each in _FK_MOVES and min(graph.coupling, default=0.0) < 0.0:
            msg = (
                f"{each} needs every coupling >= 0: the bond probability "
                "1 - exp(-J) is not a probability for J < 0, and an "
                "antiferromagnet has no like-spin clusters to flip"
            )
            raise ValueError(msg)
        if isinstance(each, PottsMove):
            refuse_negative_coupling((each,), graph)
    return moves


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

    A forbidden label (``-inf`` in ``rows``) sets ``delta`` rather than adding
    to it (issue #1146): ``-inf`` where the transposition moves any member
    onto one, else ``+inf`` where it moves any off one --- the rule of
    :func:`_recolour_drawn`.

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
    # A forbidden label (`-inf`) is decided rather than summed, as
    # `_recolour_drawn` decides it (issue #1146): any member moved onto one
    # rejects, else any moved off one accepts. Summed, `-inf - (-inf)` and
    # `inf + (-inf)` were `nan`; a finite term is added as before.
    entering = leaving = False
    for node in cluster:
        current = labels[node]
        if current == held:
            moved = swapped
        elif current == swapped:
            moved = held
        else:
            continue
        to, origin = rows[node, moved], rows[node, current]
        if to == -np.inf:
            entering = True
        elif origin == -np.inf:
            leaving = True
        else:
            delta += float(to - origin)
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

    if entering:
        delta = -np.inf
    elif leaving:
        delta = np.inf
    accepted = _niedermayer_accept(delta, beta, rng)
    if accepted:
        held_members = members[state[members] == held]
        swapped_members = members[state[members] == swapped]
        state[held_members] = swapped
        state[swapped_members] = held
    if counter is not None:
        counter.record(
            members, RecolourOutcome(proposed=True, accepted=accepted), graph
        )
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


def ghost_couplings(rows: SiteField | np.ndarray) -> np.ndarray:
    """``K[i, a] = h[i, a] - min_b h[i, b]``, each site's couplings to the ghosts of :func:`ghost_spin_sweep`.

    A function so the ablation in
    `tests/regression/sample/test_potts_cluster_field.py` can replace the
    shift with ``max(0, h)`` alone, which drops the negative part of the
    field, and show the enumeration refutes it.

    The minimum is over each site's allowed labels (issue #1154). A forbidden
    label (``-inf``, :func:`~sal.sim.potts.forbid`) keeps ``K = -inf``, which
    :func:`ghost_spin_sweep` reads as a hard constraint rather than a bond.
    Over every label the minimum was ``-inf``, every allowed ``K`` was
    ``+inf``, and each site bonded to its own ghost with probability one: the
    chain froze. A finite field's ``K`` is the bits it was.
    """
    rows = log_weight_of(rows)
    forbidden = np.isneginf(rows)
    if not forbidden.any():
        return np.asarray(rows - rows.min(axis=1, keepdims=True))
    floor = np.where(forbidden, np.inf, rows).min(axis=1, keepdims=True)
    return np.asarray(rows - floor)


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

    **A forbidden label is a hard ghost bond** (issue #1154). A ``-inf``
    entry is ``exp(beta h[i, a] [s_i = a]) = [s_i != a]``: the
    antiferromagnetic limit of a ghost bond, present with probability one and
    constraining site ``i`` off label ``a``. So a free cluster draws uniformly
    from the labels every member allows, and the shift is over the allowed
    labels alone (:func:`ghost_couplings`). From an allowed labelling no move
    enters a forbidden label, and the law restricted to the allowed
    labellings is kept: the construction is the joint measure above, with
    weight zero off the allowed set. A site on a forbidden label --- only
    reachable from a forbidden start --- has no ghost bond, and a free cluster
    whose members allow no common label keeps its own, as
    :func:`swendsen_wang_heat_bath_sweep` keeps it. ``beta = 0`` is the uniform
    law over each site's allowed labels: no bond forms, and ``0 * -inf`` is
    never formed.

    Draws: one uniform per edge, one per site, one label per site; on a field
    with a forbidden label, one uniform per site in place of the label.
    """
    rows = log_weight_of(rows)
    n_nodes, n_states = graph.n_nodes, int(rows.shape[1])
    bonds = _like_bonds(state, graph, rng, beta)
    couplings = ghost_couplings(rows) if ghost is None else ghost
    # One reduction decides the path; the mask is built only where it is read.
    forbidden = bool(couplings.min() == -np.inf)
    hard = np.isneginf(couplings) if forbidden else None
    sites = np.arange(n_nodes)
    own = couplings[sites, state]
    if hard is not None:
        # A site on a forbidden label bonds to no ghost: `-beta * -inf` was
        # `inf`, or `nan` at `beta = 0`.
        own = np.where(hard[sites, state], 0.0, own)
    ghosted = rng.random(n_nodes) < 1.0 - np.exp(-beta * own)
    roots = bond_roots(n_nodes, bonds, backend=backend)
    frozen = np.zeros(n_nodes, dtype=bool)
    frozen[roots[ghosted]] = True
    free = ~frozen[roots]
    if hard is None:
        # One label per site, read at each free cluster's root.
        labels = rng.integers(0, n_states, size=n_nodes)
        state[free] = labels[roots[free]]
        return
    # The labels each cluster's members allow between them, at its root.
    blocked = np.zeros(n_nodes * n_states, dtype=bool)
    site, label = np.nonzero(hard)
    blocked[roots[site] * n_states + label] = True
    allowed = ~blocked.reshape(n_nodes, n_states)
    count = allowed.sum(axis=1)
    # One uniform per site, read at each free cluster's root as the index of
    # a label among the allowed ones.
    pick = np.minimum((rng.random(n_nodes) * count).astype(np.int64), count - 1)
    labels = np.argmax(np.cumsum(allowed, axis=1) > pick[:, None], axis=1)
    free &= count[roots] > 0
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

    **A forbidden label is decided, not summed** (issue #1154), by the rule of
    :func:`_recolour_drawn`: a proposal onto a label any member forbids is
    rejected, else one moving any member off a forbidden label is accepted,
    each on the uniform the cluster draws anyway. Summed, a site forbidding
    both labels gave ``-inf - (-inf) = nan``, and a cluster ``inf + (-inf)``.
    The law restricted to the allowed labellings is kept at every ``beta``,
    ``beta = 0`` included, where it is uniform over them; an all-finite chain
    is bitwise unchanged.

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
    to, origin = rows[sites, proposed], rows[sites, state]
    if min(to.min(), origin.min()) > -np.inf:
        gain = beta * (to - origin)
        # Summed per cluster at its root's index; only roots are read below.
        log_ratio = np.bincount(roots, weights=gain, minlength=n_nodes)
    else:
        entering, leaving = np.isneginf(to), np.isneginf(origin)
        # The finite terms are summed as before; a forbidden one sets the
        # cluster's ratio, onto one before off one.
        difference = np.subtract(
            to, origin, out=np.zeros(n_nodes), where=~(entering | leaving)
        )
        log_ratio = np.bincount(roots, weights=beta * difference, minlength=n_nodes)
        log_ratio[np.bincount(roots, weights=leaving, minlength=n_nodes) > 0] = np.inf
        log_ratio[np.bincount(roots, weights=entering, minlength=n_nodes) > 0] = -np.inf
    hastings = _label_hastings(n_states)
    log_ratio += np.where(at_target, hastings, -hastings)
    uniforms = rng.random(n_nodes)
    moved = (uniforms < np.exp(np.minimum(log_ratio, 0.0)))[roots]
    state[moved] = proposed[moved]


class NiedermayerMove:
    """One cluster grown under Niedermayer's bond rule, two colours transposed on it.

    Wolff's arm generalized (issue #756):
    :func:`niedermayer_sweep` activates a
    bond on its energy relative to a threshold ``E_0`` rather than on its
    endpoints agreeing, so the arm runs on a coupling of either sign, and at
    :func:`niedermayer_threshold`'s value
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
        self._lists = adjacency_lists(offsets, neighbours, couplings)
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
        size = niedermayer_sweep(
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
            lists=self._lists,
        )
        return labels, size * self._per_member


@dataclass(frozen=True)
class _SandboxLattice(_Lattice):
    """:class:`~sal.sample.potts_mcmc.chains._Lattice` that also runs and charges :class:`SandboxMove`."""

    def visits(self, move: PottsMove | SandboxMove, cluster_sites: int) -> int:
        """Site visits one step costs: the ghost-spin pass reads one ghost bond per site beside a sweep."""
        graph = self.graph
        per_sweep = graph.n_nodes + 2 * len(graph.edges)
        if move is SandboxMove.GHOST_SPIN:
            return per_sweep + graph.n_nodes
        if move is SandboxMove.LABEL_DIRECTED:
            return per_sweep
        if move is SandboxMove.NIEDERMAYER:
            return cluster_sites * (1 + 2 * len(graph.edges) // graph.n_nodes)
        return step_visits(move, graph, cluster_sites)

    def sweep(
        self, move: PottsMove | SandboxMove, trace: list[ClusterCounter] | None = None
    ) -> Callable[[np.ndarray, np.random.Generator, float], int]:
        """One pass of ``move``; a :class:`PottsMove` is the supported lattice's.

        Niedermayer keeps a :class:`ClusterCounter` per step in ``trace``; the
        ghost-spin and label-directed passes build their clusters as roots
        and keep none (issue #1041).
        """
        if not isinstance(move, SandboxMove):
            return super().sweep(move, trace)
        graph, rows, cluster_backend = self.graph, self.rows, self.cluster_backend
        offsets, neighbours, couplings = self.adjacency
        if move is SandboxMove.NIEDERMAYER:
            lists = adjacency_lists(offsets, neighbours, couplings)
            threshold = niedermayer_threshold(couplings)
            arrays = (rows, offsets, neighbours, couplings)

            def grow(
                state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
            ) -> int:
                counter = ClusterCounter() if trace is not None else None
                size = niedermayer_sweep(
                    state,
                    *arrays,
                    rng,
                    counter,
                    graph,
                    beta=beta,
                    threshold=threshold,
                    lists=lists,
                )
                if counter is not None and trace is not None:
                    trace.append(counter)
                return size

            return grow
        if move is SandboxMove.GHOST_SPIN:
            # The ghost couplings, fixed by the field, once (#1041); read from
            # this module, where a test replaces it.
            ghost = ghost_couplings(rows)

            def ghost_pass(
                state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
            ) -> int:
                ghost_spin_sweep(
                    state, graph, rows, rng, beta, backend=cluster_backend, ghost=ghost
                )
                return 0

            return ghost_pass
        # The label-directed target cycles through the labels, one per call.
        calls = [0]

        def directed_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            target = calls[0] % int(rows.shape[1])
            calls[0] += 1
            label_directed_sweep(
                state, graph, rows, rng, target, beta, backend=cluster_backend
            )
            return 0

        return directed_pass


def _lattice(
    graph: PottsGraph, rows: np.ndarray, cluster_backend: Backend
) -> _SandboxLattice:
    """The lattice the drivers run on, heat-bath sweeps on the default backend."""
    return _SandboxLattice(
        graph, rows, graph.compressed_adjacency(), Backend.RUST, cluster_backend
    )


def _as_supported(moves: tuple[SandboxMove | PottsMove, ...]) -> tuple[PottsMove, ...]:
    """``moves`` typed as the shared loops take them; :class:`_SandboxLattice` dispatches each."""
    return cast("tuple[PottsMove, ...]", moves)


def sample_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    move: Moves,
    rng: np.random.Generator,
    n_sweeps: int | Budget,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
) -> PottsChain:
    """:func:`~sal.sample.potts_mcmc.sample_potts` with ``move`` a :class:`SandboxMove` (issue #1365).

    The same draws in the same order as the supported driver ran these moves
    before #1365, so a fixed-seed chain is bitwise what it was.
    """
    field = log_weight_of(field)
    moves = _moves(move, graph)
    model = tempered(graph, field, temperature)
    graph, field = model.graph, model.field
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    if start is None:
        state = np.ascontiguousarray(
            rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
        )
    else:
        state = np.ascontiguousarray(
            check_labelling(start, graph.n_nodes, n_states), dtype=np.int64
        )
    lattice = _lattice(graph, rows, cluster_backend)
    return _record_chain(
        lattice, _as_supported(moves), state, rng, n_sweeps, burn_in, thin
    )


def sample_potts_pair(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    move: Moves,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    houdayer: bool = True,
    cluster_backend: Backend = Backend.RUST,
) -> PottsPair:
    """:func:`~sal.sample.potts_mcmc.sample_potts_pair` with ``move`` a :class:`SandboxMove` (issue #1365)."""
    field = log_weight_of(field)
    moves = _moves(move, graph)
    model = tempered(graph, field, temperature)
    graph, field = model.graph, model.field
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    if houdayer and n_states != 2:
        msg = (
            f"Houdayer's move is defined on the Ising overlap q_i = s_i s'_i "
            f"and this model has {n_states} states: pass houdayer=False, or "
            "use two states (issue #756)"
        )
        raise ValueError(msg)
    lattice = _lattice(graph, rows, cluster_backend)
    return _record_pair(
        lattice, _as_supported(moves), rng, n_sweeps, burn_in, thin, houdayer=houdayer
    )


def anneal_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    schedule: TempSchedule,
    rng: np.random.Generator,
    *,
    move: Moves,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
    budget: Budget | None = None,
) -> AnnealedPotts:
    """:func:`~sal.sample.potts_mcmc.anneal_potts` with ``move`` a :class:`SandboxMove`, on a given schedule (issue #1365)."""
    field = log_weight_of(field)
    moves = _moves(move, graph)
    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    lattice = _lattice(graph, rows, cluster_backend)
    return _anneal_lattice(
        lattice, _as_supported(moves), schedule, None, rng, start, budget
    )


def step_cost(problem: Problem | Rung, move: SandboxMove) -> int:
    """Site visits one step of ``move`` is budgeted at: a sweep's, and a ghost bond per site for the ghost-spin pass."""
    extra = problem.n_nodes if move is SandboxMove.GHOST_SPIN else 0
    return problem.visits_per_sweep + extra


def run_annealed(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    move: SandboxMove,
    *,
    schedule: ScheduleParams = ANNEAL_SCHEDULE,
    steps: int | None = None,
    start: np.ndarray | None = None,
) -> MethodRun:
    """:func:`~sal.search.ground_state.run_annealed` with ``move`` a :class:`SandboxMove` (issue #1365).

    The ghost-spin and label-directed passes merge their bonds on the
    compiled union-find, which returns the Python roots, so the chain;
    Niedermayer runs its Python kernel, as before #1365.
    """
    problem = _problem(problem)
    count = max(1, budget.size // step_cost(problem, move)) if steps is None else steps
    started = time.perf_counter()
    run = anneal_potts(
        problem.graph,
        problem.field,
        schedule.build(count),
        rng,
        move=(move,),
        cluster_backend=Backend.PYTHON
        if move is SandboxMove.NIEDERMAYER
        else Backend.RUST,
        start=start,
    )
    return MethodRun(
        labelling=run.best,
        energy=run.energy,
        spent=run.spent,
        seconds=time.perf_counter() - started,
        trace=run.trace,
        termination=Termination.after(count, converged=False),
    )
