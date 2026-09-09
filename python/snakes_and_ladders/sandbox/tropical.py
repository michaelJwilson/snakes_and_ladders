"""The tropical Grassmannian relaxation: tree topologies as a smooth objective (issue #408).

``ROADMAP.md`` Stage 3 asks for a continuous relaxation of the space of tree
topologies and records the half as blocked on an oracle. It is not, once
simulation truth and enumeration count as one: below nine taxa every unrooted
topology can be scored, so "does gradient ascent on the relaxation find what
discrete search finds" has an answer here rather than an opinion. This module
takes that half; :mod:`snakes_and_ladders.learn.relaxed` took the
Gumbel-softmax one and shares nothing with it.

**It is here because the answer was no.** Neighbor joining reaches the same
enumerated maximum at no gradient steps at 5, 6, 7 and 8 taxa, and the
differentiable part is 3.2% of the run, so the relaxation buys nothing on any
fixture measured and no search calls it. It is conserved rather than deleted,
with the two oracles that referee it, because a measurement whose subject was
removed is a claim with no way back to it (``sandbox/CLAUDE.md``). Only
``tests/`` and :mod:`snakes_and_ladders.qa` import it, and
:mod:`snakes_and_ladders.qa.tropical_relaxation` keeps reporting the figure
the paper cites.

**The coordinates are a dissimilarity, not a topology.** :cite:`speyer2004`
identify the tropical Grassmannian ``Gr(2, n)`` with the space of trees: a
point is a vector of ``n (n - 1) / 2`` pairwise distances, and it is the
metric of some tree exactly when every quartet satisfies the tropical
Plücker relation --- of the three pairings ``d_ij + d_kl``, ``d_ik + d_jl``,
``d_il + d_jk``, the largest is attained twice, which is the four-point
condition of ``sec:spectral-start`` (:func:`snakes_and_ladders.search.neighbor_joining.four_point_violation`
measures how far a vector is from it). A tree metric's *smallest* pairing is
then attained once, and which pairing that is is the quartet's topology. So
the discrete object --- a resolution per quartet --- is an ``argmin`` of a
linear function of the coordinates, and relaxing the ``argmin`` relaxes the
topology.

**The relaxation is a softmin, and the objective is linear in its output.**
Write ``l[Q, r]`` for the maximized log-likelihood of quartet ``Q`` under
resolution ``r``, a table of ``3 C(n, 4)`` numbers computed once by the same
fit :func:`snakes_and_ladders.search.infer.score_topology` performs
(:func:`quartet_table`). The discrete objective is
``D(T) = sum_Q l[Q, r_T(Q)]``, and the relaxed one replaces the hard
resolution by the softmin weights::

    F_tau(d) = sum_Q  <softmax(-s_Q(d) / tau),  l[Q, .]>

with ``s_Q(d)`` the three pairing sums (``eq:tropical-relaxation``). ``F`` is *linear* in the weights, so
at a one-hot weight vector it is ``D`` exactly --- the same "extension, not a
second model" the Gumbel-softmax half is built on, and the reason the corner
test below is an equality rather than a correlation.

**Two things it is not, and both matter.** ``D`` is not the tree's
log-likelihood: it is a quartet decomposition of it, a *different surface*,
and ``search/CLAUDE.md`` licenses substituting one for the other only by
measuring that they agree at the argmax. That measurement is
``tests/regression/test_sandbox_tropical.py``, at every size where
enumeration reaches. And ``F`` is optimized over the whole ambient
``R^C(n,2)``, not over the Grassmannian: a general dissimilarity resolves
each quartet, and the collection need not come from any tree. So ascent can
leave the tree locus, the four-point violation of where it lands is
reported, and the topology is read off by neighbor joining, which is exact on
the locus and is the repository's existing, refereed projection back onto it.

**The scale of ``d`` is a gauge and is fixed here.** ``F_tau(c d) =
F_{tau/c}(d)``: multiplying every distance by ``c > 0`` sharpens the softmin
without changing a single resolution, so an unconstrained ascent maximizes
``F`` by inflating the metric rather than by moving the topology --- the same
unidentifiability ``learn/CLAUDE.md`` names for a feature constant across a
state's actions. :func:`relaxed_score` therefore normalizes ``d`` to unit
mean, leaving ``tau`` as the only sharpness knob and annealing as the only
schedule.

**The corner tolerance is derived, not chosen.** With ``g_Q`` the gap
between a corner's smallest and second-smallest pairing sum and ``R_Q`` the
range of ``l[Q, .]``, the softmin puts at most ``2 exp(-g_Q / tau)`` weight
off the corner's own resolution, so

    |F_tau(d) - D(T)|  <=  sum_Q 2 exp(-g_Q / tau) R_Q,

which is ``eq:tropical-corner`` and :func:`corner_bound`; :func:`temperature_for` inverts it for the
``tau`` that buys a requested tolerance. The 1e-11 the Gumbel-softmax half
was held to is therefore a *measured* consequence of a fixture's minimum
quartet gap rather than a constant, and the same statement transfers to a
fixture with a shorter internal branch by returning a smaller ``tau``.

There is no sampled estimator here and so no estimator bias to measure:
``F_tau`` is a deterministic, closed-form function of ``d`` and its gradient
is exact by autodiff, checked against central differences. That is the
tropical analogue of :mod:`snakes_and_ladders.learn.relaxed`'s
``stochastic=False`` path, and it is licensed for the same reason --- nothing
is being approximated by sampling.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.likelihood.distance import tree_distances
from snakes_and_ladders.search.infer import Model, score_topology
from snakes_and_ladders.search.neighbor_joining import (
    four_point_violation,
    neighbor_joining,
)
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)
from snakes_and_ladders.sim.tree import Node

#: The three ways to split four taxa into two pairs, in the order
#: :func:`snakes_and_ladders.search.neighbor_joining.four_point_violation`
#: forms its sums: ``01|23``, ``02|13``, ``03|12``.
PAIRINGS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)

#: Fewest taxa with a quartet, and so the fewest this module accepts.
MIN_TAXA = 4

#: Below this the softmin saturates in float64 and the gradient underflows,
#: so a smaller temperature is refused rather than silently returning a
#: stalled ascent. The same bound and the same reason as
#: :data:`snakes_and_ladders.learn.relaxed.MINIMUM_TEMPERATURE`.
MINIMUM_TEMPERATURE = 1e-3


def quartet_indices(n: int) -> np.ndarray:
    """Every quartet of taxon indices, ascending.

    Parameters
    ----------
    n : int
        Taxon count, at least :data:`MIN_TAXA`.

    Returns
    -------
    np.ndarray
        Shape ``(C(n, 4), 4)``, ``int64``, each row sorted and the rows in
        lexicographic order.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_TAXA` taxa are given.
    """
    if n < MIN_TAXA:
        msg = f"a quartet needs at least {MIN_TAXA} taxa, got {n}"
        raise ValueError(msg)
    return np.array(list(itertools.combinations(range(n), 4)), dtype=np.int64)


def _pair_position(n: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Where pair ``(i, j)``, ``i < j``, sits in the condensed vector of ``n`` taxa.

    The layout is :func:`scipy.spatial.distance.squareform`'s, so the same
    vector converts to a square matrix for neighbor joining without a second
    convention to keep true.
    """
    return np.asarray((first * (2 * n - first - 1)) // 2 + second - first - 1)


def pairing_positions(n: int) -> np.ndarray:
    """Condensed-vector positions of the six pairs of every quartet.

    Returns
    -------
    np.ndarray
        Shape ``(C(n, 4), 3, 2)``, ``int64``: for each quartet and each of
        :data:`PAIRINGS`, the two positions whose distances are summed.
    """
    quartets = quartet_indices(n)
    positions = np.empty((quartets.shape[0], 3, 2), dtype=np.int64)
    for pairing, ((a, b), (c, d)) in enumerate(PAIRINGS):
        for slot, (left, right) in enumerate(((a, b), (c, d))):
            positions[:, pairing, slot] = _pair_position(
                n, quartets[:, left], quartets[:, right]
            )
    return positions


@dataclass(frozen=True)
class QuartetTable:
    """A score per quartet per resolution, and the taxa it is indexed by.

    The relaxation's whole dependence on the data. It costs ``3 C(n, 4)``
    four-taxon fits to build and nothing to evaluate, which is what makes
    enumerating every topology's discrete score affordable at eight taxa
    where enumerating every topology's *likelihood* is not.

    Parameters
    ----------
    names : tuple[str, ...]
        The taxa, sorted, indexing :attr:`quartets`.
    quartets : np.ndarray
        Shape ``(m, 4)``, from :func:`quartet_indices`.
    scores : np.ndarray
        Shape ``(m, 3)``: the maximized log-likelihood of each quartet under
        each of :data:`PAIRINGS`.
    """

    names: tuple[str, ...]
    quartets: np.ndarray
    scores: np.ndarray

    @property
    def n_taxa(self) -> int:
        """Taxa the table covers."""
        return len(self.names)

    @property
    def n_quartets(self) -> int:
        """Quartets in the table."""
        return int(self.quartets.shape[0])

    @property
    def ranges(self) -> np.ndarray:
        """Per quartet, the spread of its three scores.

        The weight a misplaced quartet can cost, and the ``R_Q`` of
        :func:`corner_bound`.
        """
        return np.asarray(self.scores.max(axis=1) - self.scores.min(axis=1))


def quartet_table(
    alignment: Mapping[str, np.ndarray], k: int, *, model: Model = Model.JC
) -> QuartetTable:
    """Fit all three resolutions of every quartet.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, at least :data:`MIN_TAXA` of them.
    k : int
        Number of states.
    model : Model
        Substitution model for each four-taxon fit.

    Returns
    -------
    QuartetTable
        The table, taxa in sorted order.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_TAXA` taxa are given.
    """
    names = tuple(sorted(alignment))
    quartets = quartet_indices(len(names))
    scores = np.empty((quartets.shape[0], 3), dtype=np.float64)
    single = np.array([[0, 1, 2, 3]], dtype=np.int64)
    for row, quartet in enumerate(quartets):
        members = [names[index] for index in quartet]
        restricted = {name: alignment[name] for name in members}
        for topology in enumerate_topologies(members):
            resolution = int(resolutions(single, members, topology)[0])
            scores[row, resolution] = score_topology(
                topology, restricted, k, model=model
            )
    return QuartetTable(names=names, quartets=quartets, scores=scores)


#: A split of the leaf set separates a quartet ``(q0, q1, q2, q3)`` in one of
#: three ways, read as the four bits ``q0 q1 q2 q3`` of its membership: a
#: split and its complement give the two codes listed for each of
#: :data:`PAIRINGS`. Every other code is a split that does not separate the
#: quartet into two pairs and says nothing about it.
_PAIRING_CODES: tuple[tuple[int, int], ...] = ((12, 3), (10, 5), (9, 6))

_CODE_WEIGHTS = np.array([8, 4, 2, 1], dtype=np.int64)


def _split_masks(topology: Topology, names: Sequence[str]) -> np.ndarray:
    """The topology's leaf bipartitions as bit masks over ``names``."""
    position = {name: index for index, name in enumerate(names)}
    return np.array(
        [
            sum(1 << position[name] for name in split)
            for split in leaf_bipartitions(topology)
        ],
        dtype=np.int64,
    )


def resolutions(
    quartets: np.ndarray, names: Sequence[str], topology: Topology
) -> np.ndarray:
    """Which of :data:`PAIRINGS` a topology induces on each quartet.

    Read from the leaf bipartitions, so this is combinatorics over the
    topology and involves no metric and no float --- the ``argmin`` of the
    tropical Plücker relation is the *other* route to the same answer, and a
    test pins the two against each other on tree metrics.

    Parameters
    ----------
    quartets : np.ndarray
        Shape ``(m, 4)`` of indices into ``names``.
    names : Sequence[str]
        The taxa, in the order ``quartets`` indexes.
    topology : Topology
        An unrooted binary topology on exactly those taxa.

    Returns
    -------
    np.ndarray
        Shape ``(m,)``, ``int64``, entries in ``{0, 1, 2}``.

    Raises
    ------
    ValueError
        If the topology leaves a quartet unresolved, which a binary topology
        on the same leaf set cannot.
    """
    masks = _split_masks(topology, names)
    bits = (masks[:, np.newaxis, np.newaxis] >> quartets[np.newaxis, :, :]) & 1
    codes = bits @ _CODE_WEIGHTS
    found = np.full(quartets.shape[0], -1, dtype=np.int64)
    for pairing, allowed in enumerate(_PAIRING_CODES):
        hit = np.isin(codes, allowed).any(axis=0)
        found = np.where((found < 0) & hit, pairing, found)
    if bool((found < 0).any()):
        unresolved = quartets[found < 0][0]
        msg = f"topology leaves the quartet {tuple(unresolved)} unresolved"
        raise ValueError(msg)
    return found


def discrete_score(table: QuartetTable, topology: Topology) -> float:
    """``D(T)``: the topology's quartet score, summed over quartets.

    Parameters
    ----------
    table : QuartetTable
        The fitted quartet scores.
    topology : Topology
        An unrooted binary topology on the table's taxa.

    Returns
    -------
    float
        The sum of each quartet's score under the resolution ``topology``
        induces.
    """
    chosen = resolutions(table.quartets, table.names, topology)
    return float(table.scores[np.arange(table.n_quartets), chosen].sum())


def corner(topology: Topology, branch_length: float = 1.0) -> np.ndarray:
    """A tree metric of ``topology``: the relaxation's corner for that tree.

    The analogue of :func:`snakes_and_ladders.learn.relaxed.one_hot`. Every
    branch is given the same length, which is a tree metric like any other
    and puts every quartet's margin at ``2 * branch_length`` before the
    unit-mean normalization --- the widest margin the topology admits, so a
    corner test run here is testing the softmin rather than a fixture's
    shortest internal branch. The metric of a *fitted* tree is
    :func:`snakes_and_ladders.likelihood.distance.tree_distances` instead,
    and the two are corners of the same relaxation.

    Parameters
    ----------
    topology : Topology
        An unrooted binary topology.
    branch_length : float
        The common length, positive.

    Returns
    -------
    np.ndarray
        The condensed metric, taxa in sorted order.

    Raises
    ------
    ValueError
        If ``branch_length`` is not positive.
    """
    if branch_length <= 0.0:
        msg = f"branch_length must be positive, got {branch_length}"
        raise ValueError(msg)

    def relabel(node: Topology) -> Topology:
        return Node(
            name=node.name,
            branch_length=branch_length,
            children=tuple(relabel(child) for child in node.children),
        )

    rooted = Node(
        name=topology.name,
        branch_length=None,
        children=tuple(relabel(child) for child in topology.children),
    )
    _, square = tree_distances(rooted)
    return condensed(square)


def condensed(distances: np.ndarray) -> np.ndarray:
    """A square distance matrix as the condensed vector this module optimizes.

    The row-major upper triangle, which is what :func:`_pair_position`
    indexes and what :func:`expanded` inverts. Hand-rolled rather than taken
    from ``scipy``: no module of the package imports it, and the two
    directions here are four lines of indexing that :func:`expanded`'s round
    trip pins.

    Parameters
    ----------
    distances : np.ndarray
        Symmetric ``(n, n)`` with a zero diagonal.

    Returns
    -------
    np.ndarray
        Length ``n (n - 1) / 2``.
    """
    size = distances.shape[0]
    return np.asarray(distances[np.triu_indices(size, 1)], dtype=np.float64)


def expanded(distances: np.ndarray, n: int) -> np.ndarray:
    """A condensed vector back as a symmetric ``(n, n)`` matrix with a zero diagonal.

    Parameters
    ----------
    distances : np.ndarray
        Length ``n (n - 1) / 2``, in :func:`condensed`'s layout.
    n : int
        Taxon count.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)``.
    """
    matrix = np.zeros((n, n), dtype=np.float64)
    rows, columns = np.triu_indices(n, 1)
    matrix[rows, columns] = distances
    matrix[columns, rows] = distances
    return matrix


def _pairing_sums(positions: np.ndarray, distances: torch.Tensor) -> torch.Tensor:
    """The three pairing sums of every quartet, shape ``(m, 3)``."""
    return distances[positions[..., 0]] + distances[positions[..., 1]]


def relaxed_score(
    table: QuartetTable,
    positions: np.ndarray,
    distances: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """``F_tau(d)``: the quartet score under softmin weights, differentiable in ``d``.

    ``d`` is normalized to unit mean first, for the reason the module
    docstring gives: the scale is a gauge the objective cannot fix, and
    without the normalization ascent inflates the metric instead of moving
    the topology.

    Parameters
    ----------
    table : QuartetTable
        The fitted quartet scores.
    positions : np.ndarray
        From :func:`pairing_positions` at the table's taxon count, passed in
        rather than rebuilt because ascent calls this once per step.
    distances : torch.Tensor
        Condensed, length ``n (n - 1) / 2``, positive, ``float64``.
    temperature : float
        ``tau``, at least :data:`MINIMUM_TEMPERATURE`.

    Returns
    -------
    torch.Tensor
        A scalar.

    Raises
    ------
    ValueError
        If ``temperature`` is below :data:`MINIMUM_TEMPERATURE`.
    """
    if temperature < MINIMUM_TEMPERATURE:
        msg = (
            f"temperature must be >= {MINIMUM_TEMPERATURE}, got {temperature}: "
            "below it the softmin saturates and the gradient underflows"
        )
        raise ValueError(msg)
    normalized = distances / distances.mean()
    sums = _pairing_sums(positions, normalized)
    weights = torch.softmax(-sums / temperature, dim=1)
    scores = torch.from_numpy(table.scores)
    return (weights * scores).sum()


def corner_gaps(positions: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Per quartet, the gap between the smallest pairing sum and the next.

    On a tree metric the smallest sum is the quartet's own resolution and
    the other two are equal, so this is the margin the softmin has to resolve
    and the ``g_Q`` of :func:`corner_bound`. Computed on the unit-mean
    normalization :func:`relaxed_score` uses, so the two agree about scale.

    Parameters
    ----------
    positions : np.ndarray
        From :func:`pairing_positions`.
    distances : np.ndarray
        Condensed, positive.

    Returns
    -------
    np.ndarray
        Shape ``(m,)``, non-negative.
    """
    normalized = distances / distances.mean()
    sums = np.sort(normalized[positions[..., 0]] + normalized[positions[..., 1]])
    return np.asarray(sums[:, 1] - sums[:, 0])


def corner_bound(
    table: QuartetTable,
    positions: np.ndarray,
    distances: np.ndarray,
    temperature: float,
) -> float:
    """How far ``F_tau`` may sit from ``D`` at a tree metric.

    ``sum_Q 2 exp(-g_Q / tau) R_Q``: the softmin leaves at most
    ``2 exp(-g_Q / tau)`` weight off the quartet's own resolution and a
    misplaced unit of weight costs at most the spread of that quartet's three
    scores. An upper bound at every ``tau``, not an estimate.

    Parameters
    ----------
    table : QuartetTable
        Supplies ``R_Q`` through :attr:`QuartetTable.ranges`.
    positions : np.ndarray
        From :func:`pairing_positions`.
    distances : np.ndarray
        A tree metric, condensed.
    temperature : float
        ``tau``.

    Returns
    -------
    float
        The bound, non-negative.
    """
    gaps = corner_gaps(positions, distances)
    return float((2.0 * np.exp(-gaps / temperature) * table.ranges).sum())


def temperature_for(
    table: QuartetTable,
    positions: np.ndarray,
    distances: np.ndarray,
    tolerance: float,
) -> float:
    """The largest ``tau`` :func:`corner_bound` certifies against ``tolerance``.

    Inverts the bound at its smallest gap:
    ``tau = min_Q g_Q / log(2 m R_max / tolerance)``. Sufficient rather than
    tight --- every quartet is charged the worst quartet's range and the
    smallest quartet's gap --- which is the direction a tolerance may err in.

    Parameters
    ----------
    table : QuartetTable
        Supplies the score ranges.
    positions : np.ndarray
        From :func:`pairing_positions`.
    distances : np.ndarray
        A tree metric, condensed.
    tolerance : float
        The corner agreement wanted, positive.

    Returns
    -------
    float
        A temperature, at least :data:`MINIMUM_TEMPERATURE`.

    Raises
    ------
    ValueError
        If ``tolerance`` is not positive, or the metric has a zero quartet
        gap --- a degenerate quartet, which no temperature resolves.
    """
    if tolerance <= 0.0:
        msg = f"tolerance must be positive, got {tolerance}"
        raise ValueError(msg)
    gaps = corner_gaps(positions, distances)
    smallest = float(gaps.min())
    if smallest <= 0.0:
        msg = (
            "the metric has a quartet whose two smallest pairing sums are equal; "
            "no temperature resolves it"
        )
        raise ValueError(msg)
    budget = 2.0 * table.n_quartets * float(table.ranges.max())
    return max(smallest / float(np.log(budget / tolerance)), MINIMUM_TEMPERATURE)


@dataclass(frozen=True)
class TropicalOptimum:
    """What gradient ascent on ``F_tau`` reached.

    Parameters
    ----------
    topology : Topology
        Read off the final metric by neighbor joining, which is exact on the
        tree locus.
    score : float
        Its *discrete* score ``D``, which is the number to compare against an
        enumerated maximum. The relaxed value at ``tau > 0`` is attained by no
        topology and comparing it would flatter the method.
    relaxed_score : float
        ``F_tau`` at the final metric and final temperature.
    plucker_violation : float
        The largest four-point gap of the final metric, from
        :func:`snakes_and_ladders.search.neighbor_joining.four_point_violation`.
        Zero says ascent stayed on the tropical Grassmannian; anything else
        says how far off it the answer had to be projected.
    steps : int
        Gradient steps taken.
    """

    topology: Topology
    score: float
    relaxed_score: float
    plucker_violation: float
    steps: int


def anneal(start: float, end: float, steps: int, step: int) -> float:
    """Geometric temperature schedule, evaluated at one step.

    Geometric for the reason :func:`snakes_and_ladders.learn.relaxed.anneal`
    is: the softmin's behaviour is set by the *ratio* of quartet gaps to
    ``tau``, so equal multiplicative steps are equal steps in what matters.

    Parameters
    ----------
    start, end : float
        Endpoints, both at least :data:`MINIMUM_TEMPERATURE`, ``end`` no
        larger than ``start``.
    steps : int
        Total steps, at least 1.
    step : int
        Which step, clamped to the last.

    Returns
    -------
    float
        The temperature at ``step``.

    Raises
    ------
    ValueError
        If either endpoint is below :data:`MINIMUM_TEMPERATURE`, ``end``
        exceeds ``start``, or ``steps`` is below 1.
    """
    if min(start, end) < MINIMUM_TEMPERATURE:
        msg = f"both endpoints must be >= {MINIMUM_TEMPERATURE}, got ({start}, {end})"
        raise ValueError(msg)
    if end > start:
        msg = f"end must not exceed start, got start={start}, end={end}"
        raise ValueError(msg)
    if steps < 1:
        msg = f"steps must be at least 1, got {steps}"
        raise ValueError(msg)
    if steps == 1:
        return start
    fraction = min(step, steps - 1) / (steps - 1)
    return float(start * (end / start) ** fraction)


def optimize(
    table: QuartetTable,
    distances: np.ndarray,
    *,
    temperature: float = 0.05,
    final_temperature: float | None = None,
    steps: int = 300,
    learning_rate: float = 0.02,
) -> TropicalOptimum:
    """Gradient ascent on ``F_tau`` from a starting metric, then read the tree off.

    Ascent is on the logarithm of each distance, so the metric stays positive
    without a projection and a step is multiplicative --- which matches the
    gauge :func:`relaxed_score` normalizes away, where an additive step would
    not.

    Parameters
    ----------
    table : QuartetTable
        The fitted quartet scores.
    distances : np.ndarray
        Condensed starting metric, positive. There is no default and no
        internal random start: an estimated distance matrix is what a caller
        already has, and a metric drawn here would be either unseeded or
        seeded from a constant nobody declared.
    temperature : float
        Fixed ``tau``, or the starting ``tau`` when ``final_temperature`` is
        given.
    final_temperature : float | None
        ``None`` holds ``temperature`` fixed; a value anneals geometrically
        to it over ``steps``.
    steps, learning_rate : int, float
        Adam budget.

    Returns
    -------
    TropicalOptimum
        Carrying the discrete score of the recovered topology and the
        four-point violation of the metric it was read from.

    Raises
    ------
    ValueError
        If any starting distance is not positive.
    """
    if distances.min() <= 0.0:
        msg = "every starting distance must be positive; ascent is on their logarithm"
        raise ValueError(msg)
    positions = pairing_positions(table.n_taxa)
    logarithms = torch.log(torch.from_numpy(np.asarray(distances, dtype=np.float64)))
    logarithms.requires_grad_(True)
    optimizer = torch.optim.Adam([logarithms], lr=learning_rate)

    current = temperature
    for step in range(steps):
        current = (
            temperature
            if final_temperature is None
            else anneal(temperature, final_temperature, steps, step)
        )
        optimizer.zero_grad()
        loss = -relaxed_score(table, positions, torch.exp(logarithms), current)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    with torch.no_grad():
        final = torch.exp(logarithms)
        value = float(relaxed_score(table, positions, final, current))
        metric = final.numpy()
    square = expanded(metric, table.n_taxa)
    topology = neighbor_joining(list(table.names), square)
    return TropicalOptimum(
        topology=topology,
        score=discrete_score(table, topology),
        relaxed_score=value,
        plucker_violation=four_point_violation(square),
        steps=steps,
    )


def metric_from_split_weights(
    weights: Mapping[frozenset[str], float], names: Sequence[str]
) -> np.ndarray:
    """The tree metric a set of weighted splits implies, as a square matrix.

    Two taxa are separated by exactly the splits with one of them on each
    side, so the path length between them is the sum of those splits'
    weights. Reading a metric this way rather than from a built tree is what
    lets the Hadamard conjugation of ``eq:hadamard`` referee this module: its
    output is a weight per split, including the ``2^(n-1) - n(n-1)/2`` splits
    a tree does not have, and every one of them enters here.

    Parameters
    ----------
    weights : Mapping[frozenset[str], float]
        Split to weight, as
        :func:`snakes_and_ladders.likelihood.hadamard.split_weights` returns.
        A split is keyed by one of its two sides.
    names : Sequence[str]
        The taxa, in the order the returned matrix is indexed by.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` with a zero diagonal.
    """
    order = list(names)
    size = len(order)
    matrix = np.zeros((size, size), dtype=np.float64)
    for split, weight in weights.items():
        for row in range(size):
            for column in range(row + 1, size):
                if (order[row] in split) != (order[column] in split):
                    matrix[row, column] += weight
                    matrix[column, row] += weight
    return matrix
