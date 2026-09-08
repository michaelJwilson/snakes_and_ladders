"""How sure a discrete search is of the structure it returns (issues #270, #331).

A continuous fit reports an interval; a structure has no Hessian, so a search
that returns a topology and nothing else leaves the tree it found
indistinguishable from one it could equally have returned. Four quantities
are stated here, each as what it is and never as another:

* **neighbourhood support**: with a flat prior over structures, the weight
  of the returned one among itself and its neighbours under a move set,
  ``exp(l*) / sum exp(l)`` over the log scores, with the **margin**
  ``l* - max l_neighbour`` beside it -- which says whether one more move
  could have changed the answer. A neighbourhood quantity, exact only where
  the neighbourhood is the whole space;
* **enumerated support**: the same weight over every structure, where the
  enumeration fits -- the exact flat-prior posterior, and the oracle the
  other quantities are compared to;
* **tempered support**: the fraction of a tempered ensemble's
  temperature-one replica spent at the structure
  (:mod:`snakes_and_ladders.search.tempered`; ``docs/tex/textbook.tex``,
  ``eq:tempered-weight``): a Monte Carlo estimate of the enumerated weight
  that reaches where enumeration is refused, named a posterior weight only
  beside the exchange acceptance and autocorrelation time that show the
  ensemble mixed;
* **bootstrap support** (Felsenstein, 1985): resample sites with
  replacement, search again, and report per split the fraction of replicates
  containing it. The field's convention; its relation to the three above on
  simulated data is a finding.

The first three serve every problem class through one score. For a
**topology** the score is a *maximized* log-likelihood, not a marginal one,
so the weight is the posterior under a flat prior over topologies with
branch lengths at their optimum, and is named so rather than called a
posterior. For a **labelling** of a factor graph -- a Potts configuration,
or a hidden path, which is a labelling of the chain's graph -- the score is
the unnormalized log-density at temperature one, so the enumerated weight is
the Boltzmann weight :func:`snakes_and_ladders.likelihood.potts.enumerate_potts`
normalizes at ``beta = 1`` and the path posterior
:func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths` sums,
and the neighbourhood is every single-site change. The bootstrap is a tree
quantity alone: resampling sites is licensed by their exchangeability given
the tree, which a chain's ordered sites do not have and a labelling has no
sites to offer. Which of the four a result reports is part of the result
(``search/CLAUDE.md``).
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    refuse_oversized,
)
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.search.infer import Model, MoveSet, infer, score_topology
from snakes_and_ladders.search.statistics import integrated_autocorrelation_time
from snakes_and_ladders.search.tempered import TemperedEnsemble
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
    spr_neighbours,
)
from snakes_and_ladders.sim.factor_graph import FactorGraph


class SupportKind(StrEnum):
    """Which set the weight is taken over, and how."""

    NEIGHBOURHOOD = "neighbourhood"
    ENUMERATED = "enumerated"
    TEMPERED = "tempered"


@dataclass(frozen=True)
class Support:
    """What a discrete search reports beside its structure.

    Parameters
    ----------
    log_score : float
        The returned structure's log score: a topology's maximized
        log-likelihood, a labelling's unnormalized log-density.
    margin : float
        ``log_score`` minus the best competing structure's; positive when
        the returned structure beats every competitor in the set.
    weight : float
        ``exp(log_score)`` over the sum across the set, the returned
        structure included: the flat-prior weight. For
        :attr:`SupportKind.TEMPERED` a Monte Carlo estimate of that weight
        over the whole space, the fraction of recorded sweeps at the
        structure.
    n_candidates : int
        The size of the set, the returned structure included. For
        :attr:`SupportKind.TEMPERED` the structures the ensemble visited,
        which is what the margin is over.
    kind : SupportKind
        Whether the set is a neighbourhood, every structure, or the visits
        of a tempered ensemble.
    """

    log_score: float
    margin: float
    weight: float
    n_candidates: int
    kind: SupportKind


@dataclass(frozen=True)
class TemperedSupport(Support):
    """A tempered weight with the diagnostics that decide whether to believe it.

    Parameters
    ----------
    swap_acceptance : tuple[float, ...]
        The ensemble's exchange acceptance per adjacent pair.
    autocorrelation_time : float
        Of the indicator "the temperature-one replica is at the structure",
        in recorded sweeps (Sokal's window). ``n_recorded`` over it is the
        number of independent draws behind ``weight``.
    n_recorded : int
        Sweeps recorded at temperature one.
    """

    swap_acceptance: tuple[float, ...]
    autocorrelation_time: float
    n_recorded: int


def _distinct(topology: Topology, candidates: Iterator[Topology]) -> list[Topology]:
    """``candidates`` with duplicates and the topology itself removed, by bipartitions."""
    own = leaf_bipartitions(topology)
    seen = {own}
    distinct = []
    for candidate in candidates:
        key = leaf_bipartitions(candidate)
        if key not in seen:
            seen.add(key)
            distinct.append(candidate)
    return distinct


def _support(log_score: float, competitors: list[float], kind: SupportKind) -> Support:
    if not competitors:
        return Support(log_score, np.inf, 1.0, 1, kind)
    scores = np.array([log_score, *competitors])
    total = float(logsumexp(scores[None, :], axis=1)[0])
    return Support(
        log_score=log_score,
        margin=log_score - max(competitors),
        weight=float(np.exp(log_score - total)),
        n_candidates=len(scores),
        kind=kind,
    )


def neighbourhood_support(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    *,
    moves: MoveSet = MoveSet.NNI,
    model: Model = Model.JC,
) -> Support:
    """The flat-prior weight of ``topology`` among itself and its neighbours under ``moves``.

    Every neighbour is fitted, so the cost is the neighbourhood size in fits.
    Exact where the neighbourhood is every other topology (four taxa under
    NNI) and a neighbourhood quantity otherwise.
    """
    neighbours = nni_neighbours if moves is MoveSet.NNI else spr_neighbours
    competitors = [
        score_topology(candidate, alignment, k, model)
        for candidate in _distinct(topology, neighbours(topology))
    ]
    return _support(
        score_topology(topology, alignment, k, model),
        competitors,
        SupportKind.NEIGHBOURHOOD,
    )


def _double_factorial_count(n_leaves: int) -> int:
    count = 1
    for factor in range(2 * n_leaves - 5, 0, -2):
        count *= factor
    return count


def enumerated_support(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    *,
    model: Model = Model.JC,
    max_topologies: int = 945,
) -> Support:
    """The flat-prior weight of ``topology`` over every topology on its leaves.

    The exact quantity, and the oracle :func:`neighbourhood_support` is held
    to. Every topology is fitted, so it is refused past ``max_topologies``
    (``945`` is seven taxa).

    Raises
    ------
    ValueError
        Past ``max_topologies``.
    """
    leaves = sorted(alignment)
    refuse_oversized(
        _double_factorial_count(len(leaves)),
        what=f"topologies on {len(leaves)} taxa",
        limit=max_topologies,
    )
    competitors = [
        score_topology(candidate, alignment, k, model)
        for candidate in _distinct(topology, enumerate_topologies(leaves))
    ]
    return _support(
        score_topology(topology, alignment, k, model),
        competitors,
        SupportKind.ENUMERATED,
    )


def _check_labelling(graph: FactorGraph, labelling: np.ndarray) -> np.ndarray:
    state = np.asarray(labelling, dtype=np.int64)
    cardinality = np.array([variable.cardinality for variable in graph.variables])
    if (
        state.shape != cardinality.shape
        or (state < 0).any()
        or (state >= cardinality).any()
    ):
        msg = f"a labelling holds one state per variable inside its cardinality, got {state}"
        raise ValueError(msg)
    return state


def _log_density(graph: FactorGraph, state: np.ndarray) -> float:
    names = [variable.name for variable in graph.variables]
    return graph.log_density(dict(zip(names, map(int, state), strict=True)))


def neighbourhood_labelling_support(
    graph: FactorGraph, labelling: np.ndarray
) -> Support:
    """The flat-prior weight of ``labelling`` among itself and every single-site change.

    A single-site flip of a Potts configuration and a single-state change
    of a hidden path are the same move on the factor graph: one variable
    set to one of its other states, ``sum (cardinality - 1)`` neighbours.
    Exact where that is the whole space and a neighbourhood quantity
    otherwise; a labelling a hard factor forbids scores ``-inf`` and carries
    no weight.

    Raises
    ------
    ValueError
        If ``labelling`` is not one state per variable inside its domain.
    """
    state = _check_labelling(graph, labelling)
    competitors = []
    for position, variable in enumerate(graph.variables):
        for other in range(variable.cardinality):
            if other == int(state[position]):
                continue
            neighbour = state.copy()
            neighbour[position] = other
            competitors.append(_log_density(graph, neighbour))
    return _support(_log_density(graph, state), competitors, SupportKind.NEIGHBOURHOOD)


def enumerated_labelling_support(
    graph: FactorGraph,
    labelling: np.ndarray,
    *,
    max_configurations: int = MAX_ENUMERABLE_CONFIGURATIONS,
) -> Support:
    """The flat-prior weight of ``labelling`` over every labelling of ``graph``.

    ``exp(log_density) / Z``: the Boltzmann weight at ``beta = 1``, or the
    path posterior of a decoding, and the oracle the neighbourhood and
    tempered weights are held to. Refused past ``max_configurations``
    labellings, on :func:`snakes_and_ladders.enumeration.refuse_oversized`'s
    terms.

    Raises
    ------
    ValueError
        Past ``max_configurations``, or if ``labelling`` is unusable.
    """
    state = _check_labelling(graph, labelling)
    cardinalities = [variable.cardinality for variable in graph.variables]
    count = int(np.prod(cardinalities))
    refuse_oversized(
        count,
        what=f"{count} labellings of {len(cardinalities)} variables",
        limit=max_configurations,
    )
    own = tuple(int(value) for value in state)
    competitors = [
        _log_density(graph, np.array(candidate, dtype=np.int64))
        for candidate in itertools.product(*(range(card) for card in cardinalities))
        if candidate != own
    ]
    return _support(_log_density(graph, state), competitors, SupportKind.ENUMERATED)


def _tempered(
    ensemble: TemperedEnsemble, key: Hashable, log_score: float
) -> TemperedSupport:
    replica = ensemble.replica_at(1.0)
    visits = ensemble.keys[replica]
    indicator = np.array([name == key for name in visits], dtype=float)
    competitors = [value for name, value in ensemble.scores.items() if name != key]
    return TemperedSupport(
        log_score=log_score,
        margin=log_score - max(competitors) if competitors else np.inf,
        weight=float(indicator.mean()),
        n_candidates=len(competitors) + 1,
        kind=SupportKind.TEMPERED,
        swap_acceptance=tuple(float(value) for value in ensemble.swap_acceptance),
        autocorrelation_time=integrated_autocorrelation_time(indicator),
        n_recorded=len(visits),
    )


def tempered_labelling_support(
    graph: FactorGraph, labelling: np.ndarray, ensemble: TemperedEnsemble
) -> TemperedSupport:
    """The fraction of ``ensemble``'s temperature-one replica spent at ``labelling``.

    ``ensemble`` is a run of
    :func:`snakes_and_ladders.search.tempered.tempered_factor_graph` on
    ``graph``; the estimate is
    :func:`enumerated_labelling_support`'s weight where that is affordable,
    and the diagnostics say whether to believe it where it is not.

    Raises
    ------
    ValueError
        If no replica sits at temperature one, or ``labelling`` is unusable.
    """
    state = _check_labelling(graph, labelling)
    key = tuple(int(value) for value in state)
    return _tempered(ensemble, key, _log_density(graph, state))


def tempered_topology_support(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    ensemble: TemperedEnsemble,
    *,
    model: Model = Model.JC,
) -> TemperedSupport:
    """The fraction of ``ensemble``'s temperature-one replica spent at ``topology``.

    ``ensemble`` is a run of
    :func:`snakes_and_ladders.search.tempered.tempered_topologies` on
    ``alignment``; the estimate is :func:`enumerated_support`'s weight where
    ``(2n-5)!!`` fits, and reaches where it does not. A topology the
    ensemble never visited is fitted here, once.

    Raises
    ------
    ValueError
        If no replica sits at temperature one.
    """
    key = leaf_bipartitions(topology)
    log_score = ensemble.scores.get(key)
    if log_score is None:
        log_score = score_topology(topology, alignment, k, model)
    return _tempered(ensemble, key, log_score)


def internal_splits(topology: Topology) -> frozenset[frozenset[str]]:
    """The bipartitions with at least two leaves on each side: what a bootstrap counts."""
    splits = leaf_bipartitions(topology)
    leaves: set[str] = set()
    for split in splits:
        leaves.update(split)
    n_leaves = len(leaves) + 1  # the anchor leaf is never on a canonical side
    return frozenset(split for split in splits if 2 <= len(split) <= n_leaves - 2)


def bootstrap_support(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    rng: np.random.Generator,
    n_replicates: int,
    *,
    moves: MoveSet = MoveSet.NNI,
    model: Model = Model.JC,
    max_evaluations: int = 200,
) -> dict[frozenset[str], float]:
    """Felsenstein's bootstrap: per internal split of ``topology``, the fraction of replicates whose search returns it.

    Each replicate resamples the sites with replacement from ``rng``, then
    searches from ``topology`` under ``moves`` within ``max_evaluations``
    candidates, the search's own budget unit. The support of a split is a
    frequency over replicates and nothing more; whether it tracks the
    flat-prior weight is measured, not assumed.

    Raises
    ------
    ValueError
        If ``n_replicates`` is not positive.
    """
    if n_replicates < 1:
        msg = f"a bootstrap needs at least one replicate, got {n_replicates}"
        raise ValueError(msg)
    n_sites = next(iter(alignment.values())).shape[0]
    counts: dict[frozenset[str], int] = dict.fromkeys(internal_splits(topology), 0)
    for _ in range(n_replicates):
        columns = rng.integers(0, n_sites, size=n_sites)
        resampled = {name: states[columns] for name, states in alignment.items()}
        found = infer(
            resampled,
            k,
            topology=None,
            model=model,
            moves=moves,
            max_evaluations=max_evaluations,
            rng=rng,
        ).topology
        for split in internal_splits(found):
            if split in counts:
                counts[split] += 1
    return {split: count / n_replicates for split, count in counts.items()}
