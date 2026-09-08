"""How sure a discrete search is of the structure it returns (issue #270).

A continuous fit reports an interval; a structure has no Hessian, so a search
that returns a topology and nothing else leaves the tree it found
indistinguishable from one it could equally have returned. Three quantities
are stated here, each as what it is and never as another:

* **neighbourhood support**: with a flat prior over topologies, the weight
  of the returned tree among itself and its neighbours under a move set,
  ``exp(l*) / sum exp(l)`` over the fitted log-likelihoods, with the
  **margin** ``l* - max l_neighbour`` beside it -- which says whether one
  more move could have changed the answer. A neighbourhood quantity, exact
  only where the neighbourhood is the whole space;
* **enumerated support**: the same weight over every topology, where
  ``(2n-5)!!`` fits -- the exact flat-prior posterior over maximized
  likelihoods, and the oracle the neighbourhood quantity is compared to;
* **bootstrap support** (Felsenstein, 1985): resample sites with
  replacement, search again, and report per split the fraction of replicates
  containing it. The field's convention; its relation to the two above on
  simulated data is a finding;
* **pattern support**: per split, the fraction of sites compatible with it
  --- a character and a split are compatible when at most one state occurs
  on both sides (Felsenstein, *Inferring Phylogenies*, ch. 8). No fit and no
  resample, so it is cheap enough to be a feature of a move (issue #328);
  what it says about the bootstrap frequency is measured, not assumed.

Each ``l`` is a *maximized* log-likelihood, not a marginal one, so the weight
is the posterior under a flat prior over topologies with branch lengths at
their optimum, and is named so rather than called a posterior. Which of the
three a result reports is part of the result (``search/CLAUDE.md``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.parallel import Backend, map_tasks
from snakes_and_ladders.search.infer import Model, MoveSet, infer, score_topology
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
    spr_neighbours,
)

# How bootstrap replicates run beside each other: processes, because a
# replicate is a whole search -- Python control flow around small torch fits.
# The intra-op thread count is left at the process default in workers and
# serial alike, so the two runs reduce in the same order on one machine. No
# pool reached 2x at 4 workers at the mid-size tier; STATUS.md carries the
# measurement (issue #344).
_BOOTSTRAP_BACKEND: Backend = "processes"
_BOOTSTRAP_INTRA_OP_THREADS: int | None = None


class SupportKind(StrEnum):
    """Which set the weight is taken over."""

    NEIGHBOURHOOD = "neighbourhood"
    ENUMERATED = "enumerated"


@dataclass(frozen=True)
class Support:
    """What a discrete search reports beside its structure.

    Parameters
    ----------
    log_score : float
        The returned topology's maximized log-likelihood.
    margin : float
        ``log_score`` minus the best competing topology's; positive when the
        returned topology beats every competitor in the set.
    weight : float
        ``exp(log_score)`` over the sum across the set, the returned topology
        included: the flat-prior weight over maximized likelihoods.
    n_candidates : int
        The size of the set, the returned topology included.
    kind : SupportKind
        Whether the set is a neighbourhood or every topology.
    """

    log_score: float
    margin: float
    weight: float
    n_candidates: int
    kind: SupportKind


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


def internal_splits(topology: Topology) -> frozenset[frozenset[str]]:
    """The bipartitions with at least two leaves on each side: what a bootstrap counts."""
    splits = leaf_bipartitions(topology)
    leaves: set[str] = set()
    for split in splits:
        leaves.update(split)
    n_leaves = len(leaves) + 1  # the anchor leaf is never on a canonical side
    return frozenset(split for split in splits if 2 <= len(split) <= n_leaves - 2)


def _replicate(
    task: tuple[Mapping[str, np.ndarray], int, Model, MoveSet, int],
    rng: np.random.Generator,
) -> frozenset[frozenset[str]]:
    """One bootstrap replicate, importable so a process pool can run it.

    Resamples the sites with replacement from ``rng`` and searches the
    resample from a random start drawn from the same ``rng``, returning the
    internal splits of the topology the search returns.
    """
    alignment, k, model, moves, max_evaluations = task
    n_sites = next(iter(alignment.values())).shape[0]
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
    return internal_splits(found)


def bootstrap_support(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    rng: np.random.Generator,
    n_replicates: int,
    *,
    workers: int,
    moves: MoveSet = MoveSet.NNI,
    model: Model = Model.JC,
    max_evaluations: int = 200,
) -> dict[frozenset[str], float]:
    """Felsenstein's bootstrap: per internal split of ``topology``, the fraction of replicates whose search returns it.

    Each replicate draws its own generator, spawned from ``rng`` in
    replicate order (:func:`snakes_and_ladders.parallel.map_tasks`), resamples
    the sites with replacement from it, then searches under ``moves`` within
    ``max_evaluations`` candidates, the search's own budget unit. The support
    of a split is a frequency over replicates and nothing more; whether it
    tracks the flat-prior weight is measured, not assumed.

    ``workers`` is how many replicates run at once on a process pool; ``1``
    is the serial loop, and every count returns the same frequencies because
    replicate ``i`` draws the same stream under each. Measured under 2x at 4
    workers at the mid-size tier (``STATUS.md`` §0), so callers pass ``1``
    until a measurement on their hardware says otherwise.

    Raises
    ------
    ValueError
        If ``n_replicates`` or ``workers`` is not positive.
    """
    if n_replicates < 1:
        msg = f"a bootstrap needs at least one replicate, got {n_replicates}"
        raise ValueError(msg)
    counts: dict[frozenset[str], int] = dict.fromkeys(internal_splits(topology), 0)
    found = map_tasks(
        _replicate,
        [(alignment, k, model, moves, max_evaluations)] * n_replicates,
        workers=workers,
        backend=_BOOTSTRAP_BACKEND,
        intra_op_threads=_BOOTSTRAP_INTRA_OP_THREADS,
        generator=rng,
    )
    for splits in found:
        for split in splits:
            if split in counts:
                counts[split] += 1
    return {split: count / n_replicates for split, count in counts.items()}


def split_pattern_support(
    split: frozenset[str], alignment: Mapping[str, np.ndarray], k: int
) -> float:
    """Fraction of sites compatible with ``split``: at most one state on both sides.

    A site whose states partition the taxa without crossing the split can be
    explained with one change per extra state, so it costs the split
    nothing; a site with two states each straddling the split cannot, and is
    the pattern a bootstrap replicate rich in it would drop the split for.
    One vectorized pass over the alignment per state.

    Parameters
    ----------
    split : frozenset[str]
        One side of a bipartition of the alignment's taxa, as
        :func:`snakes_and_ladders.search.topology.leaf_bipartitions` canonicalizes it.
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.

    Returns
    -------
    float
        In ``[0, 1]``; ``1.0`` for a trivial split, which no site can cross.

    Raises
    ------
    ValueError
        If ``split`` names a taxon the alignment lacks, or is the whole
        alignment or empty --- a side with no taxa is not a bipartition.
    """
    missing = sorted(split - set(alignment))
    if missing:
        msg = f"split names taxa the alignment lacks: {missing}"
        raise ValueError(msg)
    outside = sorted(set(alignment) - split)
    if not split or not outside:
        msg = "a split needs at least one taxon on each side"
        raise ValueError(msg)
    inside_states = np.stack([alignment[name] for name in sorted(split)])
    outside_states = np.stack([alignment[name] for name in outside])
    straddling = np.zeros(inside_states.shape[1], dtype=np.int64)
    for state in range(k):
        straddling += (inside_states == state).any(axis=0) & (
            outside_states == state
        ).any(axis=0)
    return float(np.mean(straddling <= 1))


def pattern_support(
    topology: Topology, alignment: Mapping[str, np.ndarray], k: int
) -> dict[frozenset[str], float]:
    """Per internal split of ``topology``, :func:`split_pattern_support`.

    The same keys :func:`bootstrap_support` reports, so the two are compared
    split by split.
    """
    return {
        split: split_pattern_support(split, alignment, k)
        for split in internal_splits(topology)
    }
