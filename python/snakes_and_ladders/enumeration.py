"""One weighted enumeration, and one policy for when it is too large.

Enumeration is the oracle almost every claim in this repository rests on, so
the one thing that should not vary is how it enumerates and how it declines.

**How it declined** varied four ways (issue #230): ``200_000`` configurations
in :mod:`snakes_and_ladders.likelihood.potts`, ``200_000`` paths in
:mod:`snakes_and_ladders.likelihood.hmm_paths`, ``20`` nodes in
:mod:`snakes_and_ladders.search.max_cut`, and a docstring-only ``n <= 6`` in
:mod:`snakes_and_ladders.likelihood.brute_force` that nothing enforced --- four thresholds
in three units, with four differently worded errors, and one that was not a
check at all.

**The unit is configurations**, because every one of those reduces to it:
``k ** T`` hidden paths, ``2 ** n`` cuts, ``q ** n_nodes`` spin
configurations and ``k ** internal`` ancestral labellings are all counts of
things enumerated. A call site converts its own quantity and names it; this
module decides only whether the count is affordable and what the refusal
says.

**What it enumerated** varied eight ways (issue #387, from the 0.4.0 audit in
``STATUS.md``): three weighted enumerations that are one algorithm under
three result types, two uncapped ``itertools.product`` copies in
:mod:`snakes_and_ladders.learn`, and three copies of an argmax over the product with a
lexicographic tie rule. The pieces they share are here --- the product
itself, the shift-and-normalize, the accumulation into per-site bins, and the
argmax --- and each consumer keeps its own log-weight and its own result type
over them. What each function's consumers are is stated in its docstring,
because a seam with one implementer is a function that has been moved rather
than shared.

Two enumerations stay distinct and are not adapters here, as that audit
classified them: :func:`snakes_and_ladders.search.topology.enumerate_topologies` walks
unrooted topologies by stepwise insertion rather than a product, and
:func:`snakes_and_ladders.likelihood.ldpc.enumerate_codewords` enumerates information
bits and maps them through a generator matrix.

The module names no model, so anything may import it, on the same terms as
:mod:`snakes_and_ladders.numerics`. That is what makes it the home of the seam:
``learn/CLAUDE.md`` forbids importing :mod:`snakes_and_ladders.likelihood`, so a seam under
``likelihood/`` --- where issue #387's plan first put it --- could not have
served the ``learn`` copies it exists to remove.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator

import numpy as np

#: Configurations above which an enumeration is refused rather than attempted.
#: Chosen where it was already set for two of the four call sites, so this
#: consolidation moves no boundary: `200_000` configurations at a few hundred
#: bytes each is the point where a test stops being a test and becomes a
#: memory failure, and a run killed by the kernel reads as infrastructure
#: breaking rather than as the stated limit it is.
MAX_ENUMERABLE_CONFIGURATIONS = 200_000


def refuse_oversized(
    count: int, *, what: str, limit: int = MAX_ENUMERABLE_CONFIGURATIONS
) -> None:
    """Raise if enumerating ``count`` things is past the limit.

    Parameters
    ----------
    count : int
        Configurations the caller is about to enumerate, already converted
        into that unit.
    what : str
        What is being enumerated, and how the count arises --- e.g.
        ``"3 ** 12 hidden paths"``. It appears in the message, because a
        refusal that does not say which size was too large leaves the caller
        guessing at which parameter to reduce.
    limit : int
        Override for a call site with a genuinely different affordability,
        stated at that site rather than assumed here.

    Raises
    ------
    ValueError
        If ``count`` exceeds ``limit``.

    Examples
    --------
    >>> refuse_oversized(10, what="10 configurations")
    >>> refuse_oversized(10, what="10 configurations", limit=5)
    Traceback (most recent call last):
        ...
    ValueError: refusing to enumerate 10 configurations: 10 configurations is
    past the limit of 5
    """
    if count > limit:
        msg = (
            f"refusing to enumerate {count} configurations: {what} is past "
            f"the limit of {limit}"
        )
        raise ValueError(msg)


def assignments(
    n_states: int,
    n_sites: int,
    *,
    what: str,
    limit: int = MAX_ENUMERABLE_CONFIGURATIONS,
) -> Iterator[tuple[int, ...]]:
    """Every assignment of ``n_sites`` sites to ``n_states`` states, lexicographically.

    ``n_states ** n_sites`` of them, refused above ``limit`` before the first
    is produced rather than part-way through: a caller that has already
    consumed half an enumeration has spent the memory the refusal exists to
    save.

    Consumers: :func:`snakes_and_ladders.learn.potts.enumerate_configurations`,
    :func:`snakes_and_ladders.learn.hmm.enumerate_paths` and the three enumerations in
    :mod:`snakes_and_ladders.learn.relaxed`, each of which held its own uncapped
    ``itertools.product`` before issue #387.

    Parameters
    ----------
    n_states, n_sites : int
        States per site, and sites.
    what : str
        What is being enumerated, for the refusal message; see
        :func:`refuse_oversized`.
    limit : int
        Refuse above this many assignments.

    Returns
    -------
    Iterator[tuple[int, ...]]
        Lazily, in lexicographic order.

    Raises
    ------
    ValueError
        If the enumeration exceeds ``limit``.
    """
    refuse_oversized(n_states**n_sites, what=what, limit=limit)
    return itertools.product(range(n_states), repeat=n_sites)


def assignment_table(
    n_states: int,
    n_sites: int,
    *,
    what: str,
    limit: int | None = MAX_ENUMERABLE_CONFIGURATIONS,
) -> np.ndarray:
    """The same assignments as one contiguous ``(n_states ** n_sites, n_sites)`` array.

    The array form rather than the iterator, because a weighted enumeration
    scores every assignment at once: the log weight is a vectorized
    expression over a column of this table, not a Python loop over tuples.

    Consumers: :mod:`snakes_and_ladders.likelihood.potts` (configurations, and the
    transfer matrix's columns), :mod:`snakes_and_ladders.likelihood.hmm_paths` (paths) and
    :mod:`snakes_and_ladders.likelihood.spatio_sequential` (labellings, and paths per
    class).

    Parameters
    ----------
    n_states, n_sites : int
        States per site, and sites.
    what : str
        What is being enumerated, for the refusal message.
    limit : int | None
        Refuse above this many assignments. ``None`` where the caller has
        already refused a count this one divides --- the coupled model
        refuses on labellings times paths per class, which dominates either
        factor, and checking a factor afterwards could only repeat a
        judgement already made under a less informative name.

    Returns
    -------
    np.ndarray
        Shape ``(n_states ** n_sites, n_sites)``, dtype ``int64``, in
        lexicographic order.

    Raises
    ------
    ValueError
        If the enumeration exceeds ``limit``.
    """
    if limit is not None:
        refuse_oversized(n_states**n_sites, what=what, limit=limit)
    return np.array(
        list(itertools.product(range(n_states), repeat=n_sites)), dtype=np.int64
    ).reshape(-1, n_sites)


def normalize(log_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Shift by the peak, exponentiate, and report the normalizer.

    The shift is the largest log weight, so the largest exponential is
    ``1`` and nothing overflows; the smallest underflows to zero, which is
    the term that contributes nothing to the sum anyway.

    Consumers: :func:`snakes_and_ladders.likelihood.potts.enumerate_potts`, which weights
    its marginals by the normalized probability, and
    :func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths`, which
    accumulates the unnormalized weights and normalizes each site's row
    afterwards. Both forms are returned because the two roundings differ in
    the last ulp and each is what its own pins were taken against.

    Parameters
    ----------
    log_weight : np.ndarray
        Unnormalized log weights, shape ``(n_assignments,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        The shifted exponentials ``exp(w - max w)``, those divided by their
        sum, and ``log sum exp(w)``.
    """
    peak = float(log_weight.max())
    shifted = np.exp(log_weight - peak)
    total = shifted.sum()
    return shifted, shifted / total, float(np.log(total) + peak)


def accumulate(table: np.ndarray, weight: np.ndarray, n_states: int) -> np.ndarray:
    """Sum ``weight`` into per-site, per-state bins.

    ``out[site, state]`` is the total weight of the assignments that put
    ``site`` in ``state``. The sum runs in assignment order for every site,
    so a caller that replaces its own loop with this one gets the same
    floating-point result rather than a re-associated one.

    Consumers: :func:`snakes_and_ladders.likelihood.potts.enumerate_potts` (single-site
    marginals), :func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths`
    (the path posterior) and
    :func:`snakes_and_ladders.likelihood.spatio_sequential.enumerate_spatio_sequential`
    (the label posterior, and the state posterior of each class).

    Parameters
    ----------
    table : np.ndarray
        Assignments, shape ``(n_assignments, n_sites)``.
    weight : np.ndarray
        One weight per assignment, shape ``(n_assignments,)``.
    n_states : int
        States per site; sets the width of the result, which a table that
        happens not to visit every state would otherwise understate.

    Returns
    -------
    np.ndarray
        Shape ``(n_sites, n_states)``.
    """
    out = np.zeros((table.shape[1], n_states))
    for site in range(table.shape[1]):
        np.add.at(out[site], table[:, site], weight)
    return out


def best_assignment(
    candidates: Iterable[tuple[int, ...]], score: Callable[[tuple[int, ...]], float]
) -> tuple[tuple[int, ...], float]:
    """The highest-scoring assignment, ties resolving to the first seen.

    The comparison is strict, so an assignment ties only with one already
    kept and the answer is the lexicographically first maximum whenever
    ``candidates`` is in lexicographic order --- which is what makes an
    enumerated optimum reproducible rather than a matter of iteration order.

    Consumers: :func:`snakes_and_ladders.learn.potts.optimum`,
    :func:`snakes_and_ladders.learn.hmm.optimum` and
    :func:`snakes_and_ladders.learn.relaxed.enumerate_optimum`, three copies of this
    loop before issue #387.

    Parameters
    ----------
    candidates : Iterable[tuple[int, ...]]
        The assignments to score, usually from :func:`assignments`.
    score : Callable[[tuple[int, ...]], float]
        What to maximize. Called once per candidate, in order.

    Returns
    -------
    tuple[tuple[int, ...], float]
        The maximizer and its score, as ``score`` returned it.

    Raises
    ------
    ValueError
        If ``candidates`` is empty. There is no maximum of nothing, and
        returning a sentinel would let a caller report one.
    """
    best: tuple[int, ...] | None = None
    best_score = -np.inf
    for candidate in candidates:
        value = score(candidate)
        if value > best_score:
            best, best_score = candidate, value
    if best is None:
        msg = "no candidates to maximize over"
        raise ValueError(msg)
    return best, float(best_score)
