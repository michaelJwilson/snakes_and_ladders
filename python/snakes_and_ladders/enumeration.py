"""One enumeration, and one policy for when it is too large to attempt.

Enumeration is the oracle almost every claim in this repository rests on, so
the one thing that should not vary is how it declines. It varied four ways
(issue #230): ``200_000`` configurations in :mod:`snakes_and_ladders.likelihood.potts`,
``200_000`` paths in :mod:`snakes_and_ladders.likelihood.hmm_paths`, ``20`` nodes in
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

**The enumeration itself is also one** (issues #387 and #717). Five modules
built the product space ``range(n_states) ** n_sites`` with their own
``itertools.product``, two of them without the cap, and three took the
argmax over it with their own loop; :func:`configurations` is the product
space under the cap, :func:`posterior` the normalized weights and
``log Z`` over it, :func:`site_marginals` the per-site sums, and
:func:`argmax` the first maximizer. Each adapter keeps its signature and its
arithmetic: what moved is the space, not the weight, so every value the
adapters returned before is the value they return.

The module names no model, so anything may import it, on the same terms as
:mod:`snakes_and_ladders.numerics`.
"""

from __future__ import annotations

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


def configurations(
    n_states: int,
    n_sites: int,
    *,
    what: str | None = None,
    limit: int = MAX_ENUMERABLE_CONFIGURATIONS,
) -> np.ndarray:
    """Every assignment of ``n_states`` states to ``n_sites`` sites, in lexicographic order.

    Parameters
    ----------
    n_states : int
        States per site.
    n_sites : int
        Sites.
    what : str | None
        What the assignments are, for the refusal; ``None`` names them as
        ``"{n_states}**{n_sites} configurations"``.
    limit : int
        The cap, passed to :func:`refuse_oversized`.

    Returns
    -------
    np.ndarray
        Shape ``(n_states ** n_sites, n_sites)``, ``int64``, the order
        ``itertools.product(range(n_states), repeat=n_sites)`` gives.

    Raises
    ------
    ValueError
        Above the cap.

    Examples
    --------
    >>> configurations(2, 2).tolist()
    [[0, 0], [0, 1], [1, 0], [1, 1]]
    """
    count = n_states**n_sites
    refuse_oversized(
        count,
        what=f"{n_states}**{n_sites} configurations" if what is None else what,
        limit=limit,
    )
    if n_sites == 0:
        return np.zeros((1, 0), dtype=np.int64)
    grid = np.indices((n_states,) * n_sites, dtype=np.int64)
    return np.ascontiguousarray(grid.reshape(n_sites, -1).T)


def posterior(log_weights: np.ndarray) -> tuple[float, np.ndarray]:
    """``(log Z, p)`` over an enumeration: the shifted exponential, summed and normalized.

    The arithmetic :func:`snakes_and_ladders.likelihood.potts.enumerate_potts`
    carried --- shift by the peak, exponentiate, sum, divide --- so the
    numbers it returned before are the numbers it returns.
    """
    peak = float(log_weights.max())
    unnormalized = np.exp(log_weights - peak)
    total = float(unnormalized.sum())
    return float(np.log(total) + peak), unnormalized / total


def site_marginals(
    configurations: np.ndarray, weights: np.ndarray, n_states: int
) -> np.ndarray:
    """Per-site sums of ``weights`` over the state each configuration puts there.

    Shape ``(n_sites, n_states)``. The additions run in configuration order
    at every cell, as a loop over the configurations would run them.
    """
    n_sites = configurations.shape[1]
    marginals = np.zeros((n_sites, n_states))
    for site in range(n_sites):
        np.add.at(marginals[site], configurations[:, site], weights)
    return marginals


def argmax(values: np.ndarray) -> int:
    """The first index of the largest value: ties resolve to the lexicographically first configuration."""
    return int(np.argmax(values))
