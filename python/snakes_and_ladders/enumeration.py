"""One policy for when an enumeration is too large to attempt.

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

The module names no model, so anything may import it, on the same terms as
:mod:`snakes_and_ladders.numerics`.
"""

from __future__ import annotations

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
