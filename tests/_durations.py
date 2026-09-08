"""The duration guard: a slow test carries the marker that says when it runs.

`DEV.md` budgets the CI tier at 300 s and forbids asserting wall clock in the
suite, since a timing assertion fails for the machine rather than for the
change. What can be asserted is the tier: a test that took longer than the
cap on the reference host, and carries neither ``release`` nor ``stress``,
is in the wrong tier, and the pull request that added it would have added its
minute to every run unnoticed (issue #372). ``tests/conftest.py`` records
each test's call duration and, when ``SAL_DURATION_CAP`` is set -- as
``infra/validate.sh`` sets it on the reference host and CI does not -- fails
the session naming the offenders. Imported rather than collected, per the
fixture rule in `DEV.md`.
"""

from __future__ import annotations

from collections.abc import Iterable

#: The markers that say a test runs outside the per-pull-request tier.
OUTSIDE_THE_TIER = frozenset({"release", "stress"})


def over_cap(
    durations: Iterable[tuple[str, float, frozenset[str]]], cap: float
) -> list[str]:
    """Name the tests over ``cap`` seconds that carry no out-of-tier marker.

    Parameters
    ----------
    durations : Iterable[tuple[str, float, frozenset[str]]]
        Per test: its node id, its call duration in seconds, and the names of
        the markers on it.
    cap : float
        The longest a per-pull-request test may run, in seconds.

    Returns
    -------
    list[str]
        One line per offender, slowest first, giving the node id and the
        seconds; empty when every slow test is already release- or
        stress-gated.
    """
    offenders = [
        (seconds, node_id)
        for node_id, seconds, markers in durations
        if seconds > cap and not (markers & OUTSIDE_THE_TIER)
    ]
    return [
        f"{node_id}: {seconds:.1f} s over the {cap:.0f} s cap; mark it release or stress"
        for seconds, node_id in sorted(offenders, reverse=True)
    ]
