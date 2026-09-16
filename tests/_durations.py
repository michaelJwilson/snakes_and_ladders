"""The duration guard: a slow test carries the marker that says when it runs.

`DEV.md` budgets the CI tier at 300 s and forbids asserting wall clock in the
suite, since a timing assertion fails for the machine rather than for the
change. What can be asserted is the tier: a test that took longer than the
cap on the reference host, and carries none of the out-of-tier markers, is in
the wrong tier, and the pull request that added it would have added its
minute to every run unnoticed (issue #372). ``tests/conftest.py`` records
each test's call duration and, when ``SAL_DURATION_CAP`` is set -- as
``infra/validate.sh`` sets it on the reference host and CI does not -- fails
the session naming the offenders. Imported rather than collected, per the
fixture rule in `DEV.md`.

**The key tier is exempt from that cap and held to its own** (issue #399).
The key fixture is the largest declared instance of a problem whose full test
--- simulate, fit, assert --- fits a budget a per-pull-request test cannot:
120 s, ``SAL_KEY_DURATION_CAP``. Exempting it without a second cap would make
``key`` the marker any slow test acquires, so the exemption comes with a
ceiling, and a key test over *that* fails the session the same way.
"""

from __future__ import annotations

from collections.abc import Iterable

#: The markers that say a test runs outside the per-pull-request tier.
OUTSIDE_THE_TIER = frozenset({"release", "stress", "key"})

#: The marker that says a test is a key fixture's, and so is held to
#: :envvar:`SAL_KEY_DURATION_CAP` rather than exempt from every cap.
KEY = "key"


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


def key_over_cap(
    durations: Iterable[tuple[str, float, frozenset[str]]], cap: float
) -> list[str]:
    """Name the ``key`` tests over ``cap`` seconds.

    The key tier's own guard. A key fixture is *defined* as the largest
    declared instance whose full test fits this budget, so a key test over it
    is not a slow test to wait for: it is a fixture whose key instance is the
    wrong one, and the fix is to mark a coarser instance key.

    Parameters
    ----------
    durations : Iterable[tuple[str, float, frozenset[str]]]
        Per test: its node id, its call duration in seconds, and the names of
        the markers on it.
    cap : float
        The longest a key test may run, in seconds.

    Returns
    -------
    list[str]
        One line per offender, slowest first; empty when every key test fits.
    """
    offenders = [
        (seconds, node_id)
        for node_id, seconds, markers in durations
        if seconds > cap and KEY in markers
    ]
    return [
        f"{node_id}: {seconds:.1f} s over the {cap:.0f} s key cap; "
        f"the key instance is the largest that fits it"
        for seconds, node_id in sorted(offenders, reverse=True)
    ]


def outside_the_tier(
    items: Iterable[tuple[str, frozenset[str]]],
) -> list[str]:
    """Name the early-gate tests a scale marker puts outside the per-PR tier.

    `critical` and the scale markers answer different questions, and a test
    carrying both answers them inconsistently: it says "gate on this" and "do
    not run this per pull request" at once. ``-m critical`` settles it the
    wrong way --- a later ``-m`` *replaces* ``addopts``' ``-m "not release"``
    rather than intersecting with it (`DEV.md`), so the early gate wins and a
    release-tier case runs in it.

    **Read from collected items, not from the source.** A scale marker is
    attached by ``tests/_scale.at_bin`` and ``at_scale`` from the fixture
    file's own declaration, so no decorator says ``release`` and an `ast` scan
    of the tree --- which is how `tests/regression/test_test_kinds.py` reads
    markers, for its own good reasons --- sees a test that is not there. Issue
    #635 added a `critical` marker to one such test and put 91.4 s of a
    `release` case into a 42 s gate.

    Parameters
    ----------
    items : Iterable[tuple[str, frozenset[str]]]
        Per collected item: its node id and the names of the markers on it.

    Returns
    -------
    list[str]
        One line per offender, in collection order; empty when no early-gate
        test is scale-marked out of the tier.
    """
    return [
        f"{node_id}: gates early and is {'/'.join(sorted(markers & OUTSIDE_THE_TIER))}; "
        f"a critical test runs per pull request, so drop one of the two"
        for node_id, markers in items
        if "critical" in markers and markers & OUTSIDE_THE_TIER
    ]
