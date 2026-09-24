"""The duration guard: a slow test carries the marker that says when it runs.

`DEV.md` budgets the CI tier at 300 s and forbids asserting wall clock, so the
tier is asserted instead: a test over the cap on the reference host with no
out-of-tier marker is in the wrong tier (issue #372). ``tests/conftest.py``
records call durations and, with ``SAL_DURATION_CAP`` set (``infra/validate.sh``
sets it, CI does not), fails the session naming the offenders.

The key tier is exempt and held to its own ceiling, ``SAL_KEY_DURATION_CAP``
= 120 s (issue #399), so ``key`` is not a marker any slow test can acquire.
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

    ``durations`` is (node id, seconds, marker names); offenders slowest first.
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
    """Name the ``key`` tests over ``cap`` seconds, slowest first.

    A key test over the cap means the wrong instance is marked key.
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

    From collected items; #635 put 91.4 s of a `release` case into a 42 s gate.
    """
    return [
        f"{node_id}: gates early and is {'/'.join(sorted(markers & OUTSIDE_THE_TIER))}; "
        f"a critical test runs per pull request, so drop one of the two"
        for node_id, markers in items
        if "critical" in markers and markers & OUTSIDE_THE_TIER
    ]
