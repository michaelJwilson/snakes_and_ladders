"""Running one test body at two sizes, under two time budgets.

`DEV.md` sets the budgets this exists to keep: the per-pull-request suite
inside 5 minutes, the ``stress`` tier inside 10, and the release gate outside
both. Those are enforced by *size*, so a size has to be selectable rather
than written into each test.

The alternative is two tests, and it fails in the direction that matters: the
small one keeps being maintained and the large one drifts until it asserts
something the small one no longer does. :func:`at_scale` parameterizes one
body over both, so a change to the assertion reaches the large size by
construction.
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.sim.fixtures import KEY, Fixture, fixture, tiers  # noqa: F401

T = TypeVar("T")


def at_scale(argument: str, ci: T, stress: T) -> pytest.MarkDecorator:
    """Parameterize one test over its CI size and its stress size.

    The CI case runs on every pull request; the stress case carries the
    ``stress`` marker, so it is deselected there and runs under the 10-minute
    developer budget instead.

    Parameters
    ----------
    argument : str
        Name of the test argument receiving the size.
    ci : T
        The size at which an exact oracle is still available and the test
        fits the per-pull-request budget. A size that *is* a problem
        instance belongs in the registry instead: see :func:`at_fixture`.
    stress : T
        The size that shows the same property where the CI size cannot ---
        more replicates, a longer chain, a larger structure.

    Returns
    -------
    pytest.MarkDecorator
        A ``parametrize`` decorator carrying both cases.

    Examples
    --------
    >>> @at_scale("n_taxa", ci=6, stress=7)
    ... def test_neighbour_count(n_taxa: int) -> None:
    ...     ...
    """
    return pytest.mark.parametrize(
        argument,
        [
            pytest.param(ci, id=str(Scale.CI)),
            pytest.param(stress, id=str(Scale.STRESS), marks=pytest.mark.stress),
        ],
    )


#: The scheduling marker each tier carries. The CI tier carries none: it is
#: what runs when nothing is deselected.
_MARKS: dict[Scale, tuple[pytest.MarkDecorator, ...]] = {
    Scale.CI: (),
    Scale.STRESS: (pytest.mark.stress,),
    Scale.RELEASE: (pytest.mark.release,),
}


def at_fixture(argument: str, problem: str) -> pytest.MarkDecorator:
    """Parameterize one test over every instance a problem declares.

    The registry-driven form of :func:`at_scale`: the sizes are the fixture
    files' (`snakes_and_ladders.sim.fixtures`) rather than literals in the
    test, so a problem that gains a tier gains a case in every test written
    this way, and the tier a case runs at is the fixture's own.

    Parameters
    ----------
    argument : str
        Name of the test argument receiving the
        :class:`~snakes_and_ladders.sim.fixtures.Fixture`.
    problem : str
        The registry problem, which is the fixture directory's name.

    Returns
    -------
    pytest.MarkDecorator
        A ``parametrize`` decorator with one case per declared tier, each
        identified as ``<problem>-<tier>`` and carrying that tier's
        scheduling marker.

    Examples
    --------
    >>> @at_fixture("instance", "potts_lattice")
    ... def test_marginals_match_enumeration(instance: Fixture) -> None:
    ...     assert instance.oracle == "enumeration"
    """
    return pytest.mark.parametrize(
        argument,
        [
            pytest.param(
                fixture(problem, tier),
                id=f"{problem}-{tier}",
                marks=_MARKS[tier],
            )
            for tier in tiers(problem)
        ],
    )


def stress_only(reason: str) -> pytest.MarkDecorator:
    """Mark a whole test as stress-tier, with the reason it does not fit CI.

    Used where a test has no smaller size that still asserts the same thing
    --- a chain long enough to have converged, a sweep whose shape is the
    result --- so parameterizing it would leave a CI case asserting less than
    the test claims.

    Parameters
    ----------
    reason : str
        Why the CI budget cannot hold this test. Recorded on the marker so a
        reader of the test does not have to reconstruct it.

    Returns
    -------
    pytest.MarkDecorator
        The ``stress`` marker.
    """
    return pytest.mark.stress(reason=reason)


#: The scheduling marker each bin instance's declared tier carries. The ci
#: tier carries none, as :data:`_MARKS` says for a fixture's own tier.
_BIN_MARKS: dict[str, tuple[pytest.MarkDecorator, ...]] = {
    "ci": (),
    "stress": (pytest.mark.stress,),
    "release": (pytest.mark.release,),
    "key": (pytest.mark.key,),
}


def at_bin(argument: str, problem: str) -> pytest.MarkDecorator:
    """Parameterize one test over every bin factor a count-pair fixture declares.

    The registry-driven form of :func:`at_scale` for a problem whose sizes are
    *one* declared instance binned to several resolutions
    (:mod:`snakes_and_ladders.sim.count_pairs`): the cases are the file's
    ``bin`` entries and each carries the tier the file marks it with, which is
    a measurement of the instance rather than a literal in the test.

    Parameters
    ----------
    argument : str
        Name of the test argument receiving the bin factor.
    problem : str
        The registry problem whose key file declares the factors.

    Returns
    -------
    pytest.MarkDecorator
        A ``parametrize`` decorator with one case per declared factor,
        identified as ``bin-<factor>``.
    """
    declared = fixture(problem, KEY).params
    return pytest.mark.parametrize(
        argument,
        [
            pytest.param(
                entry.factor,
                id=f"bin-{entry.factor}",
                marks=_BIN_MARKS[entry.marker],
            )
            for entry in declared.bins
        ],
    )


def key_only(reason: str) -> pytest.MarkDecorator:
    """Mark a test as the key fixture's, with the reason it is not in the tier.

    The key tier is the one exempt from ``SAL_DURATION_CAP`` and held to
    ``SAL_KEY_DURATION_CAP`` instead (`DEV.md`, and ``tests/_durations.py``).
    Used where the test *is* the declared instance run end to end, so no
    smaller size asserts the same thing and the instance's own budget is the
    only one that applies.

    Parameters
    ----------
    reason : str
        Why the per-pull-request budget cannot hold this test. Recorded on the
        marker, as :func:`stress_only` records its own.

    Returns
    -------
    pytest.MarkDecorator
        The ``key`` marker.
    """
    return pytest.mark.key(reason=reason)


def scaled_values(scale: Scale, ci: dict[str, Any], stress: dict[str, Any]) -> Any:
    """Pick the parameter set matching ``scale``.

    For a test that varies several sizes together, where parameterizing each
    separately would multiply cases that are only meaningful as a set.

    Parameters
    ----------
    scale : Scale
        Which budget the caller is running under.
    ci : dict[str, Any]
        Parameters at the CI size.
    stress : dict[str, Any]
        Parameters at the stress size.

    Returns
    -------
    Any
        ``ci`` or ``stress``.
    """
    return ci if scale is Scale.CI else stress
