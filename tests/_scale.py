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
from phylo.fixtures import Scale

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
        fits the per-pull-request budget.
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
