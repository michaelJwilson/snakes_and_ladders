"""Running one test body at two sizes, under two time budgets.

`DEV.md` budgets the per-PR suite at 5 minutes and ``stress`` at 10, enforced
by size. :func:`at_scale` parameterizes one body over both, so an assertion
change reaches the large size by construction; two tests would drift.
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.sim.fixtures import KEY, Fixture, fixture, tiers

#: The interface, re-exports included: a bare re-export such as `Fixture` is
#: private under ``mypy --strict``.
__all__ = [
    "KEY",
    "Fixture",
    "at_bin",
    "at_fixture",
    "at_scale",
    "fixture",
    "key_only",
    "scaled_values",
    "stress_only",
    "tiers",
]

T = TypeVar("T")


def at_scale(argument: str, ci: T, stress: T) -> pytest.MarkDecorator:
    """Parameterize one test over its CI size and its ``stress``-marked size.

    ``ci`` keeps an exact oracle in budget; an instance belongs in :func:`at_fixture`.
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

    One case ``<problem>-<tier>`` per tier, with that tier's marker (issue #864).
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

    For a test with no smaller size that asserts the same thing.
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

    Cases ``bin-<factor>``, each with the tier the key file marks it with.
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
    """Mark a test as the key fixture's, held to ``SAL_KEY_DURATION_CAP``.

    For the declared instance run end to end (`DEV.md`, ``tests/_durations.py``).
    """
    return pytest.mark.key(reason=reason)


def scaled_values(scale: Scale, ci: dict[str, Any], stress: dict[str, Any]) -> Any:
    """Pick ``ci`` or ``stress`` by ``scale``, for sizes that vary as a set."""
    return ci if scale is Scale.CI else stress
