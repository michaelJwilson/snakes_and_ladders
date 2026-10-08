"""`search.ground_state` is a `match` onto one explicit call per `METHODS` entry, which returns the same run (issue #1305).

Read against the explicit calls themselves: for every `METHODS` entry the
generic call and its explicit call return equal runs on one seeded lattice,
every field bitwise but the seconds each run measured, from a start and
without, and with the options the entry takes. `METHODS`, `FLOORED` and the
annealed names are read from the explicit calls, so a call is an entry and
an entry a call; the `match` reaches each entry's call once; a name with no
case, and an option the entry's call does not take, are refused by name.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.search import ground_state as module
from sal.search.ground_state import (
    ANNEAL_OPTIONS,
    DESCENT_OPTIONS,
    FLOORED,
    METHODS,
    GroundState,
    MethodRun,
    entry_name,
    ground_state,
    lattice_rung,
)

RUNG = lattice_rung(6, 3, seed=1305)
BUDGET = Budget(Cost.SITE_VISITS, 12 * RUNG.visits_per_sweep)
START = np.random.default_rng(1305).integers(0, 3, RUNG.n_nodes)
#: The entries with no single starting labelling, which refuse a start.
#: What a recording call returns in place of a run.
SENTINEL: Any = object()
NO_START = frozenset({"field_argmax", "max-product", "bifurcation"})


def _explicit(name: str) -> Callable[..., MethodRun]:
    return getattr(ground_state, name.replace("-", "_"))  # type: ignore[no-any-return]


def _same(ours: MethodRun, theirs: MethodRun) -> None:
    """Every field of two runs equal, the labelling bitwise, the seconds aside."""
    for item in dataclasses.fields(ours):
        if item.name == "seconds":
            continue
        left, right = getattr(ours, item.name), getattr(theirs, item.name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right), item.name
        else:
            assert left == right, item.name


@pytest.mark.critical
@pytest.mark.analytic
def test_methods_are_the_explicit_calls_one_each_in_order() -> None:
    calls = [
        name
        for name, value in vars(GroundState).items()
        if isinstance(value, staticmethod)
    ]
    assert list(METHODS) == [entry_name(name) for name in calls]
    for name in calls:
        assert METHODS[entry_name(name)] is getattr(module, f"run_{name}"), name
        assert "solver" not in inspect.signature(getattr(GroundState, name)).parameters
    # The option tables are read from the calls' keywords, and are main's.
    assert {"icm", "icm-random"} == FLOORED
    assert {
        "anneal",
        "swendsen-wang",
        "wolff",
        "swendsen-wang-heat-bath",
        "wolff-heat-bath",
    } == module._ANNEALED_METHODS


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize("method", list(METHODS))
def test_the_match_reaches_each_entrys_explicit_call_once(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    reached: list[str] = []
    for name in METHODS:

        def record(*_args: Any, _name: str = name, **_kwargs: Any) -> Any:
            reached.append(_name)
            return SENTINEL

        monkeypatch.setattr(GroundState, name.replace("-", "_"), staticmethod(record))
    assert (
        ground_state(RUNG.graph, RUNG.field, method, BUDGET, np.random.default_rng(0))
        is SENTINEL
    )
    assert reached == [method]


@pytest.mark.analytic
@pytest.mark.parametrize("method", list(METHODS))
def test_a_generic_run_is_its_explicit_calls_bitwise(method: str) -> None:
    ours = ground_state(
        RUNG.graph, RUNG.field, method, BUDGET, np.random.default_rng(7)
    )
    theirs = _explicit(method)(RUNG.graph, RUNG.field, BUDGET, np.random.default_rng(7))
    _same(ours, theirs)
    # And both are the stage `METHODS` holds, run on the rung.
    _same(ours, METHODS[method](RUNG, BUDGET, np.random.default_rng(7)))


@pytest.mark.analytic
@pytest.mark.parametrize("method", [name for name in METHODS if name not in NO_START])
def test_a_generic_run_from_a_start_is_its_explicit_calls_bitwise(method: str) -> None:
    ours = ground_state(
        RUNG.graph, RUNG.field, method, BUDGET, np.random.default_rng(3), start=START
    )
    theirs = _explicit(method)(
        RUNG.graph, RUNG.field, BUDGET, np.random.default_rng(3), start=START
    )
    _same(ours, theirs)


@pytest.mark.analytic
@pytest.mark.parametrize("method", sorted(module._ANNEALED_METHODS | FLOORED))
def test_an_option_reaches_the_explicit_call_bitwise(method: str) -> None:
    options: dict[str, Any] = (
        {"backend": Backend.PYTHON, "min_sites": 2}
        if method in FLOORED
        else {
            "schedule": ScheduleParams(ScheduleShape.LINEAR, 1.0, 0.1),
            "steps": 5,
        }
    )
    ours = ground_state(
        RUNG.graph, RUNG.field, method, BUDGET, np.random.default_rng(5), **options
    )
    theirs = _explicit(method)(
        RUNG.graph, RUNG.field, BUDGET, np.random.default_rng(5), **options
    )
    _same(ours, theirs)


@pytest.mark.critical
@pytest.mark.analytic
def test_each_explicit_call_takes_only_its_own_options() -> None:
    for name in METHODS:
        keywords = module.explicit_keywords(name)
        assert (keywords >= DESCENT_OPTIONS) == (name in FLOORED), name
        assert not (keywords & ANNEAL_OPTIONS and keywords & DESCENT_OPTIONS), name
    icm: Any = ground_state.icm
    with pytest.raises(TypeError, match="schedule"):
        icm(
            RUNG.graph,
            RUNG.field,
            BUDGET,
            np.random.default_rng(0),
            schedule=ScheduleParams(ScheduleShape.LINEAR, 1.0, 0.1),
        )
    with pytest.raises(ValueError, match="runs no anneal"):
        ground_state(
            RUNG.graph, RUNG.field, "icm", BUDGET, np.random.default_rng(0), steps=3
        )
    with pytest.raises(ValueError, match="runs no single-site descent"):
        ground_state(
            RUNG.graph,
            RUNG.field,
            "wolff",
            BUDGET,
            np.random.default_rng(0),
            min_sites=2,
        )


@pytest.mark.critical
@pytest.mark.analytic
def test_an_entry_with_no_case_and_an_unknown_name_are_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(METHODS, "planted", METHODS["icm"])
    with pytest.raises(ValueError, match="'planted' is a method, but no case"):
        ground_state(
            RUNG.graph, RUNG.field, "planted", BUDGET, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="no ground-state method 'absent'"):
        ground_state(RUNG.graph, RUNG.field, "absent", BUDGET, np.random.default_rng(0))
