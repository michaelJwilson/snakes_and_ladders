"""Suite-wide configuration: one process is one core, slow tests are named, and no benchmark runs distributed.

**One core per process:** BLAS thread counts are set to one before NumPy
imports unless the caller set them (``qa.build`` strips them for renders).
**A slow test carries its marker** (#372): with ``SAL_DURATION_CAP`` or
``SAL_KEY_DURATION_CAP`` set (``infra/validate.sh``, not CI), an over-cap test
with no out-of-tier marker fails (``tests/_durations.py``). **Problem markers**
(#614, ``tests/_problems.py``). **No early-gate test out of the tier** (#635).
**No benchmark under ``pytest-xdist``** (#405): `pytest-benchmark` disables
itself there and 212 tests would pass measuring nothing. Stale extension: #630.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import pytest  # noqa: E402

from tests._durations import (  # noqa: E402
    key_over_cap,
    outside_the_tier,
    over_cap,
)
from tests._extension import stale_extension  # noqa: E402
from tests._problems import (  # noqa: E402
    CACHE_KEY,
    fixtures_named_in,
    problem_names,
    restore,
    snapshot,
)

DURATIONS = pytest.StashKey[list[tuple[str, float, frozenset[str]]]]()


#: What a problem marker says of itself, once registered.
PROBLEM_MARKER = (
    "{problem}: exercises the {problem} fixture, from the registry call the "
    "test module already makes; added at collection, never by an author "
    "(tests/_problems.py, issue #614)"
)


def pytest_configure(config: pytest.Config) -> None:
    config.stash[DURATIONS] = []
    # First, and before #619's registration below: a stale extension fails the
    # tests that call the signature it predates, which reads as the change
    # under test breaking them (issue #630, `tests/_extension.py`). Everything
    # after it is measuring the wrong binary.
    refusal = stale_extension(Path(__file__).resolve().parent.parent)
    if refusal:
        raise pytest.UsageError(refusal)
    # A fixture directory named after a tier marker would register it and
    # excuse its loaders from the duration cap (`tests/_durations.py`).
    reserved = {
        line.split(":", 1)[0].strip() for line in config.getini("markers") if line
    }
    for problem in problem_names():
        if problem in reserved:
            message = (
                f"fixture directory {problem!r} collides with the marker of that "
                f"name; rename the directory (issue #614)"
            )
            raise pytest.UsageError(message)
        config.addinivalue_line("markers", PROBLEM_MARKER.format(problem=problem))
    cache = getattr(config, "cache", None)
    if cache is not None:
        restore(cache.get(CACHE_KEY, None))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Mark each item with the problems its module loads, cached for the next run.

    ``tryfirst``: `-m` deselects in this hook (`test_problem_markers.py`).
    """
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        found = fixtures_named_in(path)
        for problem in found:
            item.add_marker(problem)
        # A module names a problem or it names `infra`; there is no third
        # state (issue #622). Added here rather than written by an author for
        # the same reason the problem markers are: a hand-written one goes
        # stale the first time a module changes what it exercises.
        if not found:
            item.add_marker("infra")

    conflicts = outside_the_tier(
        (item.nodeid, frozenset(marker.name for marker in item.iter_markers()))
        for item in items
    )
    if conflicts:
        raise pytest.UsageError("\n".join([EARLY_GATE_CONFLICT, *conflicts]))

    cache = getattr(config, "cache", None)
    if cache is not None and not hasattr(config, "workerinput"):
        # The controller only: `pytest-xdist`'s workers share one cache
        # directory, and several processes writing one JSON file is a corrupt
        # file rather than a faster session.
        cache.set(CACHE_KEY, snapshot())


def distributed(config: pytest.Config) -> bool:
    """Whether this session runs under `pytest-xdist`.

    Worker: ``workerinput``; controller: ``numprocesses``; ``--dist`` sets neither.
    """
    if hasattr(config, "workerinput"):
        return True
    if getattr(config.option, "numprocesses", None):
        return True
    return str(config.getoption("dist", "no")) != "no"


#: What a run collecting a scale-marked critical test is told. The refusal is
#: at collection because the conflict is one of selection, not of a result.
EARLY_GATE_CONFLICT = (
    "a test gates early and is also scale-marked out of the per-PR tier, so "
    "-m critical runs it and a bare pytest does not (tests/_durations.py, "
    "issue #635):"
)


#: What a benchmark reaching a distributed run is told. Read by the guard's
#: own test, so the message and the check cannot drift apart.
DISTRIBUTED_BENCHMARK = (
    "a benchmark reached a pytest-xdist run, where pytest-benchmark disables "
    "itself and measures nothing: run the benchmarks in their own serial "
    "invocation, with no -n (DEV.md, issue #405)"
)


#: What a capped run under xdist is told. The durations reach the workers and
#: the cap is read on the controller, so the cap would pass vacuously.
DISTRIBUTED_CAP = (
    "SAL_DURATION_CAP or SAL_KEY_DURATION_CAP is set on a pytest-xdist run, "
    "where the durations are recorded in the workers and the cap is checked "
    "on the controller: every test would pass the cap. Run the capped "
    "selection serially, with no -n (DEV.md, issue #405)"
)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Fail a benchmark that reached a distributed run.

    Keyed on the ``benchmark`` fixture; at setup, as collection errors are INTERNALERROR.
    """
    # `fixturenames` belongs to the item types that take fixtures, not to
    # `Item`, and this hook is given the base type.
    fixtures: tuple[str, ...] = getattr(item, "fixturenames", ())
    if "benchmark" in fixtures and distributed(item.config):
        pytest.fail(DISTRIBUTED_BENCHMARK, pytrace=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],  # noqa: ARG001 -- the hook's signature
) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if report.when == "call":
        markers = frozenset(marker.name for marker in item.iter_markers())
        item.config.stash[DURATIONS].append((report.nodeid, report.duration, markers))


def pytest_sessionfinish(
    session: pytest.Session,
    exitstatus: int,  # noqa: ARG001 -- the hook's signature
) -> None:
    cap = os.environ.get("SAL_DURATION_CAP")
    key_cap = os.environ.get("SAL_KEY_DURATION_CAP")
    if cap is None and key_cap is None:
        return
    if distributed(session.config) and not hasattr(session.config, "workerinput"):
        # Durations are recorded in workers, so a distributed controller sees
        # none; `infra/validate.sh` runs serially for this.
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        assert reporter is not None
        reporter.write_sep("=", "duration cap under pytest-xdist", red=True)
        reporter.write_line(DISTRIBUTED_CAP)
        session.exitstatus = 1
        return
    recorded: list[tuple[str, float, frozenset[str]]] = session.config.stash[DURATIONS]
    offenders = [] if cap is None else over_cap(recorded, float(cap))
    if key_cap is not None:
        offenders += key_over_cap(recorded, float(key_cap))
    if offenders:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        assert reporter is not None
        reporter.write_sep("=", "tests over the duration cap", red=True)
        for line in offenders:
            reporter.write_line(line)
        session.exitstatus = 1
