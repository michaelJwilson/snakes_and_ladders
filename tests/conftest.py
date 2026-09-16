"""Suite-wide configuration: one process is one core, slow tests are named, and no benchmark runs distributed.

Three things; the first two are issue #372's and the third is issue #405's.

**One process is one core.** The BLAS behind NumPy and PyTorch starts a
thread per core, so one test process reads as four on the load average and
four of them saturate a 4-core host. The three thread-count variables are set
to one here, before NumPy is imported, unless the caller set them. The figure
renderers are exempt on purpose: ``snakes_and_ladders.qa.build`` strips these
from the render environment, since a committed figure is rendered the way the
manifest renders it.

**A slow test carries its marker.** Every test's call duration is recorded;
with ``SAL_DURATION_CAP`` set, a test over it that carries none of the
out-of-tier markers fails the session, and with ``SAL_KEY_DURATION_CAP`` set,
so does a ``key`` test over that (``tests/_durations.py``).
``infra/validate.sh`` sets both caps on the reference host; CI does not, per
`DEV.md`'s rule against timing on its runners, and prints ``--durations``
instead.

**A test is selectable by the problem it exercises.** Issue #614. Every test
module's fixture calls are read at collection (``tests/_problems.py``) and one
marker per problem named is added to its items, so ``pytest -m potts_lattice``
selects that problem across `search/` and `likelihood/` at once and no author
tags anything. The names are registered below with the rest, so
``--strict-markers`` refuses a typo, and they are a third axis: ``-m
"potts_lattice and critical"`` is the intersection.

**A benchmark is never collected under ``pytest-xdist``.** ``pytest-benchmark``
disables itself whenever a run is distributed and says so in a line of its
header, so a distributed run collects the 212 benchmark tests, passes every
one of them and measures nothing --- a green suite whose timings do not exist.
The tiers are split instead (``DEV.md``): the correctness tiers run under
``-n 3`` with ``tests/benchmarks`` excluded, benchmarks run serially in their
own invocation. That split is what makes the check below unreachable, which is
why it is worth making: the next ``-n`` added to the wrong command fails here
by name rather than quietly stopping the measurement.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import pytest  # noqa: E402

from tests._durations import key_over_cap, over_cap  # noqa: E402
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
    # A fixture directory named `critical`, `key`, `stress` or `release` would
    # register a second marker of that name and then be applied by the hook to
    # every module that loads it --- and `tests/_durations.py` reads
    # `iter_markers()` to decide what the duration cap exempts, so a directory
    # name would start excusing tests from it. Refuse rather than collide.
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
    """Mark each item with the problems its module loads.

    ``tryfirst`` because `pytest`'s own mark plugin deselects on ``-m`` from
    this same hook: a marker added after that runs has been added to an item
    already thrown away, and ``-m potts_lattice`` would select nothing while
    looking like it worked. The order is pinned by
    ``tests/regression/test_problem_markers.py``, which runs the selection
    rather than trusting registration order.

    What each module names is written back to `pytest`'s cache here, so the
    next invocation reads 232 file stats rather than 232 parses.
    """
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        for problem in fixtures_named_in(path):
            item.add_marker(problem)
    cache = getattr(config, "cache", None)
    if cache is not None and not hasattr(config, "workerinput"):
        # The controller only: `pytest-xdist`'s workers share one cache
        # directory, and several processes writing one JSON file is a corrupt
        # file rather than a faster session.
        cache.set(CACHE_KEY, snapshot())


def distributed(config: pytest.Config) -> bool:
    """Whether this session is running under `pytest-xdist`.

    Read three ways because the answer differs by process and by how the run
    was asked for: a worker carries ``workerinput``, a controller started with
    ``-n`` carries ``numprocesses``, and ``--dist`` alone sets neither.

    Returns
    -------
    bool
        True if the session is distributed.
    """
    if hasattr(config, "workerinput"):
        return True
    if getattr(config.option, "numprocesses", None):
        return True
    return str(config.getoption("dist", "no")) != "no"


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

    `pytest-benchmark` turns itself off under `pytest-xdist` rather than
    failing, so the benchmarks would pass while measuring nothing. A test is
    a benchmark if it asks for the ``benchmark`` fixture, which is what that
    plugin keys on too --- not its directory, so a benchmark written anywhere
    is caught. Failing at setup rather than refusing the collection is what
    makes the message readable: a worker that raises during collection is an
    xdist ``INTERNALERROR`` and the reason does not survive it.
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
        # The second thing a distributed run turns off without saying so. The
        # durations above are recorded by `pytest_runtest_makereport`, which
        # runs in the workers, so the controller's stash is empty and every
        # test is under every cap. `infra/validate.sh` sets both caps and runs
        # the selection, so it runs serially and this is what holds it there.
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
