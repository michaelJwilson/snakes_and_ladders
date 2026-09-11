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

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import pytest  # noqa: E402

from tests._durations import key_over_cap, over_cap  # noqa: E402

DURATIONS = pytest.StashKey[list[tuple[str, float, frozenset[str]]]]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[DURATIONS] = []


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
