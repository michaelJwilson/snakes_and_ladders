"""Suite-wide configuration: one process is one core, and slow tests are named.

Two things, both from issue #372.

**One process is one core.** The BLAS behind NumPy and PyTorch starts a
thread per core, so one test process reads as four on the load average and
four of them saturate a 4-core host. The three thread-count variables are set
to one here, before NumPy is imported, unless the caller set them. The figure
renderers are exempt on purpose: ``snakes_and_ladders.qa.build`` strips these
from the render environment, since a committed figure is rendered the way the
manifest renders it.

**A slow test carries its marker.** Every test's call duration is recorded;
with ``SAL_DURATION_CAP`` set, a test over it that carries neither
``release`` nor ``stress`` fails the session (``tests/_durations.py``).
``infra/validate.sh`` sets the cap on the reference host; CI does not, per
`DEV.md`'s rule against timing on its runners, and prints ``--durations``
instead.

The duration is collected through ``pytest_runtest_logreport`` and the
markers ride on the report's ``user_properties``, because under
``pytest-xdist`` the process that runs a test is not the process that exits
the session: recording into the worker's own state left the controller with
an empty list and the guard passed a run with two tests 22.5 s and 7.8 s over
a 0.05 s cap (issue #456). ``pytest_runtest_logreport`` is called on the
controller for every worker's report and in process when the run is serial,
so one path serves both, and the cap is applied where the exit status is
decided.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import pytest  # noqa: E402

from tests._durations import over_cap  # noqa: E402

#: Node id, call duration and markers per test, on whichever process decides
#: the exit status: the controller under `pytest-xdist`, the one process
#: otherwise. Module state rather than the config stash, since the hook that
#: fills it is handed a report and no config.
DURATIONS: list[tuple[str, float, frozenset[str]]] = []

#: The report property carrying a test's markers from a worker to the
#: controller. `user_properties` is the one report field xdist serializes for
#: a plugin's own use.
MARKERS = "sal_markers"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],  # noqa: ARG001 -- the hook's signature
) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if report.when == "call":
        markers = sorted(marker.name for marker in item.iter_markers())
        report.user_properties.append((MARKERS, markers))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    markers: frozenset[str] = frozenset()
    for name, value in report.user_properties:
        if name == MARKERS:
            markers = frozenset(value)
    DURATIONS.append((report.nodeid, report.duration, markers))


def pytest_sessionfinish(
    session: pytest.Session,
    exitstatus: int,  # noqa: ARG001 -- the hook's signature
) -> None:
    cap = os.environ.get("SAL_DURATION_CAP")
    if cap is None:
        return
    if hasattr(session.config, "workeroutput"):
        # An xdist worker: it reports its durations to the controller through
        # the hook above, and its exit status is not the run's.
        return
    offenders = over_cap(DURATIONS, float(cap))
    if offenders:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        assert reporter is not None
        reporter.write_sep("=", "tests over the duration cap", red=True)
        for line in offenders:
            reporter.write_line(line)
        session.exitstatus = 1
