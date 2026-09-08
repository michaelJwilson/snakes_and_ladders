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
"""

from __future__ import annotations

import os
from collections.abc import Iterator

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import pytest  # noqa: E402

from tests._durations import over_cap  # noqa: E402

DURATIONS = pytest.StashKey[list[tuple[str, float, frozenset[str]]]]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[DURATIONS] = []


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
    if cap is None:
        return
    recorded: list[tuple[str, float, frozenset[str]]] = session.config.stash[DURATIONS]
    offenders = over_cap(recorded, float(cap))
    if offenders:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        assert reporter is not None
        reporter.write_sep("=", "tests over the duration cap", red=True)
        for line in offenders:
            reporter.write_line(line)
        session.exitstatus = 1
