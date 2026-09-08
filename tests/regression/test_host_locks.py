"""The two host locks behave as ``infra/locks.sh`` documents (issue #418).

The script is a throughput mechanism, so what wants asserting is the trade it
makes rather than that it runs: validations overlap, a measurement does not
overlap anything, and the slot count bounds the overlap. A lock whose
exclusivity silently lapsed would make every timing in a pull request
incomparable, and nothing else in the suite would notice.

Timings here are ratios between a sleep and a wall clock on the same host, not
budgets: each asserts that N one-second jobs took about one second or about
two, with margins wide enough that a loaded host does not fail them.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS = REPO_ROOT / "infra" / "locks.sh"

#: One job's duration. Long enough that process start-up is a small fraction of
#: it, short enough that the whole module stays well inside the duration cap.
JOB = 1.0


def _run(kinds: list[str], scratch: Path) -> float:
    """Start one job per entry in ``kinds`` at once; return the wall seconds."""
    script = " ".join(f"( with_lock {kind} -- sleep {JOB} ) &" for kind in kinds)
    command = f". {LOCKS}; {script} wait"
    start = time.monotonic()
    finished = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env={
            "SAL_SCRATCH": str(scratch),
            "PATH": "/usr/bin:/bin",
            "SAL_LOCK_WAIT": "60",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - start
    assert finished.returncode == 0, finished.stderr
    return elapsed


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_validations_up_to_the_slot_count_overlap(scratch: Path) -> None:
    # The point of the split: three correctness runs cost one job's wall time,
    # not three. Under the old exclusive lock this took 3 * JOB.
    elapsed = _run(["validate"] * 3, scratch)
    assert elapsed < 2 * JOB, f"three validations serialized: {elapsed:.2f}s"


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_a_fourth_validation_waits_for_a_slot(scratch: Path) -> None:
    # The slot count is a real bound, not advisory: the fourth job waits, so
    # the host is never oversubscribed past what its cores can carry.
    elapsed = _run(["validate"] * 4, scratch)
    assert elapsed >= 1.5 * JOB, f"the fourth validation did not wait: {elapsed:.2f}s"


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.critical
@pytest.mark.structural
def test_a_measurement_never_overlaps_a_validation(scratch: Path) -> None:
    # The rule the split exists to preserve. A measurement runs alone, so a
    # wall time quoted in a pull request is comparable to the budgets in
    # DEV.md. This is `critical` because its failure invalidates every timing
    # taken after it, which is exactly what the marker is for.
    elapsed = _run(["validate", "measure"], scratch)
    assert elapsed >= 1.5 * JOB, (
        f"a measurement ran beside a validation: {elapsed:.2f}s"
    )


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.critical
@pytest.mark.structural
def test_two_measurements_never_overlap(scratch: Path) -> None:
    # The older guarantee, kept: exclusivity among measurements themselves.
    elapsed = _run(["measure", "measure"], scratch)
    assert elapsed >= 1.5 * JOB, f"two measurements overlapped: {elapsed:.2f}s"


@pytest.mark.structural
def test_an_unknown_kind_is_refused(scratch: Path) -> None:
    # A typo must not run the command unlocked, which would be the silent
    # failure: the job would finish and its number would be wrong.
    finished = subprocess.run(
        ["bash", "-c", f". {LOCKS}; with_lock buidl -- true"],
        cwd=REPO_ROOT,
        env={"SAL_SCRATCH": str(scratch), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode != 0
    assert "unknown kind" in finished.stderr
