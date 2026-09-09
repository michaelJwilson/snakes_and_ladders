"""The host lock behaves as ``infra/locks.sh`` documents (issues #418, #427).

The script is a throughput mechanism, so what wants asserting is the trade it
makes rather than that it runs: validations overlap, a measurement does not
overlap anything, the slot count bounds the overlap, and a measurement that is
waiting holds none of it. A lock whose exclusivity silently lapsed would make
every timing in a pull request incomparable, and nothing else in the suite
would notice.

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


def _stamps(kinds: list[str], scratch: Path, marks: Path) -> list[tuple[float, float]]:
    """Run the jobs at once; return each one's (start, end) from its own clock.

    Wall time alone cannot tell "these overlapped" from "the host was busy":
    on a loaded host three concurrent jobs take longer than one, and process
    start-up costs grow. So each job stamps its own start and end, and the
    assertions below are about the intervals rather than the total. This test
    failed twice at load 9-11 when it asserted a total.
    """
    marks.mkdir(exist_ok=True)
    jobs = " ".join(
        f'( with_lock {kind} -- bash -c \'printf "%s " "$EPOCHREALTIME" '
        f'>> "{marks}/$$"; sleep {JOB}; printf "%s" "$EPOCHREALTIME" '
        f'>> "{marks}/$$"\' ) &'
        for kind in kinds
    )
    command = f". {LOCKS}; {jobs} wait"
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
    assert finished.returncode == 0, finished.stderr
    spans = []
    for mark in marks.iterdir():
        start, end = mark.read_text().split()
        spans.append((float(start), float(end)))
    assert len(spans) == len(kinds), f"{len(spans)} jobs stamped, expected {len(kinds)}"
    return spans


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
def test_validations_up_to_the_slot_count_overlap(
    scratch: Path, tmp_path: Path
) -> None:
    # The point of the split: three correctness runs share the host rather than
    # queueing. Asserted as genuine overlap -- all three inside the lock at one
    # instant -- rather than as a total, which a loaded host inflates past any
    # fixed bound. Under the old exclusive lock the spans are disjoint and the
    # latest start falls after the earliest end.
    spans = _stamps(["validate"] * 3, scratch, tmp_path / "marks")
    latest_start = max(start for start, _ in spans)
    earliest_end = min(end for _, end in spans)
    assert latest_start < earliest_end, (
        f"three validations did not overlap: latest start {latest_start:.3f} "
        f"is after earliest end {earliest_end:.3f}"
    )


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


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_a_measurement_is_one_thread_unless_it_asks_otherwise(scratch: Path) -> None:
    # Holding the machine is not using it. The default stays at whatever the
    # caller set -- one thread, per tests/conftest.py -- because every baseline
    # in STATUS.md and DEV.md was taken that way and a four-thread number is
    # not comparable with any of them.
    finished = subprocess.run(
        [
            "bash",
            "-c",
            f'. {LOCKS}; with_lock measure -- bash -c "echo \\$OMP_NUM_THREADS"',
        ],
        cwd=REPO_ROOT,
        env={
            "SAL_SCRATCH": str(scratch),
            "PATH": "/usr/bin:/bin",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "1"


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_a_waiting_measurement_holds_no_validation_capacity(scratch: Path) -> None:
    # Issue #427's inversion. A measurement that has not started must hold
    # nothing, so a slot it is waiting for stays available to a validation.
    #
    # The arrangement is what exposes it: a short validation takes the first
    # slot and two long ones take the rest, so when the short one ends there is
    # one free slot with the other two still busy. A measurement that took the
    # exclusive lock first and then collected slots one at a time would take
    # that free slot and block on the next, and the probe validation below --
    # which needs one slot and nothing else -- would wait behind a job that is
    # itself waiting. Measured at 3.31 s on the implementation this replaces,
    # against 0.01 s here.
    setup = (
        f". {LOCKS}; "
        f"( with_lock validate -- sleep {0.4 * JOB} ) & sleep 0.15; "
        f"( with_lock validate -- sleep {3 * JOB} ) & sleep 0.15; "
        f"( with_lock validate -- sleep {3 * JOB} ) & "
        f"sleep 0.5; ( with_lock measure -- sleep 0.1 ) & wait"
    )
    env = {"SAL_SCRATCH": str(scratch), "PATH": "/usr/bin:/bin", "SAL_LOCK_WAIT": "60"}
    background = subprocess.Popen(
        ["bash", "-c", setup], cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL
    )
    try:
        time.sleep(1.3)  # the short validation has ended; the measurement waits
        start = time.monotonic()
        probe = subprocess.run(
            ["bash", "-c", f". {LOCKS}; with_lock validate -- true"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        waited = time.monotonic() - start
        assert probe.returncode == 0, probe.stderr
    finally:
        background.wait()
    assert waited < JOB, (
        f"a validation waited {waited:.2f}s behind a measurement that had not "
        "started; the free slot was held by the measurement's own wait"
    )
