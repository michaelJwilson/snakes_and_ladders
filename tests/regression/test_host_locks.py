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
def test_wide_raises_the_thread_count_to_the_cores_it_owns(scratch: Path) -> None:
    # The exception, for measuring something that is itself parallel: rayon in
    # the Rust backend, a torch intra-op reduction, the CPU parallelism of
    # #344. One thread would measure the wrong thing there.
    cores = subprocess.run(
        ["nproc"], capture_output=True, text=True, check=True
    ).stdout.strip()
    finished = subprocess.run(
        [
            "bash",
            "-c",
            f". {LOCKS}; with_lock measure --wide -- bash -c "
            f'"echo \\$OMP_NUM_THREADS \\$RAYON_NUM_THREADS \\$SAL_MEASURE_THREADS"',
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
    assert finished.stdout.split() == [cores, cores, cores]


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_a_validation_may_not_take_the_whole_host(scratch: Path) -> None:
    # A validation shares the host with two siblings, so it may not claim the
    # cores they are using. Refusing beats quietly oversubscribing, which would
    # slow all three and show up as nothing but noise in a later measurement.
    finished = subprocess.run(
        ["bash", "-c", f". {LOCKS}; with_lock validate --wide -- true"],
        cwd=REPO_ROOT,
        env={"SAL_SCRATCH": str(scratch), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode != 0
    assert "does not own the host" in finished.stderr


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.critical
@pytest.mark.structural
def test_wide_does_not_forfeit_exclusivity(scratch: Path) -> None:
    # The obvious way to get --wide wrong is to treat it as a separate, weaker
    # path. A wide measurement still takes every lock.
    elapsed = _run(["validate", "measure --wide"], scratch)
    assert elapsed >= 1.5 * JOB, (
        f"a wide measurement ran beside a validation: {elapsed:.2f}s"
    )


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock is not on this host")
@pytest.mark.structural
def test_measure_works_from_a_copy_without_the_execute_bit(
    scratch: Path, tmp_path: Path
) -> None:
    # The script is meant to be sourced, so it must not require its own execute
    # bit to re-enter itself. A copy extracted with `git show`, or a checkout
    # that dropped the mode, failed `measure` with a bare "Permission denied"
    # -- found by hitting it, which is why this is pinned.
    copy = tmp_path / "locks_copy.sh"
    copy.write_text(LOCKS.read_text())
    copy.chmod(0o644)
    finished = subprocess.run(
        ["bash", "-c", f". {copy}; with_lock measure -- true"],
        cwd=REPO_ROOT,
        env={"SAL_SCRATCH": str(scratch), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
