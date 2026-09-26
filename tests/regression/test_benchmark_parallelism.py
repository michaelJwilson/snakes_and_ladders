"""What keeps `pytest-xdist` and `pytest-benchmark` from silently cancelling each other.

Issue #405. `pytest-benchmark` turns itself off in a distributed run without
failing, so a run under `-n` passes the 212 benchmark tests with no timing
taken. The correctness tiers run under `-n 3` without `tests/benchmarks`, the
benchmarks serially; the guard in `tests/conftest.py` catches the next `-n`
added to the wrong command. The duration cap is pinned too: it is recorded in
the workers and read on the controller, whose stash is empty.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib

import pytest

from tests._paths import REPO_ROOT
from tests.conftest import DISTRIBUTED_BENCHMARK, DISTRIBUTED_CAP

BENCHMARKS = REPO_ROOT / "tests" / "benchmarks"

#: One benchmark module, cheap to reach and never reached: the guard fires at
#: setup, so the workers start and nothing is timed.
SUBJECT = BENCHMARKS / "test_numerics_rust_bench.py"


def _distributed_pytest(
    *arguments: str, **environment: str
) -> subprocess.CompletedProcess[bytes]:
    """Run a distributed pytest in a subprocess, rooted at the repository.

    Skips without `pytest-xdist` (the `test` extra, which CI installs).
    """
    import os

    pytest.importorskip("xdist")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        env={**os.environ, **environment},
    )


@pytest.mark.infra
@pytest.mark.critical
def test_a_benchmark_under_xdist_fails_instead_of_measuring_nothing() -> None:
    """The guard fires, and says why, rather than passing 212 empty timings."""
    assert SUBJECT.is_file(), "the guard's subject moved; name another benchmark"
    result = _distributed_pytest(str(SUBJECT.relative_to(REPO_ROOT)), "-n", "2")

    assert result.returncode != 0, result.stdout.decode()
    assert DISTRIBUTED_BENCHMARK in result.stdout.decode()


@pytest.mark.infra
@pytest.mark.release  # 11.6 s in the tier, over the 10 s cap (#1088)
def test_a_capped_run_under_xdist_fails_instead_of_passing_every_cap() -> None:
    """The cap is read where the durations are not, so a capped run refuses `-n`."""
    result = _distributed_pytest(
        "tests/regression/test_numerics.py", "-n", "2", SAL_DURATION_CAP="10"
    )

    assert result.returncode != 0, result.stdout.decode()
    assert DISTRIBUTED_CAP in result.stdout.decode()


@pytest.mark.infra
@pytest.mark.critical
def test_ci_distributes_the_correctness_tier_and_leaves_the_benchmarks_serial() -> None:
    """The workflow's two invocations, and which of them may carry `-n`.

    Read off `.github/workflows/ci.yml`, so the arrangement cannot drift.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    step = workflow.split("- name: Run the selected tests", 1)[1].split("\n  docs:")[0]

    # The two commands, each with its shell continuations joined, so a flag
    # on a wrapped line counts as being on the command it belongs to.
    correctness, benchmarks = (
        " ".join(line.split())
        for line in step.replace("\\\n", " ").splitlines()
        if "uv run pytest" in line
    )
    assert "-n 3" in correctness, correctness
    assert "--ignore=tests/benchmarks" in correctness, correctness
    assert "-n" not in benchmarks.replace("--durations", ""), benchmarks
    assert "$benchmarks" in benchmarks, benchmarks


@pytest.mark.infra
def test_the_test_extra_carries_the_distribution_plugin() -> None:
    """`pytest-xdist` is a test dependency, not something a runner happens to have."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    extra = pyproject["project"]["optional-dependencies"]["test"]

    assert any(requirement.startswith("pytest-xdist") for requirement in extra), extra
