"""What `infra/bin/uv` must refuse, and what it must leave alone.

Issue #615. `.venv` is a symlink in every worktree, so a narrow `uv sync`
uninstalls extras for all of them. Each test builds a refused or a passed
situation. A stub `uv` earlier on `PATH` records its arguments; the real
command is the fault under repair and is never run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

GUARD = REPO_ROOT / "infra" / "bin" / "uv"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: A `uv` that installs nothing and reports what it was asked to do, so a
#: pass-through is observable as its argv rather than as a side effect.
STUB = '#!/usr/bin/env bash\nprintf "STUB %s\\n" "$*"\n'


def _project(root: Path, *, shared: bool) -> Path:
    """A project at ``root`` whose ``.venv`` is a symlink (``shared``), or a directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n')

    environment = root / ".venv"
    if shared:
        target = root.parent / "shared-venv"
        (target / "bin").mkdir(parents=True, exist_ok=True)
        environment.symlink_to(target)
    else:
        (environment / "bin").mkdir(parents=True, exist_ok=True)

    return root


def _run(
    project: Path, arguments: list[str], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the guard in ``project`` with a stub `uv` behind it on `PATH`."""
    stub_directory = project.parent / "stub-bin"
    stub_directory.mkdir(parents=True, exist_ok=True)
    stub = stub_directory / "uv"
    stub.write_text(STUB)
    stub.chmod(0o755)

    return subprocess.run(
        [str(GUARD), *arguments],
        cwd=project,
        capture_output=True,
        check=False,
        text=True,
        env={
            "PATH": f"{GUARD.parent}:{stub_directory}:/usr/bin:/bin",
            "HOME": str(project.parent),
            **(environment or {}),
        },
    )


def _ci_sync_commands() -> list[list[str]]:
    """Every `uv sync` line in `.github/workflows/ci.yml`, as argv, read not restated."""
    found = re.findall(r"^\s*(?:-\s*)?run:\s*(uv sync .*)$", WORKFLOW.read_text(), re.M)

    return [command.split() for command in found]


@pytest.mark.infra
def test_a_narrowing_sync_is_refused_where_the_environment_is_shared(
    tmp_path: Path,
) -> None:
    """The fault itself: the command that took 15 of 24 requirements, twice."""
    project = _project(tmp_path / "worktree", shared=True)

    result = _run(project, ["sync", "--locked", "--extra", "test"])

    assert result.returncode == 1
    assert "STUB" not in result.stdout, "the guard refused and still ran uv"
    assert str(project / ".venv") in result.stderr
    assert "repair_environment.py" in result.stderr


@pytest.mark.infra
def test_the_same_sync_is_permitted_where_the_environment_is_its_own(
    tmp_path: Path,
) -> None:
    """A real `.venv` directory has one owner, so nothing is shared to lose."""
    project = _project(tmp_path / "checkout", shared=False)

    result = _run(project, ["sync", "--locked", "--extra", "test"])

    assert result.returncode == 0
    assert result.stdout.strip() == "STUB sync --locked --extra test"


@pytest.mark.infra
def test_every_ci_sync_line_passes_through_a_real_environment(
    tmp_path: Path,
) -> None:
    """CI builds its own directory per job; the guard must be invisible to it.

    Each job's narrow extra set is correct there and refused on a symlink.
    """
    project = _project(tmp_path / "runner", shared=False)
    commands = _ci_sync_commands()

    assert len(commands) >= 4, f"no uv sync lines found in {WORKFLOW}"

    for command in commands:
        result = _run(project, command[1:])

        assert result.returncode == 0, command
        assert result.stdout.strip() == f"STUB {' '.join(command[1:])}"


@pytest.mark.infra
def test_a_syncing_run_is_refused_and_a_non_syncing_one_is_not(
    tmp_path: Path,
) -> None:
    """`uv run` syncs unless told not to, and `UV_NO_SYNC=1` is what tells it.

    `infra/new_worktree.sh` relies on that command (`DEV.md`).
    """
    project = _project(tmp_path / "worktree", shared=True)

    refused = _run(project, ["run", "python", "-c", "0"])
    permitted = _run(project, ["run", "python", "-c", "0"], {"UV_NO_SYNC": "1"})

    assert refused.returncode == 1
    assert "STUB" not in refused.stdout
    assert permitted.returncode == 0
    assert permitted.stdout.strip() == "STUB run python -c 0"


@pytest.mark.infra
@pytest.mark.parametrize(
    "arguments",
    [["--version"], ["pip", "list"], ["lock", "--check"], ["pip", "install", "sync"]],
)
def test_a_command_that_does_not_write_the_environment_passes_through(
    tmp_path: Path, arguments: list[str]
) -> None:
    """Including `uv pip install sync`, where `sync` is a package and not the
    subcommand: the guard reads the first non-option token, so an argument
    that merely spells one of the refused names is not one."""
    project = _project(tmp_path / "worktree", shared=True)

    result = _run(project, arguments)

    assert result.returncode == 0
    assert result.stdout.strip() == f"STUB {' '.join(arguments)}"


@pytest.mark.infra
def test_an_explicit_private_environment_is_permitted_from_a_shared_worktree(
    tmp_path: Path,
) -> None:
    """`UV_PROJECT_ENVIRONMENT` is the escape the refusal names, so it works."""
    project = _project(tmp_path / "worktree", shared=True)
    private = tmp_path / "mine"
    (private / "bin").mkdir(parents=True)

    result = _run(project, ["sync"], {"UV_PROJECT_ENVIRONMENT": str(private)})

    assert result.returncode == 0
    assert result.stdout.strip() == "STUB sync"


@pytest.mark.infra
def test_new_worktree_emits_the_path_that_installs_the_guard() -> None:
    """The guard reaches only shells that export it, so the script that makes
    a worktree is the one place that must not stop saying so (issue #404)."""
    script = (REPO_ROOT / "infra" / "new_worktree.sh").read_text()

    assert r'echo "export PATH=$main/infra/bin:\$PATH"' in script
