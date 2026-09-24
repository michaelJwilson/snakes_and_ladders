"""The shared environment is reported as an environment fault, not as a test failure.

Issue #556. One `.venv` is symlinked into every worktree on the development
host, so a `uv sync` with a narrower extra set strips packages *for everyone*.
The damage then surfaces wherever the missing package is first imported --- at
filing, a missing `gymnasium` appeared as an error inside
`tests/regression/docs/test_seams_survey.py`, which reads as a defect in the
branch under review rather than as a broken environment.

These tests exist to be the first thing that fails, and to say why.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import repair_environment

from tests._paths import REPO_ROOT

#: Every tool the agent briefs, `DEV.md` and `infra/*.sh` assume is present.
#: `uv sync --locked --all-extras` installs all of them; a narrower sync does
#: not, and that is the failure this names.
EXPECTED = (
    "numpy",
    "torch",
    "numba",
    "pytest",
    "yaml",
    "matplotlib",
    "networkx",
    "scipy",
)


@pytest.mark.infra
def test_the_environment_carries_every_package_the_repository_assumes() -> None:
    missing = [name for name in EXPECTED if importlib.util.find_spec(name) is None]

    assert missing == [], (
        f"the shared environment is incomplete: {missing}. This is an "
        f"environment fault, not a defect in the branch under test — a narrower "
        f"`uv sync` strips packages for every worktree sharing the `.venv`. "
        f"Repair it with `infra/repair_environment.py` (issue #556)."
    )


@pytest.mark.infra
def test_a_shared_environment_carries_every_declared_requirement() -> None:
    """Nothing declared is missing from an environment the worktrees share.

    One `uv sync` removed `gymnasium`, `ruff`, `mypy` and 13 more; needs a symlinked `.venv`.
    """
    if not (REPO_ROOT / ".venv").is_symlink():
        pytest.skip("not a shared environment: CI installs one extra set per job")

    specs = repair_environment.declared(REPO_ROOT / "pyproject.toml")
    missing = repair_environment.absent(specs)

    assert missing == [], (
        f"{len(missing)} of {len(specs)} declared requirements are missing from "
        f"the environment every worktree on this host shares: {missing}. Repair "
        f"it with `infra/repair_environment.py`, which installs what is absent "
        f"and nothing else. Do NOT run `uv sync --all-extras`: it would restore "
        f"these and reinstall the project as an editable path, which is the "
        f"fault the next test catches (issue #556)."
    )


@pytest.mark.infra
def test_the_package_under_test_is_this_worktrees_own() -> None:
    """The import resolves inside this tree, not a sibling's working copy.

    Rechecked after `infra/new_worktree.sh`: a sibling can repoint the install later.
    """
    import snakes_and_ladders

    resolved = Path(snakes_and_ladders.__file__).resolve()

    assert resolved.is_relative_to(REPO_ROOT), (
        f"the tests import {resolved}, which is outside {REPO_ROOT}: this "
        f"session is testing another worktree's code. Set "
        f"PYTHONPATH={REPO_ROOT / 'python'} (issue #556). "
        f"PYTHONPATH is currently {os.environ.get('PYTHONPATH') or 'unset'}, "
        f"and sys.path[0] is {sys.path[0]!r}."
    )
