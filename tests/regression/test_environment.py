"""The shared environment is reported as an environment fault, not as a test failure.

Issue #556. One `.venv` is symlinked into every worktree on the development
host, so a `uv sync` with a narrower extra set strips packages *for everyone*.
The damage then surfaces wherever the missing package is first imported --- at
filing, a missing `gymnasium` appeared as an error inside
`tests/regression/docs/test_seams_survey.py`, which reads as a defect in the
branch under review rather than as a broken environment.

These two tests exist to be the first thing that fails, and to say why.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    # In the `frameworks` extra, which CI's `python-tests` job installs and
    # `search/gym.py` imports at module scope. Its absence surfaced as an
    # error inside an unrelated test, which is the case this list exists for.
    "gymnasium",
)


@pytest.mark.structural
def test_the_environment_carries_every_package_the_repository_assumes() -> None:
    missing = [name for name in EXPECTED if importlib.util.find_spec(name) is None]

    assert missing == [], (
        f"the shared environment is incomplete: {missing}. This is an "
        f"environment fault, not a defect in the branch under test — a narrower "
        f"`uv sync` strips packages for every worktree sharing the `.venv`. "
        f"Re-sync with `uv sync --locked --all-extras` (issue #556)."
    )


@pytest.mark.structural
def test_the_package_under_test_is_this_worktrees_own() -> None:
    """The import resolves inside this tree, not a sibling's working copy.

    `infra/new_worktree.sh` checks this at creation, with `PYTHONPATH` set. It
    is checked again here because the fault appears *later*, when a sibling
    repoints the shared editable install, and at that point no creation-time
    check can see it.
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
