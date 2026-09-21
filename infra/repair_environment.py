"""Restore the shared environment additively, because `uv sync` cannot.

Issue #556. On this host one `.venv` is a real directory in the main clone and
a symlink in every other worktree, so what one command does to it, every
worktree gets. Two different commands damage it, and the obvious repair for
the first *is* the second:

``uv sync`` **without** ``--all-extras`` uninstalls every package outside the
extras it was given. It is how ``ruff``, ``mypy``, ``sphinx``, ``nbformat`` and
the whole ``frameworks`` extra left this environment at once.

``uv sync`` **with** ``--all-extras`` puts them back --- and reinstalls the
project as an editable path, which is a single mutable pointer the other
worktrees then follow into a stranger's working copy. That is the fault
``tests/regression/test_environment.py`` exists to catch, so telling a reader
to run it to fix the first fault trades one for the other.

This does neither. It reads what ``pyproject.toml`` declares --- core
dependencies and every optional group --- asks the running interpreter which of
them are absent, and installs only those, through ``uv pip install``, which
adds distributions and removes none. The project itself is never named, so the
editable pointer is untouched whichever worktree this is run from.

Run it with the shared environment's own interpreter, which is the one it
repairs::

    /home/user/snakes_and_ladders/.venv/bin/python infra/repair_environment.py

``--dry-run`` names what is missing and exits 1 without installing, which is
what the guard in ``tests/regression/test_environment.py`` reports.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from _paths import REPO_ROOT

#: Where a PEP 508 requirement stops naming a distribution and starts
#: constraining it. `name`, `name>=1.2`, `name[extra]`, `name @ url` and
#: `name; marker` all begin with the name and then one of these.
_CONSTRAINT_START = "[<>=!~;@ "


def requirement_name(spec: str) -> str:
    """The distribution a requirement string names, without its constraint."""
    cuts = [spec.index(char) for char in _CONSTRAINT_START if char in spec]

    return spec[: min(cuts)].strip() if cuts else spec.strip()


def declared(pyproject: Path) -> list[str]:
    """Every requirement the project declares: core, then each extra.

    Order follows the file, so a dry run reads in the order a maintainer
    wrote them rather than in an arbitrary one.
    """
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    specs: list[str] = list(project.get("dependencies", []))

    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)

    return specs


def absent(specs: list[str]) -> list[str]:
    """Those of ``specs`` the running interpreter cannot find installed.

    Asked of the distribution rather than of an import, because the two
    differ often enough to need a table otherwise --- `pyyaml` imports as
    `yaml`, and `pre-commit` and `pip-audit` are commands that import as
    nothing at all.
    """
    missing: list[str] = []

    for spec in specs:
        try:
            distribution(requirement_name(spec))
        except PackageNotFoundError:
            missing.append(spec)

    return missing


def main(argv: list[str] | None = None) -> int:
    """Report what is missing, and install it unless ``--dry-run``."""
    parser = argparse.ArgumentParser(description="Restore missing declared packages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="name what is missing and exit 1 without installing it",
    )
    arguments = parser.parse_args(argv)

    specs = declared(REPO_ROOT / "pyproject.toml")
    missing = absent(specs)

    if not missing:
        print(f"{sys.executable}: all {len(specs)} declared requirements present")
        return 0

    print(f"{len(missing)} of {len(specs)} declared requirements are missing:")
    for spec in missing:
        print(f"  {spec}")

    if arguments.dry_run:
        return 1

    # `uv pip install`, never `uv sync`: this adds distributions and removes
    # none, and it never names the project, so the editable pointer every
    # other worktree follows is left where it is.
    completed = subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, *missing],
        check=False,
    )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
