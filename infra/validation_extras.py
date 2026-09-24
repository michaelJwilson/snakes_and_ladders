"""The `validation-*` extras `pyproject.toml` declares, as `uv sync` flags (issue #972).

CI's `validation` job installs every framework the package is validated
against and nothing else it does not need. Written out in the workflow, the
list would drift from `pyproject.toml` on the next framework's ticket; read
from it, the job follows the file.

    uv sync --locked --extra test $(python3 infra/validation_extras.py)

Standard library only, so it runs before any environment exists.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from _paths import REPO_ROOT

#: What every framework's extra is called: this prefix and one name.
PREFIX = "validation-"


def validation_extras(pyproject: Path = REPO_ROOT / "pyproject.toml") -> list[str]:
    """The `validation-*` extras, in the order the file declares them."""
    with pyproject.open("rb") as handle:
        declared = tomllib.load(handle)["project"].get("optional-dependencies", {})
    return [name for name in declared if name.startswith(PREFIX)]


def main() -> None:
    print(" ".join(f"--extra {name}" for name in validation_extras()))


if __name__ == "__main__":
    main()
