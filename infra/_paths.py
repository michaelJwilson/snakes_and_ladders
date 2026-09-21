"""The repository root, derived once for everything under ``infra/``.

Nineteen modules here each wrote ``Path(__file__).resolve().parents[1]``, and
two wrote ``parent.parent`` for the same path. A derivation repeated per file
is a derivation that moves when one file moves: a script relocated into
``infra/bin/`` keeps a count that is now wrong, and reads the wrong tree
without failing (issue #863, design-audit row R20).

The tests derive the same path from their own side, in `tests/_paths.py`;
`tests/regression/test_duplication_guards.py` holds the two to one value.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root: the parent of this directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
