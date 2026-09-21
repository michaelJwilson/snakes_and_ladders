"""The repository root, derived once for everything under ``tests/``.

Thirty-five test modules each wrote ``Path(__file__).resolve().parents[n]``
with ``n`` counted from where the file sits, so moving a module between
``tests/regression/`` and a subdirectory of it silently rooted it one level
up (issue #863, design-audit row R20).

Derived here rather than imported from `infra/_paths.py`: a test reads the
tree it is a part of, and the two are held to one value by
`tests/regression/test_duplication_guards.py`.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root: the parent of this directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
