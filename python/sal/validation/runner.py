"""The runner's old import path, kept for one release (issue #1282).

:mod:`sal.external.runner` is the one subprocess path to a framework; this
module re-exports it, as the packages #1010 split re-export what they held,
so ``from sal.validation.runner import run`` still resolves. ``available`` is
:func:`sal.external.runner.installed` under its old name: in
:mod:`sal.external`, ``available`` takes a solver. Import from
:mod:`sal.external.runner`; this module goes with the next release.
"""

from __future__ import annotations

from sal.external.runner import (
    SCRIPTS,
    Run,
    ScriptError,
    installed,
    package,
    run,
)

#: :func:`sal.external.runner.installed`, under the name it had here.
available = installed

__all__ = [
    "SCRIPTS",
    "Run",
    "ScriptError",
    "available",
    "installed",
    "package",
    "run",
]
