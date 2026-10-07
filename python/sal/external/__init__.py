"""External solvers behind one subprocess path (issue #1282).

The package compares its solvers with other people's: gco, PyMaxflow, HiGHS,
hmmlearn and BlackJAX. Every one of them runs in a subprocess, through
:func:`sal.external.runner.run` and the ``.npz`` file protocol of
:mod:`sal.external.protocol`; no file here imports a framework, so GPL and
research-only terms, a framework's native library and its threads stay out
of the package process.

:mod:`sal.external.solvers` names what can be called: a :class:`Solver` per
framework method, the :class:`Capability` set each declares, and the
:class:`Provenance` each answer carries, read from
:data:`~sal.external.frameworks.FRAMEWORKS`. A call outside a solver's
capabilities is refused before any subprocess starts; an absent framework
raises :class:`ExternalUnavailable` naming its extra.

:func:`session` keeps one worker per solver for many calls, so a call pays
interpreter and import start-up once (:mod:`sal.external.sessions`); its
:class:`Transport` moves arrays as memory-mapped files (the default) or
``.npz`` files, with the same bytes reaching the framework each way.

:func:`ground_state` returns a Potts ground state from gco or PyMaxflow as
:func:`sal.search.ground_state.ground_state` returns one
(:mod:`sal.external.potts`). It is imported on first use, so a worker that
imports this package does not pay for the search ladder.

``sal.validation`` drives the same frameworks as referees for the test suite
and imports its runner and registry from here; nothing here imports
``sal.validation``. ``CLAUDE.md`` in this directory states the rules and
``tests/regression/test_external.py`` asserts them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from sal import _submodules
from sal.external.runner import ScriptError
from sal.external.sessions import Session, session
from sal.external.solvers import (
    Capability,
    CapabilityRefused,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    provenance,
    require,
)
from sal.external.transport import Transport

if TYPE_CHECKING:
    from sal.external.potts import ExternalRun, ground_state

#: Names imported on first use, and the module each comes from.
_LAZY = {"ExternalRun": "sal.external.potts", "ground_state": "sal.external.potts"}

_submodule = _submodules(__name__)


def __getattr__(name: str) -> Any:
    """Resolve a :data:`_LAZY` name or a submodule on first use, then cache it."""
    if name not in _LAZY:
        return _submodule(name)
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


__all__ = [
    "Capability",
    "CapabilityRefused",
    "ExternalRun",
    "ExternalUnavailable",
    "Provenance",
    "ScriptError",
    "Session",
    "Solver",
    "Transport",
    "available",
    "ground_state",
    "invoke",
    "provenance",
    "require",
    "session",
]
