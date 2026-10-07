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

The problems are namespaced by family, as the package is, and the root
holds only the infrastructure above: :mod:`sal.external.potts` carries the
Potts model's ``ground_state`` and ``lower_bound``, each in its ``search``
sibling's terms, and later families take a submodule of their own beside it
(``hmm``, ``hmc``). A submodule is imported on first use, so a worker that
imports this package does not pay for the search ladder.

``sal.validation`` drives the same frameworks as referees for the test suite
and imports its runner and registry from here; nothing here imports
``sal.validation``. ``CLAUDE.md`` in this directory states the rules and
``tests/regression/test_external.py`` asserts them.
"""

from __future__ import annotations

from typing import Any

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

_submodule = _submodules(__name__)


def __getattr__(name: str) -> Any:
    """Resolve a submodule, such as :mod:`sal.external.potts`, on first use."""
    return _submodule(name)


__all__ = [
    "Capability",
    "CapabilityRefused",
    "ExternalUnavailable",
    "Provenance",
    "ScriptError",
    "Session",
    "Solver",
    "Transport",
    "available",
    "invoke",
    "provenance",
    "require",
    "session",
]
