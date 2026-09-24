"""The validation home: external frameworks as referees, each in a subprocess (issue #972).

`sandbox` keeps the package's own replaced implementations as referees; this
package drives other people's. Each framework the package is checked or timed
against has one adapter module here, one script under ``scripts/`` and one
``validation-<framework>`` extra in ``pyproject.toml``. The adapter writes a
problem to an ``.npz``, runs the script under the same interpreter
(:func:`snakes_and_ladders.validation.runner.run`), and reads the answer back
with the wall time the script measured around the framework's own call.

**No framework is imported into the package process, whatever its licence.**
The script is the only file that imports it. That keeps GPL and research-only
terms out of the process under test (#126, #952), keeps a framework's native
library, threads and second autodiff stack apart from it (#963), and puts
every framework behind one code path.

Only ``tests/`` imports from here, never ``sim``, ``likelihood``, ``opt``,
``search`` or ``learn``, and nothing is re-exported from the package root.
``CLAUDE.md`` in this directory states the rules and
``tests/regression/test_validation.py`` asserts them.

:data:`FRAMEWORKS` is the registry: one :class:`Framework` per
``validation-*`` extra, naming the module its script imports, so the guard
knows which imports to refuse outside ``scripts/``. The adapter, the script,
the test module and the extra share the framework's name, which need not be
the module it imports: PyMaxflow imports as ``maxflow``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from snakes_and_ladders import _submodules

__getattr__ = _submodules(__name__)


@dataclass(frozen=True)
class Framework:
    """One external framework: its extra, what its script imports, where it comes from."""

    #: The adapter's name: ``validation/<name>.py``, ``scripts/<name>.py`` and
    #: ``tests/validation/test_<name>.py``, and the extra ``validation-<name>``.
    name: str
    #: The distribution the extra names, as PyPI spells it.
    distribution: str
    #: The top-level module its script imports; refused everywhere else.
    module: str
    #: The licence, as the distribution declares it.
    licence: str
    #: Its source repository.
    source: str
    #: The ticket that approved it.
    ticket: int

    @property
    def extra(self) -> str:
        """The ``pyproject.toml`` extra that installs it."""
        return f"validation-{self.name}"


#: Every framework the package is validated or benchmarked against, by name.
FRAMEWORKS: Mapping[str, Framework] = {}
