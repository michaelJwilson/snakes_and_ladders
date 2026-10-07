"""The validation home: external frameworks as referees, each in a subprocess (issue #972).

`sandbox` keeps the package's own replaced implementations as referees; this
package drives other people's. Each framework the package is checked or timed
against has one adapter module here, one script under ``scripts/`` and one
``validation-<framework>`` extra in ``pyproject.toml``, or for a framework on
no package index a build script (OpenGM's ``infra/build_opengm.sh``, #1279). The adapter writes a
problem to an ``.npz``, runs the script under the same interpreter
(:func:`sal.external.runner.run`), and reads the answer back
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

:data:`FRAMEWORKS` is the registry, one :class:`Registration` per
:class:`Framework`; all three live in :mod:`sal.external.frameworks` since
#1282 (#1304 named the entry :class:`Registration` and the key
:class:`Framework`) and are re-exported here. The runner and the file protocol moved to :mod:`sal.external` with
them; ``sal.validation.runner`` and ``sal.validation.protocol`` re-export
them for one release.
"""

from __future__ import annotations

from sal import _submodules
from sal.external.frameworks import FRAMEWORKS, Framework, Registration

__getattr__ = _submodules(__name__)

__all__ = ["FRAMEWORKS", "Framework", "Registration"]
