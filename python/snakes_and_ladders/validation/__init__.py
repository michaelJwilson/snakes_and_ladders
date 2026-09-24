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
    #: ``tests/validation/test_<name>.py``, and the extra ``validation-<name>``
    #: with each underscore a hyphen.
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
        """The ``pyproject.toml`` extra that installs it, hyphenated as PEP 685 spells it."""
        return f"validation-{self.name.replace('_', '-')}"


#: Every framework the package is validated or benchmarked against, by name.
FRAMEWORKS: Mapping[str, Framework] = {
    framework.name: framework
    for framework in (
        Framework(
            name="pymaxflow",
            distribution="PyMaxflow",
            module="maxflow",
            licence="GPL-3.0",
            source="https://github.com/pmneila/PyMaxflow",
            ticket=973,
        ),
        Framework(
            name="gco",
            distribution="gco-wrapper",
            module="gco",
            licence="MIT (wrapper); gco-v3.0 research-use",
            source="https://github.com/Borda/pyGCO",
            ticket=974,
        ),
        Framework(
            name="hmmlearn",
            distribution="hmmlearn",
            module="hmmlearn",
            licence="BSD-3-Clause",
            source="https://github.com/hmmlearn/hmmlearn",
            ticket=975,
        ),
        Framework(
            name="scikit_learn",
            distribution="scikit-learn",
            module="sklearn",
            licence="BSD-3-Clause",
            source="https://github.com/scikit-learn/scikit-learn",
            ticket=975,
        ),
        Framework(
            name="blackjax",
            distribution="blackjax",
            module="blackjax",
            licence="Apache-2.0",
            source="https://github.com/blackjax-devs/blackjax",
            ticket=963,
        ),
        Framework(
            name="rustworkx",
            distribution="rustworkx",
            module="rustworkx",
            licence="Apache-2.0",
            source="https://github.com/Qiskit/rustworkx",
            ticket=976,
        ),
        Framework(
            name="gymnasium",
            distribution="gymnasium",
            module="gymnasium",
            licence="MIT",
            source="https://github.com/Farama-Foundation/Gymnasium",
            ticket=977,
        ),
        Framework(
            name="torchrl",
            distribution="torchrl",
            module="torchrl",
            licence="MIT",
            source="https://github.com/pytorch/rl",
            ticket=977,
        ),
        Framework(
            name="torch_geometric",
            distribution="torch_geometric",
            module="torch_geometric",
            licence="MIT",
            source="https://github.com/pyg-team/pytorch_geometric",
            ticket=977,
        ),
        Framework(
            name="jax",
            distribution="jax",
            module="jax",
            licence="Apache-2.0",
            source="https://github.com/jax-ml/jax",
            ticket=991,
        ),
    )
}
