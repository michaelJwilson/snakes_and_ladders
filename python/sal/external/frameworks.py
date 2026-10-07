"""The external frameworks the package drives, by name (issues #972, #1282).

:data:`FRAMEWORKS` is the registry: one :class:`Framework` per
``validation-*`` extra, and one for HiGHS, which SciPy, a core dependency,
carries with no extra; each names the module its script imports, so the
guard knows which imports to refuse outside ``validation/scripts/``, and the
licence its answers carry. The adapter, the script, the test module and the
extra share the framework's name, which need not be the module it imports:
PyMaxflow imports as ``maxflow``. #1282 moved the registry here from
``sal.validation``, which re-exports it, so a solver's
:class:`~sal.external.solvers.Provenance` reads it without importing the
validation home.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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
    #: Whether every part of :attr:`licence` is OSI-approved: gco-v3.0's
    #: research-use terms are not, so gco's answers say so (#1282).
    osi: bool
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
            osi=True,
            source="https://github.com/pmneila/PyMaxflow",
            ticket=973,
        ),
        Framework(
            name="gco",
            distribution="gco-wrapper",
            module="gco",
            licence="MIT (wrapper); gco-v3.0 research-use",
            osi=False,
            source="https://github.com/Borda/pyGCO",
            ticket=974,
        ),
        Framework(
            name="hmmlearn",
            distribution="hmmlearn",
            module="hmmlearn",
            licence="BSD-3-Clause",
            osi=True,
            source="https://github.com/hmmlearn/hmmlearn",
            ticket=975,
        ),
        Framework(
            name="scikit_learn",
            distribution="scikit-learn",
            module="sklearn",
            licence="BSD-3-Clause",
            osi=True,
            source="https://github.com/scikit-learn/scikit-learn",
            ticket=975,
        ),
        Framework(
            name="blackjax",
            distribution="blackjax",
            module="blackjax",
            licence="Apache-2.0",
            osi=True,
            source="https://github.com/blackjax-devs/blackjax",
            ticket=963,
        ),
        Framework(
            name="rustworkx",
            distribution="rustworkx",
            module="rustworkx",
            licence="Apache-2.0",
            osi=True,
            source="https://github.com/Qiskit/rustworkx",
            ticket=976,
        ),
        Framework(
            name="gymnasium",
            distribution="gymnasium",
            module="gymnasium",
            licence="MIT",
            osi=True,
            source="https://github.com/Farama-Foundation/Gymnasium",
            ticket=977,
        ),
        Framework(
            name="torchrl",
            distribution="torchrl",
            module="torchrl",
            licence="MIT",
            osi=True,
            source="https://github.com/pytorch/rl",
            ticket=977,
        ),
        Framework(
            name="torch_geometric",
            distribution="torch_geometric",
            module="torch_geometric",
            licence="MIT",
            osi=True,
            source="https://github.com/pyg-team/pytorch_geometric",
            ticket=977,
        ),
        Framework(
            name="jax",
            distribution="jax",
            module="jax",
            licence="Apache-2.0",
            osi=True,
            source="https://github.com/jax-ml/jax",
            ticket=991,
        ),
        # HiGHS ships inside SciPy, a core dependency, as
        # `linprog(method="highs")`: no extra installs it (#1063).
        Framework(
            name="highs",
            distribution="scipy",
            module="scipy",
            licence="MIT (HiGHS); BSD-3-Clause (SciPy)",
            osi=True,
            source="https://github.com/ERGO-Code/HiGHS",
            ticket=1063,
        ),
    )
}
