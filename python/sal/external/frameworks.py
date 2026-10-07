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

**A source build (#1279).** OpenGM is on no package index: its entry names
the script that builds it (:attr:`Framework.build`) and the shared library
the build writes, and has no extra. :func:`built` finds that library under
:func:`built_home`, and :func:`built_version` reads the commit the build
pinned, so neither loads the library.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
    #: The script that builds it, where no package index carries it: then
    #: :attr:`module` is the shared library the build writes, without its
    #: ``lib`` prefix and suffix, and no extra installs it (#1279).
    build: str | None = None

    @property
    def extra(self) -> str:
        """The ``pyproject.toml`` extra that installs it, hyphenated as PEP 685 spells it."""
        return f"validation-{self.name.replace('_', '-')}"

    @property
    def remedy(self) -> str:
        """What makes it available: its extra installed, or its build run."""
        if self.build is not None:
            return f"run `{self.build}`"
        return f"install the `{self.extra}` extra"


def built_home(framework: Framework) -> Path:
    """The directory ``framework``'s build writes, as its build script resolves it.

    ``$SAL_<NAME>_HOME`` where set, else ``<cache>/sal/<name>`` with
    ``<cache>`` ``$XDG_CACHE_HOME`` or ``~/.cache``.
    """
    variable = f"SAL_{framework.name.upper()}_HOME"
    if variable in os.environ:
        return Path(os.environ[variable])
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "sal" / framework.name


def built(framework: Framework) -> Path | None:
    """The shared library ``framework``'s build wrote, or ``None`` where it has not run."""
    if framework.build is None:
        return None
    library = built_home(framework) / f"lib{framework.module}.so"
    return library if library.is_file() else None


def built_version(framework: Framework) -> str:
    """The commit ``framework``'s build pinned, as it recorded it beside the library."""
    return (built_home(framework) / "COMMIT").read_text().strip()


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
        # A source build: the header-only inference, compiled with sal's own
        # entry point, `infra/opengm/sal_opengm.cxx`; none of the research-only
        # externals OpenGM's CMake downloads is fetched (#1279).
        Framework(
            name="opengm",
            distribution="OpenGM",
            module="sal_opengm",
            licence="MIT",
            osi=True,
            source="https://github.com/opengm/opengm",
            ticket=1279,
            build="infra/build_opengm.sh",
        ),
    )
}
