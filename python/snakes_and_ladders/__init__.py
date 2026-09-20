"""The package's surface: the seams it declares, reachable from the root.

``import snakes_and_ladders as sal`` (issue #301) handed a user 150 modules and
one exported name, a placeholder. What the root exports now is what the
package is built on and nothing more (issue #717): the four protocols and
contracts a submodule implements or consumes --- :class:`Objective`,
:class:`Environment`, :class:`FactorGraph` and :class:`PottsGraph` --- the
fixture registry every instance in the suite is declared through, the
backend a caller names a kernel by, and the parallel map a loop of
independent bodies runs on. Everything else is imported from the submodule
that owns it, as before, or read as written --- ``sal.sim.tree`` --- since
every package resolves its submodules on first use too
(:func:`_submodules`).

Each name is resolved on first use rather than at import: ``learn`` imports
``torch``, and a caller reaching for a graph or a fixture pays for neither.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The packages are resolved by `_submodule` below; they are named here so
    # `mypy` reads `sal.sim.tree` as the module it is, not as an attribute.
    from snakes_and_ladders import (  # noqa: F401
        learn,
        likelihood,
        opt,
        parallel,
        qa,
        sandbox,
        search,
        sim,
    )
    from snakes_and_ladders.backend import Backend
    from snakes_and_ladders.fixtures import Params, load_params
    from snakes_and_ladders.learn.environment import Environment
    from snakes_and_ladders.opt.objective import Objective
    from snakes_and_ladders.sim.factor_graph import FactorGraph
    from snakes_and_ladders.sim.fixtures import fixture
    from snakes_and_ladders.sim.graph import PottsGraph
    from snakes_and_ladders.sim.simulator import Simulator

__all__ = [
    "Backend",
    "Environment",
    "FactorGraph",
    "Objective",
    "Params",
    "PottsGraph",
    "Simulator",
    "fixture",
    "load_params",
    "parallel",
]

#: Where each exported name lives; ``None`` exports the module itself.
_SURFACE: dict[str, tuple[str, str | None]] = {
    "Backend": ("snakes_and_ladders.backend", "Backend"),
    "Environment": ("snakes_and_ladders.learn.environment", "Environment"),
    "FactorGraph": ("snakes_and_ladders.sim.factor_graph", "FactorGraph"),
    "Objective": ("snakes_and_ladders.opt.objective", "Objective"),
    "Params": ("snakes_and_ladders.fixtures", "Params"),
    "PottsGraph": ("snakes_and_ladders.sim.graph", "PottsGraph"),
    "Simulator": ("snakes_and_ladders.sim.simulator", "Simulator"),
    "fixture": ("snakes_and_ladders.sim.fixtures", "fixture"),
    "load_params": ("snakes_and_ladders.fixtures", "load_params"),
    "parallel": ("snakes_and_ladders.parallel", None),
}


def _submodules(package: str) -> Callable[[str], ModuleType]:
    """A module ``__getattr__`` that imports ``package.<name>`` on first use.

    Python imports a subpackage only when something imports it: after
    ``import snakes_and_ladders as sal``, ``sal.sim`` is an ``AttributeError``
    until ``snakes_and_ladders.sim`` has been imported somewhere. Every
    package here installs this as its ``__getattr__``, so ``sal.sim.tree``
    resolves on first use and costs ``sim.tree`` alone; a name that is no
    submodule raises ``AttributeError`` as an ordinary attribute would.
    """

    def __getattr__(name: str) -> ModuleType:
        if (
            name.startswith("_")
            or importlib.util.find_spec(f"{package}.{name}") is None
        ):
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        module = importlib.import_module(f"{package}.{name}")
        setattr(importlib.import_module(package), name, module)
        return module

    return __getattr__


#: The subpackages and root modules, resolved the same way as the seams.
_submodule = _submodules(__name__)


def __getattr__(name: str) -> Any:
    """Resolve an exported name or a submodule on first use, then cache it."""
    try:
        module_name, attribute = _SURFACE[name]
    except KeyError:
        return _submodule(name)
    module = importlib.import_module(module_name)
    value: Any = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
