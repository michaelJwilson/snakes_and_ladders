"""The package's surface: the seams it declares, reachable from the root.

``import snakes_and_ladders as sal`` (issue #301) handed a user 150 modules and
one exported name, a placeholder. What the root exports now is what the
package is built on and nothing more (issue #717): the four protocols and
contracts a submodule implements or consumes --- :class:`Objective`,
:class:`Environment`, :class:`FactorGraph` and :class:`PottsGraph` --- the
fixture registry every instance in the suite is declared through, the
backend a caller names a kernel by, and the parallel map a loop of
independent bodies runs on. Everything else is imported from the submodule
that owns it, as before.

Each name is resolved on first use rather than at import: ``learn`` imports
``torch``, and a caller reaching for a graph or a fixture pays for neither.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from snakes_and_ladders import parallel as parallel
    from snakes_and_ladders.backend import Backend as Backend
    from snakes_and_ladders.learn.environment import Environment as Environment
    from snakes_and_ladders.opt.objective import Objective as Objective
    from snakes_and_ladders.sim.factor_graph import FactorGraph as FactorGraph
    from snakes_and_ladders.sim.fixtures import fixture as fixture
    from snakes_and_ladders.sim.graph import PottsGraph as PottsGraph

__all__ = [
    "Backend",
    "Environment",
    "FactorGraph",
    "Objective",
    "PottsGraph",
    "fixture",
    "parallel",
]

#: Where each exported name lives; ``None`` exports the module itself.
_SURFACE: dict[str, tuple[str, str | None]] = {
    "Backend": ("snakes_and_ladders.backend", "Backend"),
    "Environment": ("snakes_and_ladders.learn.environment", "Environment"),
    "FactorGraph": ("snakes_and_ladders.sim.factor_graph", "FactorGraph"),
    "Objective": ("snakes_and_ladders.opt.objective", "Objective"),
    "PottsGraph": ("snakes_and_ladders.sim.graph", "PottsGraph"),
    "fixture": ("snakes_and_ladders.sim.fixtures", "fixture"),
    "parallel": ("snakes_and_ladders.parallel", None),
}


def __getattr__(name: str) -> Any:
    """Resolve an exported name on first use, then cache it on the module."""
    try:
        module_name, attribute = _SURFACE[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None
    module = importlib.import_module(module_name)
    value: Any = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
