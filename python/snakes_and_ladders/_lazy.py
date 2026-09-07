"""Lazy attribute access to a package's submodules, so ``sal.sim.graph`` resolves.

``import snakes_and_ladders as sal`` is the stated way to refer to the package
(root ``CLAUDE.md``, issue #301). A package attribute that is a submodule only
exists once that submodule has been imported, so without this ``sal.sim``
would raise until someone imported ``sal.sim`` by its full
name. :func:`submodule_getattr` gives a package a module ``__getattr__``
(PEP 562) that imports a submodule on first access and nothing before: the
top-level import stays as cheap as it was, and no symbol is re-exported --
only the modules the package already contains become reachable by attribute.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from types import ModuleType


def submodule_getattr(package: str, path: list[str]) -> Callable[[str], ModuleType]:
    """Build a package's ``__getattr__`` that imports its submodules on demand.

    Parameters
    ----------
    package : str
        The package's ``__name__``.
    path : list[str]
        The package's ``__path__``, searched for the submodules it may serve.

    Returns
    -------
    Callable[[str], ModuleType]
        Raises ``AttributeError`` naming the package for anything that is not
        one of its submodules, so a typo fails the way it always did.
    """
    names = frozenset(info.name for info in pkgutil.iter_modules(path))

    def __getattr__(name: str) -> ModuleType:
        if name in names:
            return importlib.import_module(f"{package}.{name}")
        msg = f"module {package!r} has no attribute {name!r}"
        raise AttributeError(msg)

    return __getattr__
