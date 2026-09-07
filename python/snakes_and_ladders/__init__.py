"""Top-level package: ``import snakes_and_ladders as sal``.

Re-exports only the package's own utilities, not submodule contents. A
submodule is reached by attribute -- ``sal.likelihood.pruning`` -- and is
imported on first access, so the top-level import brings in nothing else
(root ``CLAUDE.md``, issue #301).
"""

from snakes_and_ladders._lazy import submodule_getattr

from .oxi_snakes_and_ladders import double

__all__ = ["double"]
__getattr__ = submodule_getattr(__name__, __path__)
