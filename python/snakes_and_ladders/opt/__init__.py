"""Continuous parameter fitting by autodiff. See CLAUDE.md in this directory."""

from snakes_and_ladders._lazy import submodule_getattr

__getattr__ = submodule_getattr(__name__, __path__)
