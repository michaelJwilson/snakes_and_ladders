"""Move sets, samplers and exact solvers. See CLAUDE.md in this directory."""

from sal import _submodules

__getattr__ = _submodules(__name__)
