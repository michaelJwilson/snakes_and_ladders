"""Samplers, annealers and tempering. See CLAUDE.md in this directory."""

from sal import _submodules

__getattr__ = _submodules(__name__)
