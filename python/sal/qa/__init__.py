"""Cross-cutting quality-assurance figures and tables. See CLAUDE.md in this directory."""

from sal import _submodules

__getattr__ = _submodules(__name__)
