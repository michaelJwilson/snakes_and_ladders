"""Reinforcement learning over discrete search problems.

The interface is :class:`~sal.learn.environment.Environment`
and the reference instance is a Potts environment, exactly as ``sal.opt``
pairs :class:`~sal.opt.objective.Objective` with a Potts chain and an HMM.
The interface, the policy and the estimators import no application module, and
``CLAUDE.md`` in this directory names the three that do.

Root ``CLAUDE.md`` re-exports nothing from a package's top level, so import
submodule contents explicitly.
"""

from sal import _submodules

__getattr__ = _submodules(__name__)
