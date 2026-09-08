"""Reinforcement learning over discrete search problems.

Infrastructure, not application: nothing here knows what a tree is, and a
test asserts it. The interface is :class:`~snakes_and_ladders.learn.environment.Environment`
and the reference instance is a Potts landscape, exactly as ``snakes_and_ladders.opt``
pairs :class:`~snakes_and_ladders.opt.objective.Objective` with a Potts chain and an HMM.

Root ``CLAUDE.md`` re-exports nothing from a package's top level, so import
submodule contents explicitly.
"""
