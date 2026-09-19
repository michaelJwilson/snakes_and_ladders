"""Deprecated: moved to :mod:`snakes_and_ladders.learn.tree` (issue #779).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/learn/test_learn_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.learn.tree import (
    FeatureSet,
    RewardModel,
    TreeEnvironment,
    exchanged_subtrees,
    standardize,
    with_uniform_branch_lengths,
)

warnings.warn(
    "snakes_and_ladders.search.rl moved to snakes_and_ladders.learn.tree "
    "(issue #779); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FeatureSet",
    "RewardModel",
    "TreeEnvironment",
    "exchanged_subtrees",
    "standardize",
    "with_uniform_branch_lengths",
]
