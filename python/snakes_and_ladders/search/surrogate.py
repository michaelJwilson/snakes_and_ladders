"""Deprecated: moved to :mod:`snakes_and_ladders.learn.ranking` (issue #779).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/learn/test_learn_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.learn.ranking import (
    LatticeTarget,
    LearnedTreeSurrogate,
    TreeTarget,
    enumerated_log_partition_target,
    fixed_length_target,
    ground_state_offset,
    ground_state_target,
    lattice_examples,
    lattice_instances,
    maximized_target,
    mean_field_offset,
    plug_in_offset,
    shuffle_children,
    strip_log_partition_target,
    tree_examples,
    tree_node_tokens,
)

warnings.warn(
    "snakes_and_ladders.search.surrogate moved to "
    "snakes_and_ladders.learn.ranking (issue #779); this shim is removed in "
    "the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LatticeTarget",
    "LearnedTreeSurrogate",
    "TreeTarget",
    "enumerated_log_partition_target",
    "fixed_length_target",
    "ground_state_offset",
    "ground_state_target",
    "lattice_examples",
    "lattice_instances",
    "maximized_target",
    "mean_field_offset",
    "plug_in_offset",
    "shuffle_children",
    "strip_log_partition_target",
    "tree_examples",
    "tree_node_tokens",
]
