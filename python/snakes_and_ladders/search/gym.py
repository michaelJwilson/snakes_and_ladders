"""Deprecated: moved to :mod:`snakes_and_ladders.learn.gym` (issue #779).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/learn/test_learn_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.learn.gym import GymnasiumEnvironment, Observation

warnings.warn(
    "snakes_and_ladders.search.gym moved to snakes_and_ladders.learn.gym "
    "(issue #779); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GymnasiumEnvironment", "Observation"]
