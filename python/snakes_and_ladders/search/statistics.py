"""Deprecated: moved to :mod:`snakes_and_ladders.sample.statistics` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/sample/test_sample_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
    sign_test_p_value,
)

warnings.warn(
    "snakes_and_ladders.search.statistics moved to snakes_and_ladders.sample.statistics "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "chi_square_p_value",
    "integrated_autocorrelation_time",
    "sign_test_p_value",
]
