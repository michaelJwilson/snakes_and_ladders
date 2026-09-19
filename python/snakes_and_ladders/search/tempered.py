"""Deprecated: moved to :mod:`snakes_and_ladders.sample.tempered` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name the new one defines; the objects are the same objects,
which ``tests/regression/sample/test_sample_deprecated_paths.py`` asserts
with ``is``. The shim is removed in the release after 0.3.0, whose number
issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.tempered import (
    S,
    TemperedEnsemble,
    adapt_ladder_round_trips,
    round_trip_time,
    round_trips,
    tempered_factor_graph,
    tempered_potts_pair,
    tempered_topologies,
    up_fraction,
)

warnings.warn(
    "snakes_and_ladders.search.tempered moved to snakes_and_ladders.sample.tempered "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "S",
    "TemperedEnsemble",
    "adapt_ladder_round_trips",
    "round_trip_time",
    "round_trips",
    "tempered_factor_graph",
    "tempered_potts_pair",
    "tempered_topologies",
    "up_fraction",
]
