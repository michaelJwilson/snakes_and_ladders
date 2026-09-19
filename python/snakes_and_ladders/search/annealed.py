"""Deprecated: moved to :mod:`snakes_and_ladders.sample.annealed` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name the new one defines; the objects are the same objects,
which ``tests/regression/sample/test_sample_deprecated_paths.py`` asserts
with ``is``. The shim is removed in the release after 0.3.0, whose number
issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.annealed import (
    LogPartition,
    Resampling,
    SimulatedTempered,
    annealed_importance_sampling,
    geometric_betas,
    population_annealing,
    rung_weights,
    simulated_tempering,
)

warnings.warn(
    "snakes_and_ladders.search.annealed moved to snakes_and_ladders.sample.annealed "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LogPartition",
    "Resampling",
    "SimulatedTempered",
    "annealed_importance_sampling",
    "geometric_betas",
    "population_annealing",
    "rung_weights",
    "simulated_tempering",
]
